#!/usr/bin/env python3
"""RETIRED 2026-08-26 - _share/mapp.tmx is now the authoring map.

This script copied root mapp.tmx OVER _share/mapp.tmx. Running it now would
destroy live work. It refuses to run. Kept only for its MAP/DROP/SPLIT tables,
which document how the original import was wired.
"""
import sys
print("REFUSING TO RUN: _share/mapp.tmx is the authoring map now.",file=sys.stderr)
print("This tool overwrites it from the retired root copy and would lose your edits.",file=sys.stderr)
sys.exit(2)

# ---- everything below is reference only, never executed ----
import os,re,sys,struct,xml.etree.ElementTree as ET
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH=os.path.join(ROOT,"_share")

MAP={  # root tileset source -> path inside _share
 "../8-17/Tile-Sets/Tsx/ISS_Floor_Dirt-64x32.tsx":"Blocker/blocker-mask.tsx",
 "../../Downloads/tileset-water.tsx":"River/water-shallows.tsx",
 "tileset.tsx":"River/water-deep-to-shore.tsx",
 "roa.tsx":"Roads/asphalt-dark.tsx",
 "tileset-crossroad.tsx":"Roads/crossroad.tsx",
 "tileset-wroad.tsx":"Ground/grass-bright.tsx",
 "objects.tsx":"Props/Vehicles/street-objects.tsx",
 "landmarks.tsx":"Props/Object-order/Buildings/building-shell.tsx",
 "new_building.tsx":"Props/Object-order/Above-Trees/trees-and-buildings.tsx",
 "walls.tsx":"Props/Walls/walls.tsx",
 "tileset-stone.tsx":"Ground/stone.tsx",
 "realistic-building.tsx":"Props/Object-order/Buildings/city-buildings.tsx",
 "aaaa.tsx":"_landmarks/red-fort/red-fort.tsx",
 "tileset-white.tsx":"Ground/white-blank.tsx",
 "tileset-dirt.tsx":"Ground/dirt-tan.tsx",
 "tileset-grass.tsx":"Ground/grass.tsx",
 "tileset-pave.tsx":"Roads/paving-sandstone.tsx",
 "Walls-qutubminar.tsx":"_landmarks/qutub-minar/qutub-minar-walls.tsx",
 "tileset-gndndrt.tsx":"Ground/grass-dry-patchy.tsx",
 "tileset-flatgrass.tsx":"Ground/grass-mown.tsx",
 "tileset-tepidgrass.tsx":"Ground/grass-lush.tsx",
 "tileset-roaddirt.tsx":"Ground/dirt-road.tsx",
 "tileset-orange-stone.tsx":"Ground/stone-orange.tsx",
 "tileset-asphalt-road.tsx":"Roads/asphalt-road.tsx",
 "tileset-vijay-road.tsx":"Roads/vijay-road.tsx",
 "tileset-black-tiles.tsx":"Ground/black-tiles.tsx",
 "tileset-shantivan-tile.tsx":"Ground/shantivan-ground.tsx",
 "assets/Park/PARKS.tsx":"Props/Other-Objects/parks.tsx",
}
# unused AND broken in the authoring map - deliberately not carried across
DROP={"gnd.tsx","tileset-sssssss.tsx","new_landmark.tsx","new_objects.tsx",
 "other-side_builiding.tsx","realistic-buildingss.tsx","assets/Courts.tsx",
 "tileset-asphalt_road.tsx",
 "../../../../Volumes/T7/IanneFolder/karachicity/Buildings/Props/props.tsx"}
# one collection tileset is split into per-landmark tilesets, by tile id
SPLIT={"realisitc landdmarks.tsx":{5:"_landmarks/qutub-minar/qutub-minar.tsx",
 9:"_landmarks/humayuns-tomb/humayuns-tomb.tsx",11:"_landmarks/delhi-gate/delhi-gate.tsx"}}

mp=os.path.join(ROOT,"mapp.tmx")
root=ET.parse(mp).getroot()
entries=[];unknown=[]
for e in root.findall("tileset"):
    fg=int(e.get("firstgid")); s=e.get("source")
    if s is None:
        entries.append((fg,None)); continue
    if s in DROP: continue
    if s in SPLIT:
        for tid,dest in sorted(SPLIT[s].items()): entries.append((fg+tid,dest))
        continue
    if s in MAP: entries.append((fg,MAP[s]))
    else: unknown.append((fg,s))
if unknown:
    print("ERROR: authoring map uses tileset(s) with no _share mapping:",file=sys.stderr)
    for fg,s in unknown: print(f"   firstgid={fg}  {s}",file=sys.stderr)
    print("Add them to MAP (or DROP) in tools/sync-share.py, and copy the art into _share.",file=sys.stderr)
    sys.exit(1)
entries.sort(key=lambda x:x[0])  # stable: duplicate firstgids MUST keep file order

# wall aspects read from the real files, so another art swap self-corrects
ASPECT={}
for fg,dest in entries:
    if dest!="Props/Walls/walls.tsx": continue
    t=ET.parse(os.path.join(SH,dest)).getroot()
    for tl in t.findall("tile"):
        im=tl.find("image")
        d=open(os.path.join(SH,os.path.dirname(dest),im.get("source")),'rb').read(26)
        w,h=struct.unpack('>II',d[16:24]); ASPECT[fg+int(tl.get("id"))]=w/h

lines=open(mp,encoding="utf-8").read().split("\n")
first=next(i for i,l in enumerate(lines) if "<tileset" in l)
last =next(i for i,l in enumerate(lines) if "<layer id=" in l)
out=[]
for fg,dest in entries:
    if dest is None:
        out+=[f' <tileset firstgid="{fg}" name="soil-dark" tilewidth="64" tileheight="32" tilecount="1224" columns="51">',
              '  <image source="Ground/soil-dark.png" width="3264" height="768"/>',' </tileset>']
    else: out.append(f' <tileset firstgid="{fg}" source="{dest}"/>')

body="\n".join(lines[last:])
body=re.sub(r'(<group id="28" name=")[^"]*(")',r'\1Blocker\2',body)
F=0x0FFFFFFF; n=0
def fix(m):
    global n
    line=m.group(0); g=re.search(r'gid="(\d+)"',line)
    if not g: return line
    gid=int(g.group(1))&F
    if gid not in ASPECT: return line
    hm=re.search(r'height="([0-9.]+)"',line); wm=re.search(r'width="([0-9.]+)"',line)
    if not hm or not wm: return line
    n+=1
    return line[:wm.start(1)]+str(round(float(hm.group(1))*ASPECT[gid],4))+line[wm.end(1):]
body=re.sub(r'<object\b[^>]*/>',fix,body)
open(os.path.join(SH,"mapp.tmx"),"w",encoding="utf-8").write("\n".join(lines[:first]+out)+"\n"+body)
print(f"synced _share/mapp.tmx  ({len(entries)} tilesets, {n} wall objects rescaled)")
