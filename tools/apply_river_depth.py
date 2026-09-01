#!/usr/bin/env python3
"""Swap the flat four-colour River layer for the depth-graded plates.

The old layer is kept in the file, renamed and hidden, so the change is one
click to undo in Tiled.  Tiles on the River layer that were never part of the
river -- grass, road and blocker stamps that got painted there -- are carried
across to the new layer unchanged.
"""
import os
import re

ROOT = "/Users/withcenter10/Desktop/Delhi-India_map/_share"
MAP = os.path.join(ROOT, "mapp.tmx")
BANDS = ["river-depth-%d" % i for i in range(1, 7)]
MW = MH = 300

# gid ranges of the two hand-painted water sheets in the current map
WATER_RANGES = [(7, 1199), (1232, 2287)]


def is_water(gid):
    return any(a <= gid <= b for a, b in WATER_RANGES)


def main():
    src = open(MAP).read()

    # ---------------------------------------------------------- current layer
    m = re.search(r'( <layer id="3" name="River"[^>]*>\s*<data encoding="csv">\n)'
                  r'(.*?)(\n</data>\n </layer>\n)', src, re.S)
    if not m:
        raise SystemExit("River layer not found")
    old = [[int(c) for c in r.strip().rstrip(',').split(',')]
           for r in m.group(2).strip().split('\n')]

    # ------------------------------------------------------ next free firstgid
    gid = 0
    for t in re.finditer(r'<tileset firstgid="(\d+)" source="([^"]+)"/>', src):
        fg, path = int(t.group(1)), os.path.normpath(os.path.join(ROOT, t.group(2)))
        try:
            n = int(re.search(r'tilecount="(\d+)"', open(path).read()).group(1))
        except Exception:
            n = 0
        gid = max(gid, fg + n)
    for t in re.finditer(r'<tileset firstgid="(\d+)" name="[^"]+"[^>]*tilecount="(\d+)"', src):
        gid = max(gid, int(t.group(1)) + int(t.group(2)))

    # ----------------------------------------------------------- place plates
    new = [[0] * MW for _ in range(MH)]
    entries, placed = [], 0
    for name in BANDS:
        tsx = os.path.join(ROOT, "River", name + ".tsx")
        csv = os.path.join(ROOT, "River", name + ".csv")
        count = int(re.search(r'tilecount="(\d+)"', open(tsx).read()).group(1))
        lines = open(csv).read().strip().split("\n")
        x0, y0, x1, y1 = (int(v) for v in lines[0].split())
        for cy, row in enumerate(lines[1:]):
            for cx, v in enumerate(int(v) for v in row.split(",")):
                if v <= 0:
                    continue
                x, y = x0 + cx, y0 + cy
                if 0 <= x < MW and 0 <= y < MH:
                    new[y][x] = gid + v - 1
                    placed += 1
        entries.append((gid, "River/%s.tsx" % name, count))
        gid += count

    # anything on the old layer that was never river stays exactly as it was
    kept = 0
    for y in range(MH):
        for x in range(MW):
            g = old[y][x]
            if g and not is_water(g):
                new[y][x] = g
                kept += 1

    # ------------------------------------------------------------ rewrite map
    last = list(re.finditer(r' <tileset firstgid="\d+" source="[^"]+"/>\n', src))[-1]
    block = "".join(' <tileset firstgid="%d" source="%s"/>\n' % (f, p)
                    for f, p, _ in entries)
    src = src[:last.end()] + block + src[last.end():]

    # re-find the layer (offsets moved) and rebuild it
    m = re.search(r'( <layer id="3" name=")River("[^>]*>\s*<data encoding="csv">\n)'
                  r'(.*?)(\n</data>\n </layer>\n)', src, re.S)
    head = m.group(1) + "River (flat, replaced)" + m.group(2)
    if 'visible="0"' not in head:
        head = head.replace('name="River (flat, replaced)"',
                            'name="River (flat, replaced)" visible="0"', 1)
    data = "\n".join(",".join(str(v) for v in row) + ("," if y < MH - 1 else "")
                     for y, row in enumerate(new))
    layer = ('%s%s%s'
             ' <layer id="30" name="River" width="300" height="300">\n'
             '  <data encoding="csv">\n%s\n</data>\n </layer>\n'
             % (head, m.group(3), m.group(4), data))
    src = src[:m.start()] + layer + src[m.end():]
    src = src.replace('nextlayerid="30"', 'nextlayerid="31"', 1)

    open(MAP, "w").write(src)
    print("tilesets added:")
    for f, p, c in entries:
        print("  firstgid=%-6d %-34s %d tiles" % (f, p, c))
    print("plate tiles placed: %d   non-river tiles carried over: %d" % (placed, kept))


if __name__ == "__main__":
    main()
