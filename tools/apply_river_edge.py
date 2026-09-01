#!/usr/bin/env python3
"""Add the sand/grass blend overlay as its own layer above Ground.

Nothing existing is modified: the overlay is a new tile layer inserted directly
after Ground, so it draws over the join and hiding it restores the old look.
Re-running replaces the previous overlay layer rather than stacking a second.
"""
import os
import re

ROOT = "/Users/withcenter10/Desktop/Delhi-India_map/_share"
MAP = os.path.join(ROOT, "mapp.tmx")
PLATES = ["river-edge-%d" % i for i in range(1, 7)]
LAYER_NAME = "River Edge"
MW = MH = 300


def main():
    src = open(MAP).read()

    # drop a previous run's layer so this stays idempotent
    src, n_old = re.subn(r' <layer id="\d+" name="%s"[^>]*>\s*<data encoding="csv">.*?\n</data>\n </layer>\n'
                         % re.escape(LAYER_NAME), '', src, flags=re.S)
    src = re.sub(r' <tileset firstgid="\d+" source="River/river-edge-\d+\.tsx"/>\n', '', src)
    if n_old:
        print("replaced an existing %r layer" % LAYER_NAME)

    # ------------------------------------------------------ next free firstgid
    gid = 0
    for m in re.finditer(r'<tileset firstgid="(\d+)" source="([^"]+)"/>', src):
        p = os.path.normpath(os.path.join(ROOT, m.group(2)))
        try:
            n = int(re.search(r'tilecount="(\d+)"', open(p).read()).group(1))
        except Exception:
            n = 0
        gid = max(gid, int(m.group(1)) + n)
    for m in re.finditer(r'<tileset firstgid="(\d+)" name="[^"]+"[^>]*tilecount="(\d+)"', src):
        gid = max(gid, int(m.group(1)) + int(m.group(2)))

    # --------------------------------------------------------------- placement
    cells = [[0] * MW for _ in range(MH)]
    entries, placed = [], 0
    for name in PLATES:
        tsx = os.path.join(ROOT, "River", name + ".tsx")
        csv = os.path.join(ROOT, "River", name + ".csv")
        if not os.path.exists(tsx):
            print("  skipping %s (not built)" % name)
            continue
        n = int(re.search(r'tilecount="(\d+)"', open(tsx).read()).group(1))
        lines = open(csv).read().strip().split("\n")
        x0, y0, x1, y1 = (int(v) for v in lines[0].split())
        for cy, row in enumerate(lines[1:]):
            for cx, v in enumerate(int(t) for t in row.split(",")):
                if v <= 0:
                    continue
                x, y = x0 + cx, y0 + cy
                if 0 <= x < MW and 0 <= y < MH and not cells[y][x]:
                    cells[y][x] = gid + v - 1
                    placed += 1
        entries.append((gid, "River/%s.tsx" % name, n))
        gid += n

    # ------------------------------------------------------------- rewrite map
    last = list(re.finditer(r' <tileset firstgid="\d+" source="[^"]+"/>\n', src))[-1]
    src = src[:last.end()] + "".join(
        ' <tileset firstgid="%d" source="%s"/>\n' % (f, p) for f, p, _ in entries) + src[last.end():]

    m = re.search(r' <layer id="7" name="Ground"[^>]*>\s*<data encoding="csv">\n.*?\n</data>\n </layer>\n',
                  src, re.S)
    if not m:
        raise SystemExit("Ground layer not found")
    lid = max(int(v) for v in re.findall(r'<layer id="(\d+)"', src) +
              re.findall(r'<objectgroup id="(\d+)"', src) +
              re.findall(r'<group id="(\d+)"', src)) + 1
    data = "\n".join(",".join(str(v) for v in row) + ("," if y < MH - 1 else "")
                     for y, row in enumerate(cells))
    layer = (' <layer id="%d" name="%s" width="300" height="300">\n'
             '  <data encoding="csv">\n%s\n</data>\n </layer>\n' % (lid, LAYER_NAME, data))
    src = src[:m.end()] + layer + src[m.end():]
    src = re.sub(r'nextlayerid="\d+"', 'nextlayerid="%d"' % (lid + 1), src, count=1)

    open(MAP, "w").write(src)
    print("tilesets added:")
    for f, p, c in entries:
        print("  firstgid=%-6d %-32s %d tiles" % (f, p, c))
    print("overlay layer id=%d, %d tiles placed" % (lid, placed))


if __name__ == "__main__":
    main()
