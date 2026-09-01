#!/usr/bin/env python3
"""Swap the flat pale sand on the Ground layer for the generated bank plates.

Only cells that were drawn with the old sheet's unshaded sand band are touched;
every other Ground tile is left exactly as it was.
"""
import os
import re

import numpy as np
from PIL import Image

ROOT = "/Users/withcenter10/Desktop/Delhi-India_map/_share"
MAP = os.path.join(ROOT, "mapp.tmx")
PLATES = ["river-sand-1", "river-sand-2"]
MW = MH = 300


def main():
    src = open(MAP).read()

    tsets = [(int(m.group(1)), os.path.normpath(os.path.join(ROOT, m.group(2))))
             for m in re.finditer(r'<tileset firstgid="(\d+)" source="([^"]+)"/>', src)]

    # ------------------------------------------- which Ground cells are pale sand
    sheet = next(p for f, p in tsets if p.endswith("water-deep-to-shore.tsx"))
    first = next(f for f, p in tsets if p.endswith("water-deep-to-shore.tsx"))
    t = open(sheet).read()
    img = Image.open(os.path.join(os.path.dirname(sheet),
                                  re.search(r'<image source="([^"]+)"', t).group(1))).convert("RGBA")
    cols = int(re.search(r'columns="(\d+)"', t).group(1))
    count = int(re.search(r'tilecount="(\d+)"', t).group(1))

    is_pale = {}
    for g in range(first, first + count):
        l = g - first
        a = np.asarray(img.crop(((l % cols) * 64, (l // cols) * 32,
                                 (l % cols) * 64 + 64, (l // cols) * 32 + 32)), np.float32)
        op = a[..., 3] > 128
        if not op.any():
            is_pale[g] = False
            continue
        rgb = a[..., :3][op].mean(0)
        is_pale[g] = bool(rgb.mean() > 185 and (rgb.max() - rgb.min()) < 70)

    m = re.search(r'( <layer id="7" name="Ground"[^>]*>\s*<data encoding="csv">\n)'
                  r'(.*?)(\n</data>\n </layer>\n)', src, re.S)
    if not m:
        raise SystemExit("Ground layer not found")
    G = [[int(c) for c in r.strip().rstrip(',').split(',')]
         for r in m.group(2).strip().split('\n')]

    # ------------------------------------------------------ next free firstgid
    gid = 0
    for fg, path in tsets:
        try:
            n = int(re.search(r'tilecount="(\d+)"', open(path).read()).group(1))
        except Exception:
            n = 0
        gid = max(gid, fg + n)
    for t2 in re.finditer(r'<tileset firstgid="(\d+)" name="[^"]+"[^>]*tilecount="(\d+)"', src):
        gid = max(gid, int(t2.group(1)) + int(t2.group(2)))

    # ------------------------------------------------------------- place plates
    entries, swapped, skipped = [], 0, 0
    for name in PLATES:
        tsx = os.path.join(ROOT, "River", name + ".tsx")
        csv = os.path.join(ROOT, "River", name + ".csv")
        n = int(re.search(r'tilecount="(\d+)"', open(tsx).read()).group(1))
        lines = open(csv).read().strip().split("\n")
        x0, y0, x1, y1 = (int(v) for v in lines[0].split())
        for cy, row in enumerate(lines[1:]):
            for cx, v in enumerate(int(v) for v in row.split(",")):
                if v <= 0:
                    continue
                x, y = x0 + cx, y0 + cy
                if not (0 <= x < MW and 0 <= y < MH):
                    continue
                old = G[y][x]
                if is_pale.get(old):            # only replace the flat sand band
                    G[y][x] = gid + v - 1
                    swapped += 1
                else:
                    skipped += 1
        entries.append((gid, "River/%s.tsx" % name, n))
        gid += n

    # -------------------------------------------------------------- rewrite map
    last = list(re.finditer(r' <tileset firstgid="\d+" source="[^"]+"/>\n', src))[-1]
    src = src[:last.end()] + "".join(
        ' <tileset firstgid="%d" source="%s"/>\n' % (f, p) for f, p, _ in entries) + src[last.end():]

    m = re.search(r'( <layer id="7" name="Ground"[^>]*>\s*<data encoding="csv">\n)'
                  r'(.*?)(\n</data>\n </layer>\n)', src, re.S)
    data = "\n".join(",".join(str(v) for v in row) + ("," if y < MH - 1 else "")
                     for y, row in enumerate(G))
    src = src[:m.start()] + m.group(1) + data + m.group(3) + src[m.end():]

    open(MAP, "w").write(src)
    print("tilesets added:")
    for f, p, c in entries:
        print("  firstgid=%-6d %-32s %d tiles" % (f, p, c))
    print("Ground cells reskinned: %d   plate tiles landing on non-sand (left alone): %d"
          % (swapped, skipped))


if __name__ == "__main__":
    main()
