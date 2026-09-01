#!/usr/bin/env python3
"""Regenerate the flat pale sand that sits on the Ground layer beside the river.

Those cells still use the old water-deep-to-shore sheet, where the bank is a
single unshaded colour.  Next to the new depth-graded plates they read as washed
out.  This rebuilds them from the *same* depth pipeline and the same globally
addressed noise, so the Ground sand and the river bank are one continuous
surface rather than two textures meeting at a line.
"""
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/withcenter10/Desktop/Delhi-India_map/_share"
MAP = os.path.join(ROOT, "mapp.tmx")
OUT = os.path.join(ROOT, "River")

# the boxes that contain the pale-sand clusters, with a little slack
BOXES = [(206, 0, 262, 73, "river-sand-1"),
         (226, 85, 238, 99, "river-sand-2")]


def pale_ground_cells():
    """Ground-layer cells that are riverbank sand.

    Matches both the old sheet's unshaded sand band and any cell already
    reskinned with a generated plate, so the script can be re-run after a
    rebuild without finding nothing to do.
    """
    src = open(MAP).read()
    tsets = [(int(m.group(1)), os.path.normpath(os.path.join(ROOT, m.group(2))))
             for m in re.finditer(r'<tileset firstgid="(\d+)" source="([^"]+)"/>', src)]
    m = re.search(r'<layer id="7" name="Ground"[^>]*>\s*<data encoding="csv">(.*?)\n</data>',
                  src, re.S)
    G = np.array([[int(c) for c in r.strip().rstrip(',').split(',')]
                  for r in m.group(1).strip().split('\n')])

    sheet = next(p for f, p in tsets if p.endswith("water-deep-to-shore.tsx"))
    first = next(f for f, p in tsets if p.endswith("water-deep-to-shore.tsx"))
    t = open(sheet).read()
    img = Image.open(os.path.join(os.path.dirname(sheet),
                                  re.search(r'<image source="([^"]+)"', t).group(1))).convert("RGBA")
    cols = int(re.search(r'columns="(\d+)"', t).group(1))
    count = int(re.search(r'tilecount="(\d+)"', t).group(1))

    # ranges already occupied by generated sand plates count as ours too
    mine = []
    for f, p in tsets:
        if os.path.basename(p).startswith("river-sand-"):
            n = int(re.search(r'tilecount="(\d+)"', open(p).read()).group(1))
            mine.append((f, f + n))

    cells = set()
    seen = {}
    for y in range(300):
        for x in range(300):
            g = int(G[y][x])
            if any(a <= g < b for a, b in mine):
                cells.add((x, y))
                continue
            if not (first <= g < first + count):
                continue
            if g not in seen:
                l = g - first
                a = np.asarray(img.crop(((l % cols) * 64, (l // cols) * 32,
                                         (l % cols) * 64 + 64, (l // cols) * 32 + 32)),
                               np.float32)
                op = a[..., 3] > 128
                rgb = a[..., :3][op].mean(0) if op.any() else np.zeros(3)
                seen[g] = rgb.mean() > 185 and (rgb.max() - rgb.min()) < 70
            if seen[g]:
                cells.add((x, y))
    return cells


def main():
    cells = pale_ground_cells()
    print("pale sand cells on Ground layer:", len(cells))
    mask = os.path.join(HERE, ".river-sand-mask.txt")
    with open(mask, "w") as f:
        f.write("".join("%d %d\n" % c for c in sorted(cells)))

    for x0, y0, x1, y1, name in BOXES:
        n = sum(1 for (x, y) in cells if x0 <= x < x1 and y0 <= y < y1)
        print("=== %s  x %d..%d  y %d..%d  (%d cells)" % (name, x0, x1, y0, y1, n))
        subprocess.check_call([sys.executable, os.path.join(HERE, "build_river_depth.py"),
                               str(x0), str(y0), str(x1), str(y1), OUT, name, mask])
    os.remove(mask)
    print("ALL DONE")


if __name__ == "__main__":
    main()
