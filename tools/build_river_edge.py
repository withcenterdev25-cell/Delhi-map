#!/usr/bin/env python3
"""Dissolve the sand/grass boundary along the river with an overlay layer.

The bank plates and the grass tiles each look right on their own, but they meet
on a tile diamond, so the join is a staircase.  Rather than repaint either one,
this builds a translucent overlay that sits above Ground and reworks only the
contact: the boundary is pushed off the grid by the same globally addressed
noise the river uses, sand reaches into the grass in fingers, and grass breaks
up into clumps and tufts as it runs out onto the sand.

Because it is its own layer, hiding it restores the previous look exactly.

Usage:  build_river_edge.py X0 Y0 X1 Y1 OUTDIR NAME
"""
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_river_depth import (CTX, P, TW, TH, SS, PAD, MARGIN,
                               fbm, sstep, lerp3, blur, bilinear)

ROOT = "/Users/withcenter10/Desktop/Delhi-India_map/_share"
MAP = os.path.join(ROOT, "mapp.tmx")

# grass sampled off the sheets that actually border the river
G_DARK, G_MID, G_LIGHT = (30, 66, 2), (78, 117, 16), (134, 172, 58)
S_DRY, S_MID = (228, 212, 190), (206, 186, 158)


def surface_classes():
    """Per cell: 1 = river-plate sand, 2 = grass, 3 = plate water, 0 = other.

    Ground is drawn after River, so a Ground tile hides the River tile below it.
    """
    src = open(MAP).read()
    tsets = []
    for m in re.finditer(r'<tileset firstgid="(\d+)" source="([^"]+)"/>', src):
        p = os.path.normpath(os.path.join(ROOT, m.group(2)))
        t = open(p).read()
        tsets.append((int(m.group(1)), int(re.search(r'tilecount="(\d+)"', t).group(1)),
                      os.path.basename(p), p, t))
    for m in re.finditer(r'<tileset firstgid="(\d+)" name="([^"]+)" tilewidth="\d+" '
                         r'tileheight="\d+" tilecount="(\d+)" columns="(\d+)">\s*'
                         r'<image source="([^"]+)"', src):
        tsets.append((int(m.group(1)), int(m.group(3)), m.group(2),
                      os.path.join(ROOT, m.group(5)), None))

    cache = {}

    def sheet_and_rgb(gid):
        gid &= 0x1FFFFFFF
        b = max((t for t in tsets if gid >= t[0]), key=lambda t: t[0], default=None)
        if not b or gid >= b[0] + b[1]:
            return None, None
        fg, n, nm, path, txt = b
        if nm not in cache:
            if txt is None:
                cache[nm] = (Image.open(path).convert("RGBA"), 51, 64, 32)
            else:
                im = re.search(r'<image source="([^"]+)"', txt).group(1)
                cache[nm] = (Image.open(os.path.join(os.path.dirname(path), im)).convert("RGBA"),
                             int(re.search(r'columns="(\d+)"', txt).group(1)),
                             int(re.search(r'tilewidth="(\d+)"', txt).group(1)),
                             int(re.search(r'tileheight="(\d+)"', txt).group(1)))
        img, cols, tw, th = cache[nm]
        l = gid - fg
        a = np.asarray(img.crop(((l % cols) * tw, (l // cols) * th,
                                 (l % cols) * tw + tw, (l // cols) * th + th)), np.float32)
        op = a[..., 3] > 128
        return nm, (a[..., :3][op].mean(0) if op.any() else None)

    def layer(lid):
        m = re.search(r'<layer id="%s" name="[^"]*"[^>]*>\s*<data encoding="csv">(.*?)\n</data>'
                      % lid, src, re.S)
        return np.array([[int(c) for c in r.strip().rstrip(',').split(',')]
                         for r in m.group(1).strip().split('\n')])

    VIS = np.where(layer('7') > 0, layer('7'), layer('30'))
    CLS = np.zeros(VIS.shape, np.uint8)
    GRASS = ("grass.tsx", "grass-lush.tsx", "grass-mown.tsx",
             "grass-bright.tsx", "grass-dry-patchy.tsx")
    for g in np.unique(VIS):
        if not g:
            continue
        nm, rgb = sheet_and_rgb(int(g))
        c = 0
        if nm in GRASS:
            c = 2
        elif nm and (nm.startswith("river-depth") or nm.startswith("river-sand")) and rgb is not None:
            r, gr, b = rgb
            c = 3 if (b > r * 1.4 and b > 110) else 1
        if c:
            CLS[VIS == g] = c
    return CLS


def build(x0, y0, x1, y1, outdir, name, CLS):
    mx0, my0 = max(0, x0 - MARGIN), max(0, y0 - MARGIN)
    mx1, my1 = min(300, x1 + MARGIN), min(300, y1 + MARGIN)
    cut_x, cut_y = x0 - mx0, y0 - my0
    Wc, Hc = mx1 - mx0, my1 - my0
    W, H = Wc * P, Hc * P
    shape = (H, W)
    CTX["ox"], CTX["oy"] = mx0 * P, my0 * P

    sub = CLS[my0:my1, mx0:mx1]
    river = ((sub == 1) | (sub == 3)).astype(np.float32)   # the river side
    grass = (sub == 2).astype(np.float32)

    up = lambda a: np.asarray(
        Image.fromarray((a * 255).astype(np.uint8), "L").resize((W, H), Image.BICUBIC),
        np.float32) / 255.0
    S = blur(up(river), max(2, P // 3))
    Gr = blur(up(grass), max(2, P // 3))
    valid = S + Gr
    f = (S - Gr) / np.maximum(valid, 1e-3)                 # +1 sand side, -1 grass side
    del S, Gr

    # push the contact off the tile grid with the same noise the river rides on
    xs = np.arange(W, dtype=np.float32)[None, :] + np.zeros((H, 1), np.float32)
    ys = np.arange(H, dtype=np.float32)[:, None] + np.zeros((1, W), np.float32)
    wx = ((fbm([6, 15, 38], 901, shape) - 0.5) * 0.80 * P +
          (fbm([80, 190], 903, shape) - 0.5) * 0.52 * P)
    wy = ((fbm([6, 15, 38], 911, shape) - 0.5) * 0.80 * P +
          (fbm([80, 190], 913, shape) - 0.5) * 0.52 * P)
    d = bilinear(f, xs + wx, ys + wy)
    valid = bilinear(valid, xs + wx * 0.7, ys + wy * 0.7)
    del f, xs, ys, wx, wy

    ragged = d + (fbm([150, 340, 780], 961, shape) - 0.5) * 0.42
    t = sstep(ragged, -0.16, 0.16)                         # 1 = sand, 0 = grass
    window = 1.0 - sstep(np.abs(d), 0.26, 0.92)            # only work near the contact
    window *= sstep(valid, 0.42, 0.86)                     # keep off roads and paving

    # ------------------------------------------------------------------ sand
    grain = fbm([300, 700, 1400], 311, shape)
    patch = fbm([8, 21, 55], 313, shape)
    st = np.clip(0.55 * grain + 0.45 * patch, 0, 1)
    sand = np.zeros((H, W, 3), np.float32)
    sand[:] = S_DRY
    sand = lerp3(sand, S_MID, sstep(patch, 0.45, 0.95) * 0.5)
    sand = lerp3(sand, (196, 178, 150), sstep(st, 0.55, 0.85) * 0.45)
    sand = lerp3(sand, (238, 226, 208), sstep(st, 0.10, 0.42) * 0.35)
    del grain, patch, st

    # ----------------------------------------------------------------- grass
    blade = fbm([460, 900, 1600], 921, shape)
    clump = fbm([26, 70, 170], 923, shape)
    gt = np.clip(0.58 * blade + 0.42 * clump, 0, 1)
    grass_c = np.where((gt < 0.5)[..., None],
                       lerp3(np.broadcast_to(np.float32(G_DARK), (H, W, 3)).copy(),
                             G_MID, np.clip(gt * 2, 0, 1)),
                       lerp3(np.broadcast_to(np.float32(G_MID), (H, W, 3)).copy(),
                             G_LIGHT, np.clip((gt - 0.5) * 2, 0, 1)))
    # grass gives up as it runs onto sand: drier, yellower, sandier at the tips
    dry = sstep(t, 0.10, 0.75)
    grass_c = lerp3(grass_c, (150, 150, 74), dry * 0.55)
    grass_c = lerp3(grass_c, (176, 168, 116), sstep(t, 0.45, 0.95) * 0.45)

    rgb = lerp3(grass_c, sand, t)
    del sand, grass_c, blade

    # a damp soil line right where the two meet, and windblown sand on the grass
    contact = (1.0 - np.abs(d) * 9.0).clip(0, 1)
    rgb = lerp3(rgb, (170, 152, 118), contact * t * 0.22)
    blown = sstep(fbm([140, 340, 700], 931, shape), 0.72, 0.95)
    rgb = lerp3(rgb, (222, 208, 182), blown * (1 - t) * sstep(d, -0.75, 0.0) * 0.40)

    # ----------------------------------------------------------------- alpha
    # solid on the sand side; clumpy on the grass side so the base grass shows
    # between tufts instead of being repainted as a sheet
    tuft = sstep(np.clip(0.42 * clump + 0.58 * fbm([120, 300, 700], 941, shape), 0, 1),
                 0.40, 0.86)
    a = window * np.clip(t + (1.0 - t) * tuft * 0.85, 0, 1)
    # let a few blades stand proud out on the sand
    stray = sstep(fbm([120, 300, 660], 951, shape), 0.86, 0.985)
    a = np.clip(a + stray * sstep(t, 0.30, 0.85) * (1 - sstep(t, 0.90, 1.0)) * 0.8, 0, 1)
    del clump, tuft, stray, contact, blown, dry, window, valid, d, t

    plan = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    del rgb, a

    Wc, Hc = x1 - x0, y1 - y0
    plan = plan[cut_y * P:(cut_y + Hc) * P, cut_x * P:(cut_x + Wc) * P]
    alpha = alpha[cut_y * P:(cut_y + Hc) * P, cut_x * P:(cut_x + Wc) * P]

    # --------------------------------------------------------- isometric warp
    rgba = np.dstack([plan, alpha])
    del plan, alpha
    img = Image.fromarray(np.pad(rgba, ((PAD, PAD), (PAD, PAD), (0, 0)), "edge"), "RGBA")
    del rgba
    img = img.resize((img.width * SS, img.height * SS), Image.BILINEAR)
    tw2, th2 = TW * SS, TH * SS
    IW, IH = (Wc + Hc) * tw2 // 2, (Wc + Hc) * th2 // 2
    iso = img.transform((IW, IH), Image.AFFINE,
                        (1, 2, -Hc * tw2 / 2.0 + PAD * SS, -1, 2, Hc * tw2 / 2.0 + PAD * SS),
                        resample=Image.BICUBIC)
    del img
    iso = iso.resize(((Wc + Hc) * TW // 2, (Wc + Hc) * TH // 2), Image.LANCZOS)

    m = np.zeros((TH, TW), np.uint8)
    for row in range(TH):
        half = 2 * (row if row < TH // 2 else TH - 1 - row) + 1
        m[row, TW // 2 - half:TW // 2 + half] = 255
    dia = Image.fromarray(m, "L")

    ox1 = Hc * (TW // 2)
    tiles, index = [], {}
    for cy in range(Hc):
        for cx in range(Wc):
            bx = (cx - cy) * (TW // 2) + ox1 - TW // 2
            by = (cx + cy) * (TH // 2)
            tl = iso.crop((bx, by, bx + TW, by + TH)).convert("RGBA")
            av = np.minimum(np.asarray(tl)[..., 3], np.asarray(dia))
            if av.max() < 10:
                continue
            tl.putalpha(Image.fromarray(av, "L"))
            index[(cx, cy)] = len(tiles)
            tiles.append(tl)
    del iso
    if not tiles:
        print("  (nothing to blend in this band)")
        return

    COLS = 48
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGBA", (COLS * TW, rows * TH), (0, 0, 0, 0))
    for i, tl in enumerate(tiles):
        sheet.paste(tl, ((i % COLS) * TW, (i // COLS) * TH))
    os.makedirs(outdir, exist_ok=True)
    sheet.save(os.path.join(outdir, name + ".png"), optimize=True)
    with open(os.path.join(outdir, name + ".tsx"), "w") as f2:
        f2.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                 f'<tileset version="1.10" tiledversion="1.10.2" name="{name}"'
                 f' tilewidth="{TW}" tileheight="{TH}" tilecount="{len(tiles)}"'
                 f' columns="{COLS}">\n <image source="{name}.png"'
                 f' width="{COLS * TW}" height="{rows * TH}"/>\n</tileset>\n')
    with open(os.path.join(outdir, name + ".csv"), "w") as f2:
        f2.write("%d %d %d %d\n" % (x0, y0, x1, y1))
        for cy in range(Hc):
            f2.write(",".join(str(index.get((cx, cy), -1) + 1) for cx in range(Wc)) + "\n")
    print("sheet", sheet.size, "tiles", len(tiles), "region", (x0, y0, x1, y1))


if __name__ == "__main__":
    a = sys.argv[1:]
    build(int(a[0]), int(a[1]), int(a[2]), int(a[3]), a[4], a[5], surface_classes())
