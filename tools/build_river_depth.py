#!/usr/bin/env python3
"""Build a depth-graded Yamuna river plate: dry sand -> wet sand -> foam line ->
shallows -> mid water -> deep channel, as one continuous surface sliced into
64x32 isometric tiles.

The existing River layer is only four flat colours, so every band boundary lands
exactly on a tile diamond and reads as a staircase.  Here the layer is instead
read as a *depth field*: each painted cell contributes a depth value, the field
is smoothed to sub-tile resolution, warped by noise so the waterline meanders off
the grid, and only then coloured.  Nothing in the ramp knows where the tile
edges are, so no edge can show.

The river is longer than one plate can hold, so it is built in row bands.  All
noise is addressed in global map coordinates and every band is computed with a
margin of spare cells that is cropped off before slicing, so neighbouring bands
agree exactly across their join and no seam appears.

Usage:  build_river_depth.py X0 Y0 X1 Y1 OUTDIR NAME
"""
import os
import re
import sys

import numpy as np
from PIL import Image

MAP = "/Users/withcenter10/Desktop/Delhi-India_map/_share/mapp.tmx"
TW, TH = 64, 32
P  = 64      # plan-view pixels per map cell
SS = 2       # supersample factor for the iso warp
PAD = 24

# depth assigned to each painted band, in [-1, 1]; land is negative
D_LAND, D_SAND, D_FOAM, D_SHALLOW, D_DEEP = -1.00, -0.34, 0.03, 0.44, 1.00


REF = 56          # frequencies are quoted against a 56-cell window
CTX = {"ox": 0.0, "oy": 0.0}   # window origin in global plan pixels


# ------------------------------------------------------------------ helpers
def _lattice(jx, jy, seed):
    """Deterministic value in [0,1) for integer lattice point (jx, jy)."""
    h = (jx.astype(np.int64) * 374761393 +
         jy.astype(np.int64) * 668265263 +
         np.int64(seed) * 1442695043)
    h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
    h = h ^ (h >> np.int64(16))
    return (h & np.int64(0xFFFFFF)).astype(np.float32) / float(0xFFFFFF)


def noise(freq, seed, shape, px=1.0):
    """Value noise sampled in global map space.

    `freq` is the number of features across a REF-cell window, so the same
    number means the same feature size in every band.  `px` is how many plan
    pixels one array pixel covers, for fields evaluated at reduced resolution.
    Interpolation is Catmull-Rom: it overshoots the lattice the way a bicubic
    upsample does, which keeps the contrast the colour ramp was tuned against.
    """
    H, W = shape
    step = REF * P / float(freq) / px           # array pixels per lattice cell
    gx = (np.arange(W, dtype=np.float64) + CTX["ox"] / px) / step
    gy = (np.arange(H, dtype=np.float64) + CTX["oy"] / px) / step
    ix0, iy0 = int(np.floor(gx[0])) - 1, int(np.floor(gy[0])) - 1
    nx = int(np.floor(gx[-1])) - ix0 + 4
    ny = int(np.floor(gy[-1])) - iy0 + 4
    lat = _lattice((np.arange(nx) + ix0)[None, :],
                   (np.arange(ny) + iy0)[:, None], seed)

    def taps(g, i0):
        u = (g - i0).astype(np.float32)
        i = np.floor(u).astype(np.int32)
        t = u - i
        t2, t3 = t * t, t * t * t
        return i, [(-0.5 * t3 + t2 - 0.5 * t),
                   (1.5 * t3 - 2.5 * t2 + 1.0),
                   (-1.5 * t3 + 2.0 * t2 + 0.5 * t),
                   (0.5 * t3 - 0.5 * t2)]

    iu, wu = taps(gx, ix0)
    iv, wv = taps(gy, iy0)
    iu = iu[None, :]
    wu = [w[None, :] for w in wu]

    out = np.empty(shape, np.float32)
    for r0 in range(0, H, 512):                  # row blocks keep peak RAM down
        r1 = min(H, r0 + 512)
        jv = iv[r0:r1, None]
        acc = np.zeros((r1 - r0, W), np.float32)
        for dy in range(4):
            row = np.zeros((r1 - r0, W), np.float32)
            for dx in range(4):
                row += lat[jv + dy, iu + dx] * wu[dx]
            acc += row * wv[dy][r0:r1, None]
        out[r0:r1] = acc
    return np.clip(out, 0.0, 1.0)


def fbm(freqs, seed, shape, weights=None, px=1.0):
    weights = weights or [1.0 / (i + 1) for i in range(len(freqs))]
    acc = np.zeros(shape, np.float32)
    for i, (w, f) in enumerate(zip(weights, freqs)):
        acc += w * noise(f, seed + i * 977, shape, px)
    return acc / sum(weights)


def sstep(a, e0, e1):
    t = np.clip((a - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp3(c0, c1, t):
    t = t[..., None]
    return np.asarray(c0, np.float32) * (1 - t) + np.asarray(c1, np.float32) * t


def _box1d(a, r, axis):
    a = np.moveaxis(a, axis, -1)
    pad = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(r + 1, r)], "edge")
    c = np.cumsum(pad, axis=-1, dtype=np.float32)
    out = (c[..., 2 * r + 1:] - c[..., :-(2 * r + 1)]) / (2 * r + 1)
    return np.moveaxis(out, -1, axis)


def blur(a, r):
    """Separable box blur, repeated for a near-gaussian falloff."""
    for _ in range(3):
        a = _box1d(_box1d(a, r, 0), r, 1)
    return a


def bilinear(field, u, v):
    """Sample `field` at float coords (u, v), clamped at the border."""
    h, w = field.shape
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0 = u.astype(np.int32)
    y0 = v.astype(np.int32)
    fx = (u - x0)[..., None][..., 0]
    fy = (v - y0)[..., None][..., 0]
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    return ((field[y0, x0] * (1 - fx) + field[y0, x1] * fx) * (1 - fy) +
            (field[y1, x0] * (1 - fx) + field[y1, x1] * fx) * fy)


# ------------------------------------------------- read the painted river bands
def read_depth_cells(x0, y0, x1, y1):
    """Turn the hand-painted River layer into one depth value per map cell."""
    src = open(MAP).read()
    root = os.path.dirname(MAP)
    tsets = [(int(m.group(1)), os.path.normpath(os.path.join(root, m.group(2))))
             for m in re.finditer(r'<tileset firstgid="(\d+)" source="([^"]+)"/>', src)]

    cache, mean = {}, {}

    def band(gid):
        if gid in mean:
            return mean[gid]
        best = max((t for t in tsets if gid >= t[0]), key=lambda t: t[0], default=None)
        v = D_LAND
        if best:
            fg, path = best
            if path not in cache:
                try:
                    t = open(path).read()
                    img = re.search(r'<image source="([^"]+)"', t).group(1)
                    cache[path] = (
                        Image.open(os.path.normpath(
                            os.path.join(os.path.dirname(path), img))).convert("RGBA"),
                        int(re.search(r'columns="(\d+)"', t).group(1)),
                        int(re.search(r'tilecount="(\d+)"', t).group(1)),
                        int(re.search(r'tilewidth="(\d+)"', t).group(1)),
                        int(re.search(r'tileheight="(\d+)"', t).group(1)))
                except Exception:
                    cache[path] = None
            e = cache[path]
            if e:
                im, cols, count, tw, th = e
                l = gid - fg
                if l < count:
                    a = np.asarray(im.crop(((l % cols) * tw, (l // cols) * th,
                                            (l % cols) * tw + tw,
                                            (l // cols) * th + th)), np.float32)
                    op = a[..., 3] > 128
                    if op.any():
                        r, g, b = a[..., :3][op].mean(0)
                        if b > 175 and r < 70:            # bright cyan shallows
                            v = D_SHALLOW
                        elif b > 120 and r < 70:          # dark blue channel
                            v = D_DEEP
                        elif g > r and g > 150 and r > 90:  # green-grey foam edge
                            v = D_FOAM
                        elif r > 195 and g > 180:         # pale sand bank
                            v = D_SAND
        mean[gid] = v
        return v

    m = re.search(r'<layer id="3" name="[^"]*"[^>]*>\s*<data encoding="csv">(.*?)</data>',
                  src, re.S)
    grid = [[int(c) for c in r.strip().rstrip(',').split(',')]
            for r in m.group(1).strip().split('\n')]

    d = np.full((y1 - y0, x1 - x0), D_LAND, np.float32)
    cover = np.zeros_like(d)
    for y in range(y0, y1):
        for x in range(x0, x1):
            gid = grid[y][x]
            if not gid:
                continue
            v = band(gid)
            d[y - y0, x - x0] = v
            if v > D_LAND:
                cover[y - y0, x - x0] = 1.0
    return d, cover


# --------------------------------------------------------------------- build
MARGIN = 6        # spare cells computed on every side, cropped before slicing


def build(x0, y0, x1, y1, outdir, name, cover_cells=None):
    # widen to the margin, clamped to the map, and remember what to crop back
    mx0, my0 = max(0, x0 - MARGIN), max(0, y0 - MARGIN)
    mx1, my1 = min(300, x1 + MARGIN), min(300, y1 + MARGIN)
    cut_x, cut_y = x0 - mx0, y0 - my0

    Wc, Hc = mx1 - mx0, my1 - my0
    W, H = Wc * P, Hc * P
    shape = (H, W)
    CTX["ox"], CTX["oy"] = mx0 * P, my0 * P

    dcell, ccell = read_depth_cells(mx0, my0, mx1, my1)
    if cover_cells is not None:
        # paint the ramp onto an explicit cell set instead of the river footprint
        ccell = np.zeros_like(ccell)
        for (cx, cy) in cover_cells:
            if mx0 <= cx < mx1 and my0 <= cy < my1:
                ccell[cy - my0, cx - mx0] = 1.0

    # cell field -> plan field.  One bicubic upsample already removes the hard
    # per-cell steps; the extra blur widens each band into a real gradient.
    up = lambda a: np.asarray(
        Image.fromarray(((a + 1.0) * 127.5).astype(np.uint8), "L")
        .resize((W, H), Image.BICUBIC), np.float32) / 127.5 - 1.0
    depth0 = blur(up(dcell), max(2, P // 16))
    cov0 = blur(up(ccell * 2.0 - 1.0) * 0.5 + 0.5, max(2, P // 12))

    # domain warp: push the whole field around with noise so the waterline
    # wanders across tile diamonds instead of tracking them
    xs = np.arange(W, dtype=np.float32)[None, :] + np.zeros((H, 1), np.float32)
    ys = np.arange(H, dtype=np.float32)[:, None] + np.zeros((1, W), np.float32)
    wx = ((fbm([4, 11, 29], 101, shape) - 0.5) * 1.15 * P +
          (fbm([60, 140], 103, shape) - 0.5) * 0.20 * P)
    wy = ((fbm([4, 11, 29], 211, shape) - 0.5) * 1.15 * P +
          (fbm([60, 140], 213, shape) - 0.5) * 0.20 * P)
    depth = bilinear(depth0, xs + wx, ys + wy)
    cover = bilinear(cov0, xs + wx * 0.8, ys + wy * 0.8)
    del depth0, cov0, xs, ys

    # shore-normal and channel tangent, for shore-parallel waves and flow streaks
    gy, gx = np.gradient(blur(depth, max(3, P // 6)))
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = gx / gl, gy / gl
    del gx, gy, gl

    # ------------------------------------------------------------ colour ramp
    S_DRY   = (228, 212, 190)
    S_MID   = (206, 186, 158)
    S_WET   = (150, 132, 106)
    W_EDGE  = (104, 178, 168)
    W_SHAL  = ( 40, 158, 178)
    W_MID   = ( 12, 126, 168)
    W_DEEP  = (  8,  76, 124)
    W_ABYSS = (  5,  44,  84)

    rgb = np.zeros((H, W, 3), np.float32)
    rgb[:] = S_DRY
    rgb = lerp3(rgb, S_MID,   sstep(depth, -0.85, -0.30))
    rgb = lerp3(rgb, S_WET,   sstep(depth, -0.30, -0.02))
    rgb = lerp3(rgb, W_EDGE,  sstep(depth, -0.02,  0.13))
    rgb = lerp3(rgb, W_SHAL,  sstep(depth,  0.10,  0.34))
    rgb = lerp3(rgb, W_MID,   sstep(depth,  0.32,  0.58))
    rgb = lerp3(rgb, W_DEEP,  sstep(depth,  0.56,  0.82))
    rgb = lerp3(rgb, W_ABYSS, sstep(depth,  0.80,  1.00))

    wet = sstep(depth, -0.02, 0.06)          # 0 on the bank, 1 in the water

    # ------------------------------------------------------------- sand detail
    grain = fbm([300, 700, 1400], 311, shape)
    patch = fbm([8, 21, 55], 313, shape)
    sand_t = np.clip(0.55 * grain + 0.45 * patch, 0, 1)
    rgb = lerp3(rgb, (196, 178, 150), (1 - wet) * sstep(sand_t, 0.55, 0.85) * 0.45)
    rgb = lerp3(rgb, (238, 226, 208), (1 - wet) * sstep(sand_t, 0.10, 0.42) * 0.35)
    # tide marks: damp arcs left behind, following the depth contours
    tide = np.sin(depth * 11.0 + fbm([12, 33], 317, shape) * 7.0)
    tbreak = sstep(fbm([18, 48, 120], 319, shape), 0.40, 0.82)
    rgb = lerp3(rgb, (188, 170, 142),
                (1 - wet) * sstep(tide, 0.72, 0.99) * tbreak *
                sstep(depth, -0.60, -0.22) * 0.22)
    # dry sand keeps a visible tooth so the bank never reads as flat paint
    tooth = fbm([600, 1200], 323, shape)
    rgb = lerp3(rgb, (246, 236, 220), (1 - wet) * sstep(tooth, 0.62, 0.92) * 0.30)
    rgb = lerp3(rgb, (182, 164, 134), (1 - wet) * sstep(1 - tooth, 0.64, 0.94) * 0.26)

    # ------------------------------------------------------- dry back-beach
    # the flats away from the water are wide, and fine grain alone leaves them
    # looking like one sheet of paper, so comb them with wind ripples and let
    # the tone drift on a scale you can actually see from a map zoom
    dry = 1.0 - sstep(depth, -0.62, -0.28)
    gxs = (np.arange(W, dtype=np.float32) + CTX["ox"])[None, :]
    gys = (np.arange(H, dtype=np.float32) + CTX["oy"])[:, None]
    rip = np.sin((gxs * 0.6 + gys) / P * 6.5 + fbm([9, 26], 331, shape) * 9.0)
    rgb = lerp3(rgb, (212, 196, 172), np.clip(rip, 0, 1) ** 2 * dry * 0.17)
    rgb = lerp3(rgb, (242, 232, 217), np.clip(-rip, 0, 1) ** 2 * dry * 0.13)
    drift = fbm([3, 7, 16], 337, shape)
    rgb = lerp3(rgb, (211, 194, 168), dry * sstep(drift, 0.52, 0.90) * 0.30)
    rgb = lerp3(rgb, (243, 234, 219), dry * sstep(1 - drift, 0.52, 0.90) * 0.22)
    # wind-scoured streaks and a scatter of darker grit
    grit = sstep(fbm([160, 380, 800], 341, shape), 0.80, 0.97)
    rgb = lerp3(rgb, (168, 150, 122), grit * dry * 0.22)
    del gxs, gys, rip, drift, grit, dry
    del grain, patch, sand_t, tide, tbreak, tooth

    # ------------------------------------------------- shore-parallel wave sets
    # phase driven by depth, so every crest follows the shoreline the way real
    # swell refracts into a beach
    ph = depth * 17.0 + fbm([7, 19, 48], 401, shape) * 9.0
    swell = np.sin(ph) * 0.65 + np.sin(ph * 2.3 + 1.3) * 0.35
    # break the crests into runs so they read as chop, not as contour lines
    runs = sstep(fbm([13, 34, 88], 403, shape), 0.30, 0.72)
    band = sstep(depth, 0.02, 0.16) * (1 - sstep(depth, 0.30, 0.62)) * runs
    rgb = lerp3(rgb, (166, 220, 222), np.clip(swell, 0, 1) ** 2 * band * 0.30)
    rgb = lerp3(rgb, ( 12, 120, 152), np.clip(-swell, 0, 1) ** 2 * band * 0.22)
    del ph, band, runs

    # ---------------------------------------------------------------- caustics
    ca = fbm([40, 90, 190], 421, shape)
    cb = fbm([44, 96, 200], 431, shape)
    caust = (1.0 - np.abs(ca - cb) * 9.0).clip(0, 1) ** 2
    patchy = sstep(fbm([10, 28, 72], 433, shape), 0.34, 0.80)
    fade = sstep(depth, 0.06, 0.22) * (1 - sstep(depth, 0.36, 0.70))
    rgb = lerp3(rgb, (196, 238, 234), caust * patchy * fade * 0.20)
    del ca, cb, caust, patchy, fade

    # ------------------------------------------------- deep-channel flow smear
    # streaks pulled along the channel by walking a noise field down the tangent
    q = 4
    small = (H // q, W // q)
    base = fbm([26, 70, 160], 511, small, px=q)
    tx = -ny[::q, ::q][:small[0], :small[1]]
    ty = nx[::q, ::q][:small[0], :small[1]]
    sx = np.arange(small[1], dtype=np.float32)[None, :] + np.zeros((small[0], 1), np.float32)
    sy = np.arange(small[0], dtype=np.float32)[:, None] + np.zeros((1, small[1]), np.float32)
    lic = np.zeros(small, np.float32)
    for k in range(-6, 7):
        lic += bilinear(base, sx + tx * k * 3.0, sy + ty * k * 3.0)
    lic /= 13.0
    lic = np.asarray(Image.fromarray((np.clip(lic, 0, 1) * 255).astype(np.uint8), "L")
                     .resize((W, H), Image.BICUBIC), np.float32) / 255.0
    deepf = sstep(depth, 0.42, 0.72)
    rgb = lerp3(rgb, ( 30, 122, 162), sstep(lic, 0.44, 0.78) * deepf * 0.52)
    rgb = lerp3(rgb, (  4,  38,  74), sstep(1 - lic, 0.44, 0.80) * deepf * 0.46)
    del base, tx, ty, sx, sy, lic, small

    # --------------------------------------------------------------- foam line
    # a sinuous band just off the waterline, plus scattered leftover bubbles
    fw = fbm([16, 44, 110], 601, shape)
    core = (1.0 - np.abs(depth - 0.055 - (fw - 0.5) * 0.10) * 22.0).clip(0, 1)
    lace = sstep(fbm([90, 210, 440], 607, shape), 0.42, 0.80)
    foam = np.clip(core * (0.45 + 0.55 * lace), 0, 1)
    spray = sstep(fbm([120, 300], 611, shape), 0.72, 0.94) * \
        sstep(depth, -0.10, 0.02) * (1 - sstep(depth, 0.10, 0.26))
    rgb = lerp3(rgb, (244, 250, 248), np.clip(foam * 0.85 + spray * 0.5, 0, 1))
    # thin dark contact line where water meets sand, so the shore reads as an edge
    contact = (1.0 - np.abs(depth + 0.015) * 40.0).clip(0, 1)
    rgb = lerp3(rgb, (118, 116, 100), contact * 0.30)
    del fw, core, lace, foam, spray, contact

    # ------------------------------------------------------- surface and light
    # sky sheen: water lightens toward the far bank, and a slow specular roll
    sheen = sstep(fbm([6, 15], 701, shape), 0.35, 0.85)
    rgb = lerp3(rgb, (150, 214, 226), sheen * wet * 0.14)
    glint = sstep(fbm([200, 420, 900], 707, shape), 0.86, 0.98)
    rgb = lerp3(rgb, (255, 255, 250),
                glint * wet * (1 - sstep(depth, 0.55, 0.85)) * 0.5)
    rgb *= (0.955 + 0.09 * fbm([4, 10], 719, shape))[..., None]
    del sheen, glint

    # --------------------------------------------------- fine surface texture
    # high-frequency chop, stretched along the channel so it reads as moving water
    chop = (0.55 * fbm([220, 460], 801, shape) +
            0.45 * fbm([110, 300, 640], 803, shape))
    amp = wet * (0.55 + 0.45 * (1 - sstep(depth, 0.35, 0.80)))
    rgb = lerp3(rgb, (198, 236, 240), sstep(chop, 0.46, 0.86) * amp * 0.34)
    rgb = lerp3(rgb, (  6,  60, 102), sstep(1 - chop, 0.46, 0.86) * amp * 0.30)
    # a second, coarser set of surface facets to give the sheet some body
    # deep water holds still, so let the facets fade out over the channel
    facet = fbm([55, 130], 807, shape)
    fa = wet * (1 - 0.62 * sstep(depth, 0.45, 0.85))
    rgb = lerp3(rgb, (150, 208, 218), sstep(facet, 0.52, 0.90) * fa * 0.20)
    rgb = lerp3(rgb, (  8,  68, 110), sstep(1 - facet, 0.52, 0.90) * fa * 0.18)
    del chop, amp, facet, fa

    plan = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = (np.clip(sstep(cover, 0.30, 0.62), 0, 1) * 255).astype(np.uint8)
    del rgb, depth, cover, wet, nx, ny

    # drop the margin: everything above was computed with real neighbours, so
    # what is left matches the adjoining band pixel for pixel
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
    ox2 = Hc * tw2 / 2.0
    pad2 = PAD * SS
    iso = img.transform((IW, IH), Image.AFFINE,
                        (1, 2, -ox2 + pad2, -1, 2, ox2 + pad2),
                        resample=Image.BICUBIC)
    del img
    iso = iso.resize(((Wc + Hc) * TW // 2, (Wc + Hc) * TH // 2), Image.LANCZOS)

    # ---------------------------------------------------------------- slicing
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
            t = iso.crop((bx, by, bx + TW, by + TH)).convert("RGBA")
            a = np.minimum(np.asarray(t)[..., 3], np.asarray(dia))
            if a.max() < 8:
                continue
            t.putalpha(Image.fromarray(a, "L"))
            index[(cx, cy)] = len(tiles)
            tiles.append(t)
    del iso

    if not tiles:
        raise SystemExit("%s: no covered cells in this region -- nothing to slice" % name)
    COLS = 48
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGBA", (COLS * TW, rows * TH), (0, 0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % COLS) * TW, (i // COLS) * TH))

    os.makedirs(outdir, exist_ok=True)
    sheet.save(os.path.join(outdir, name + ".png"), optimize=True)
    with open(os.path.join(outdir, name + ".tsx"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<tileset version="1.10" tiledversion="1.10.2" name="{name}"'
                f' tilewidth="{TW}" tileheight="{TH}" tilecount="{len(tiles)}"'
                f' columns="{COLS}">\n'
                f' <image source="{name}.png" width="{COLS * TW}"'
                f' height="{rows * TH}"/>\n</tileset>\n')

    # placement map: local id + 1 at each cell of the source region
    with open(os.path.join(outdir, name + ".csv"), "w") as f:
        f.write("%d %d %d %d\n" % (x0, y0, x1, y1))
        for cy in range(Hc):
            f.write(",".join(str(index.get((cx, cy), -1) + 1) for cx in range(Wc)) + "\n")

    print("sheet", sheet.size, "tiles", len(tiles), "region", (x0, y0, x1, y1))


if __name__ == "__main__":
    a = sys.argv[1:]
    cells = None
    if len(a) > 6:                      # optional "x y" per line coverage mask
        cells = {tuple(int(v) for v in ln.split())
                 for ln in open(a[6]) if ln.strip()}
    build(int(a[0]), int(a[1]), int(a[2]), int(a[3]), a[4], a[5], cells)
