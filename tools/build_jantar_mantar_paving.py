#!/usr/bin/env python3
"""Build the Jantar Mantar approach-walk ground plate: red sandstone flagstone
paving with the central mown grass median, sliced into 64x32 isometric tiles."""
import os
import numpy as np
from PIL import Image

OUT = "/Users/withcenter10/Desktop/Delhi-India_map/_share/Ground"
NAME = "jantar-mantar-paving"

N    = 32     # plate is N x N map cells
P    = 128    # plan-view pixels per tile
TW, TH = 64, 32
SS   = 2      # supersample factor for the iso warp
PAD  = 8

W = H = N * P
rng = np.random.default_rng(20260831)


def noise(freq, seed, size=None):
    """Smooth value noise in [0,1] at plan resolution."""
    size = size or (W, H)
    g = np.random.default_rng(seed).random((freq, freq)).astype(np.float32)
    im = Image.fromarray((g * 255).astype(np.uint8), "L").resize(size, Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def fbm(freqs, seed, weights=None):
    weights = weights or [1.0 / (i + 1) for i in range(len(freqs))]
    acc = np.zeros((H, W), np.float32)
    tot = 0.0
    for i, f in enumerate(freqs):
        acc += weights[i] * noise(f, seed + i * 977)
        tot += weights[i]
    return acc / tot


def sstep(a, e0, e1):
    t = np.clip((a - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp3(c0, c1, t):
    t = t[..., None]
    return np.asarray(c0, np.float32) * (1 - t) + np.asarray(c1, np.float32) * t


# ---------------------------------------------------------------- flagstones
xs = np.arange(W, dtype=np.float32)[None, :]
ys = np.arange(H, dtype=np.float32)[:, None]

# long courses running with the walk, as in the photo: slabs are wider than deep
SLAB_X, SLAB_Y = 1.55, 1.05
gx = xs / (P * SLAB_X) + (fbm([9, 33], 11) - 0.5) * 0.12
gy = ys / (P * SLAB_Y) + (fbm([9, 33], 23) - 0.5) * 0.10

ix = np.floor(gx).astype(np.int16)
iy = np.floor(gy).astype(np.int16)
fx = gx - ix
fy = gy - iy

# per-slab random variates
TA = rng.random((64, 64)).astype(np.float32)
TB = rng.random((64, 64)).astype(np.float32)
sa = TA[np.mod(iy, 64), np.mod(ix, 64)]
sb = TB[np.mod(iy, 64), np.mod(ix, 64)]

grain = fbm([256, 512, 1024], 41)
mottle = fbm([6, 17, 44], 43)
tone = np.clip(0.46 * sa + 0.30 * grain + 0.24 * mottle, 0.0, 1.0)

DARK  = (148, 107,  97)
MID   = (186, 142, 129)
LIGHT = (214, 173, 158)
rgb = np.where((tone < 0.5)[..., None],
               lerp3(DARK, MID, np.clip(tone * 2.0, 0, 1)),
               lerp3(MID, LIGHT, np.clip((tone - 0.5) * 2.0, 0, 1)))

# grey weathering bloom, heaviest where slabs sit low and hold damp
bloom = sstep(fbm([12, 40, 120], 57), 0.48, 0.76)
rgb = lerp3(rgb, (156, 145, 134), bloom * 0.62)

# damp dark patches, like the ones the photo shows across the walk
stain = sstep(fbm([7, 19, 61], 73), 0.56, 0.74) * (0.30 + 0.70 * sb)
rgb = lerp3(rgb, (118,  88,  80), stain * 0.62)
# a few near-black scuffs and tyre/foot marks
scuff = sstep(fbm([14, 38, 96], 79), 0.80, 0.93)
rgb = lerp3(rgb, (96,  73,  67), scuff * 0.45)

# the mortar joints between slabs
d = np.minimum(np.minimum(fx, 1.0 - fx), np.minimum(fy, 1.0 - fy))
joint = 1.0 - sstep(d, 0.012, 0.045)
rgb = lerp3(rgb, (126,  95,  85), joint * 0.72)
# a thin catch-light on the lower lip of each joint so slabs read as solid
lip = (1.0 - sstep(np.minimum(fy, fx), 0.045, 0.085)) * (1.0 - joint)
rgb = lerp3(rgb, (226, 191, 176), lip * 0.22)

del gx, gy, fx, fy, d, joint, lip, tone, grain, mottle, sa, sb, bloom, stain, scuff

# ------------------------------------------------------------- grass median
MED0, MED1 = 15.0, 17.0
yt = ys / P + (fbm([5, 23], 91) - 0.5) * 0.045   # slightly ragged edge
xt = xs / P

inside = sstep(yt, MED0, MED0 + 0.035) * (1.0 - sstep(yt, MED1 - 0.035, MED1))

# mown grass: streaks pulled long along the walk direction
blade = np.asarray(
    Image.fromarray((np.random.default_rng(131).random((720, 40)) * 255).astype(np.uint8), "L")
    .resize((W, H), Image.BICUBIC), np.float32) / 255.0
gclump = fbm([16, 48, 140], 151)
gt = np.clip(0.36 * blade + 0.44 * gclump + 0.20 * fbm([420, 900], 157), 0, 1)
grass = np.where((gt < 0.5)[..., None],
                 lerp3((60,  88,  40), ( 96, 130,  56), np.clip(gt * 2, 0, 1)),
                 lerp3(( 96, 130,  56), (136, 165,  80), np.clip((gt - 0.5) * 2, 0, 1)))
# scuffed / dry ground showing through, as on the trodden median
bare = sstep(fbm([10, 34, 90], 167), 0.70, 0.90)
grass = lerp3(grass, (150, 128,  86), bare * 0.5)
# darker where the turf meets the stone
edge = np.maximum(1.0 - sstep(yt - MED0, 0.0, 0.30), 1.0 - sstep(MED1 - yt, 0.0, 0.30))
grass = lerp3(grass, (52,  78,  34), np.clip(edge, 0, 1) * 0.45 * inside)

rgb = rgb * (1 - inside[..., None]) + grass * inside[..., None]
del grass, blade, gclump, gt, bare, edge

# dark contact gap between turf and paving, plus the raised kerb highlight
gap = (sstep(yt, MED0 - 0.055, MED0 - 0.012) * (1 - sstep(yt, MED0 - 0.012, MED0 + 0.018)) +
       sstep(yt, MED1 - 0.018, MED1 + 0.012) * (1 - sstep(yt, MED1 + 0.012, MED1 + 0.055)))
rgb = lerp3(rgb, (78, 52, 44), np.clip(gap, 0, 1) * 0.75)
kerb = (sstep(yt, MED0 - 0.20, MED0 - 0.10) * (1 - sstep(yt, MED0 - 0.10, MED0 - 0.045)) +
        sstep(yt, MED1 + 0.045, MED1 + 0.10) * (1 - sstep(yt, MED1 + 0.10, MED1 + 0.20)))
rgb = lerp3(rgb, (223, 178, 158), np.clip(kerb, 0, 1) * 0.5)
del gap, kerb, inside, yt, xt

# broad, very soft light variation so a big fill never looks stamped
rgb *= (0.945 + 0.11 * fbm([4, 11], 211))[..., None]
plan = np.clip(rgb, 0, 255).astype(np.uint8)
del rgb

# ------------------------------------------------------------- isometric warp
padded = Image.fromarray(np.pad(plan, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge"), "RGB")
del plan

tw2, th2 = TW * SS, TH * SS
IW, IH = N * tw2, N * th2
ox2 = IW / 2.0
# output(X,Y) samples plan(u,v):  u = (X-ox)+2Y ,  v = 2Y-(X-ox)
iso = padded.transform((IW, IH), Image.AFFINE,
                       (1, 2, -ox2 + PAD, -1, 2, ox2 + PAD),
                       resample=Image.BICUBIC).resize((N * TW, N * TH), Image.LANCZOS)
del padded

# ------------------------------------------------------------------- slicing
m = np.zeros((TH, TW), np.uint8)
for row in range(TH):
    half = 2 * (row if row < TH // 2 else TH - 1 - row) + 1
    m[row, TW // 2 - half:TW // 2 + half] = 255
mask = Image.fromarray(m, "L")

COLS = N
sheet = Image.new("RGBA", (COLS * TW, N * N // COLS * TH), (0, 0, 0, 0))
ox1 = N * (TW // 2)
for y in range(N):
    for x in range(N):
        bx = (x - y) * (TW // 2) + ox1 - TW // 2
        by = (x + y) * (TH // 2)
        t = iso.crop((bx, by, bx + TW, by + TH)).convert("RGBA")
        t.putalpha(mask)
        i = y * N + x
        sheet.paste(t, ((i % COLS) * TW, (i // COLS) * TH))

os.makedirs(OUT, exist_ok=True)
sheet.save(os.path.join(OUT, NAME + ".png"), optimize=True)

count = N * N
rows = count // COLS
with open(os.path.join(OUT, NAME + ".tsx"), "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<tileset version="1.10" tiledversion="1.10.2" name="{NAME}"'
            f' tilewidth="{TW}" tileheight="{TH}" tilecount="{count}" columns="{COLS}">\n'
            f' <image source="{NAME}.png" width="{COLS * TW}" height="{rows * TH}"/>\n'
            '</tileset>\n')

data = "\n".join(",".join(str(y * N + x + 1) for x in range(N)) + ("," if y < N - 1 else "")
                 for y in range(N))
with open(os.path.join(OUT, NAME + ".tmx"), "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<map version="1.10" tiledversion="1.10.2" orientation="isometric"'
            f' renderorder="right-down" width="{N}" height="{N}" tilewidth="{TW}"'
            f' tileheight="{TH}" infinite="0" nextlayerid="2" nextobjectid="1">\n'
            f' <tileset firstgid="1" source="{NAME}.tsx"/>\n'
            f' <layer id="1" name="Ground" width="{N}" height="{N}">\n'
            '  <data encoding="csv">\n' + data + '\n</data>\n'
            ' </layer>\n</map>\n')

print("sheet", sheet.size, "tiles", count, "cols", COLS, "rows", rows)
