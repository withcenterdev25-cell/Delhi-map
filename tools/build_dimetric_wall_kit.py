#!/usr/bin/env python3
"""Build a texture-mapped wall kit on an exact 2:1 dimetric pixel grid."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "assets/generated/wall-material-atlas.png"
OUT = ROOT / "assets/generated/dimetric-wall-kit"

# Exact 2:1 screen-space basis. Both ground axes have |dy/dx| = 32/64 = 0.5.
U = np.array([64.0, 32.0])
V = np.array([-64.0, 32.0])
# At azimuth 45 degrees and elevation 30 degrees, a ground unit whose
# horizontal projection is 64 px gives a vertical unit of 64*sqrt(3/2).
Z = np.array([0.0, -64.0 * math.sqrt(3.0 / 2.0)])


def project(p):
    u, v, z = p
    return u * U + v * V + z * Z


def shade_texture(texture: Image.Image, factor: float) -> np.ndarray:
    tex = ImageEnhance.Brightness(texture.convert("RGB")).enhance(factor)
    return np.asarray(tex, dtype=np.uint8)


def paint_parallelogram(canvas: np.ndarray, texture: np.ndarray, p0, p1, p3):
    """Texture an affine parallelogram p0,p1,p2,p3 into an RGBA array."""
    p0 = np.asarray(p0, dtype=float)
    a = np.asarray(p1, dtype=float) - p0
    b = np.asarray(p3, dtype=float) - p0
    p2 = p0 + a + b
    pts = np.stack([p0, p1, p2, p3])
    lo = np.maximum(np.floor(pts.min(axis=0)).astype(int), 0)
    hi = np.minimum(np.ceil(pts.max(axis=0)).astype(int) + 1, [canvas.shape[1], canvas.shape[0]])
    if np.any(hi <= lo):
        return
    matrix = np.column_stack([a, b])
    if abs(np.linalg.det(matrix)) < 1e-8:
        return
    inv = np.linalg.inv(matrix)
    xs = np.arange(lo[0], hi[0]) + 0.5
    ys = np.arange(lo[1], hi[1]) + 0.5
    xx, yy = np.meshgrid(xs, ys)
    rel = np.stack([xx - p0[0], yy - p0[1]], axis=-1)
    uv = rel @ inv.T
    mask = (uv[..., 0] >= 0) & (uv[..., 0] <= 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= 1)
    th, tw = texture.shape[:2]
    tx = np.clip((uv[..., 0] * (tw - 1)).astype(int), 0, tw - 1)
    ty = np.clip(((1.0 - uv[..., 1]) * (th - 1)).astype(int), 0, th - 1)
    sample = texture[ty, tx]
    region = canvas[lo[1]:hi[1], lo[0]:hi[0]]
    region[mask, :3] = sample[mask]
    region[mask, 3] = 255


class Surface:
    def __init__(self, quad3, material, shade, depth):
        self.quad3 = quad3
        self.material = material
        self.shade = shade
        self.depth = depth


def cuboid(surfaces, u0, u1, v0, v1, z0, z1, material="concrete"):
    # Camera looks from positive U and positive V. Only these two side faces plus top are visible.
    surfaces.append(Surface(
        [(u0, v1, z0), (u1, v1, z0), (u1, v1, z1), (u0, v1, z1)],
        material, 0.82, (u0 + u1) / 2 + v1 + z0 * 0.01,
    ))
    surfaces.append(Surface(
        [(u1, v0, z0), (u1, v1, z0), (u1, v1, z1), (u1, v0, z1)],
        material, 0.68, u1 + (v0 + v1) / 2 + z0 * 0.01,
    ))
    surfaces.append(Surface(
        [(u0, v0, z1), (u1, v0, z1), (u1, v1, z1), (u0, v1, z1)],
        material, 1.05, (u0 + u1 + v0 + v1) / 2 + z1 * 0.01,
    ))


def add_post(surfaces, u, v, height=1.05):
    d = 0.14
    cuboid(surfaces, u - d, u + d, v - d, v + d, 0, height, "concrete")
    cuboid(surfaces, u - 0.17, u + 0.17, v - 0.17, v + 0.17, height, height + 0.07, "concrete")


def add_wall(surfaces, a, b, height=0.92, material="brick", posts=True):
    u0, v0 = a
    u1, v1 = b
    t = 0.09
    if abs(v1 - v0) < 1e-6:
        lo, hi = sorted([u0, u1])
        cuboid(surfaces, lo, hi, v0 - t, v0 + t, 0.00, 0.10, "concrete")
        cuboid(surfaces, lo, hi, v0 - t, v0 + t, 0.10, height - 0.10, material)
        cuboid(surfaces, lo, hi, v0 - 0.11, v0 + 0.11, height - 0.10, height, "concrete")
    elif abs(u1 - u0) < 1e-6:
        lo, hi = sorted([v0, v1])
        cuboid(surfaces, u0 - t, u0 + t, lo, hi, 0.00, 0.10, "concrete")
        cuboid(surfaces, u0 - t, u0 + t, lo, hi, 0.10, height - 0.10, material)
        cuboid(surfaces, u0 - 0.11, u0 + 0.11, lo, hi, height - 0.10, height, "concrete")
    else:
        raise ValueError("Wall segments must align to U or V")
    if posts:
        add_post(surfaces, *a, height + 0.05)
        add_post(surfaces, *b, height + 0.05)


def add_gate(surfaces, a, b, pedestrian=False):
    add_wall(surfaces, a, b, height=0.84 if pedestrian else 0.80, material="metal", posts=False)
    add_post(surfaces, *a, 1.02)
    add_post(surfaces, *b, 1.02)


def module_surfaces(spec):
    surfaces = []
    kind = spec[0]
    if kind == "segments":
        for a, b in spec[1]:
            add_wall(surfaces, a, b, posts=False)
        joints = {tuple(p) for seg in spec[1] for p in seg}
        for p in joints:
            add_post(surfaces, *p, 0.97)
    elif kind == "wall":
        add_wall(surfaces, spec[1], spec[2], height=spec[3] if len(spec) > 3 else 0.92)
    elif kind == "gate":
        add_gate(surfaces, spec[1], spec[2], len(spec) > 3 and spec[3] == "pedestrian")
    elif kind == "opening":
        add_post(surfaces, *spec[1], 1.02)
        add_post(surfaces, *spec[2], 1.02)
    elif kind == "broken":
        a, b = spec[1], spec[2]
        if a[1] == b[1]:
            mid = (a[0] + b[0]) / 2
            add_wall(surfaces, a, (mid - 0.45, a[1]), height=0.88)
            add_wall(surfaces, (mid + 0.45, a[1]), b, height=0.70)
        else:
            mid = (a[1] + b[1]) / 2
            add_wall(surfaces, a, (a[0], mid - 0.45), height=0.88)
            add_wall(surfaces, (a[0], mid + 0.45), b, height=0.70)
    elif kind == "post":
        add_post(surfaces, 0, 0, 1.02)
    return surfaces


def render_module(textures, spec):
    surfaces = module_surfaces(spec)
    points = np.vstack([project(p) for s in surfaces for p in s.quad3])
    lo = np.floor(points.min(axis=0) - 10)
    hi = np.ceil(points.max(axis=0) + 10)
    size = (hi - lo).astype(int)
    canvas = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    offset = -lo
    prepared = {
        (name, shade): shade_texture(tex, shade)
        for name, tex in textures.items()
        for shade in (0.68, 0.82, 1.05)
    }
    for s in sorted(surfaces, key=lambda x: x.depth):
        q = [project(p) + offset for p in s.quad3]
        paint_parallelogram(canvas, prepared[(s.material, s.shade)], q[0], q[1], q[3])
    image = Image.fromarray(canvas, "RGBA")
    bbox = image.getbbox()
    return image.crop((bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4))


MODULES = {
    "straight-short-u": ("wall", (-1, 0), (1, 0)),
    "straight-short-v": ("wall", (0, -1), (0, 1)),
    "straight-long-u": ("wall", (-2, 0), (2, 0)),
    "straight-long-v": ("wall", (0, -2), (0, 2)),
    "corner-ne": ("segments", [((0, 0), (2, 0)), ((0, 0), (0, -2))]),
    "corner-se": ("segments", [((0, 0), (2, 0)), ((0, 0), (0, 2))]),
    "corner-sw": ("segments", [((0, 0), (-2, 0)), ((0, 0), (0, 2))]),
    "corner-nw": ("segments", [((0, 0), (-2, 0)), ((0, 0), (0, -2))]),
    "t-north": ("segments", [((-2, 0), (0, 0)), ((0, 0), (2, 0)), ((0, 0), (0, -2))]),
    "t-south": ("segments", [((-2, 0), (0, 0)), ((0, 0), (2, 0)), ((0, 0), (0, 2))]),
    "t-east": ("segments", [((0, -2), (0, 0)), ((0, 0), (0, 2)), ((0, 0), (2, 0))]),
    "t-west": ("segments", [((0, -2), (0, 0)), ((0, 0), (0, 2)), ((0, 0), (-2, 0))]),
    "cross": ("segments", [((-2, 0), (0, 0)), ((0, 0), (2, 0)), ((0, -2), (0, 0)), ((0, 0), (0, 2))]),
    "vehicle-gate-u": ("gate", (-1.35, 0), (1.35, 0)),
    "vehicle-gate-v": ("gate", (0, -1.35), (0, 1.35)),
    "pedestrian-gate-u": ("gate", (-0.65, 0), (0.65, 0), "pedestrian"),
    "pedestrian-gate-v": ("gate", (0, -0.65), (0, 0.65), "pedestrian"),
    "open-entrance-u": ("opening", (-1.1, 0), (1.1, 0)),
    "open-entrance-v": ("opening", (0, -1.1), (0, 1.1)),
    "half-wall-u": ("wall", (-1.5, 0), (1.5, 0), 0.52),
    "half-wall-v": ("wall", (0, -1.5), (0, 1.5), 0.52),
    "broken-wall-u": ("broken", (-2, 0), (2, 0)),
    "broken-wall-v": ("broken", (0, -2), (0, 2)),
    "connector-post": ("post",),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    atlas = Image.open(ATLAS).convert("RGB")
    w, h = atlas.size
    textures = {
        "brick": atlas.crop((0, 0, int(w * 0.32), h)),
        "concrete": atlas.crop((int(w * 0.34), 0, int(w * 0.66), h)),
        "metal": atlas.crop((int(w * 0.68), 0, w, h)),
    }
    rendered = []
    for name, spec in MODULES.items():
        image = render_module(textures, spec)
        path = OUT / f"{name}.png"
        image.save(path)
        rendered.append((name, image))

    cols, cell_w, cell_h = 4, 410, 290
    rows = math.ceil(len(rendered) / cols)
    sheet = Image.new("RGBA", (cols * cell_w, rows * cell_h), (0, 0, 0, 0))
    preview = Image.new("RGB", sheet.size, (210, 210, 210))
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for i, (name, image) in enumerate(rendered):
        c, r = i % cols, i // cols
        thumb = image.copy()
        thumb.thumbnail((cell_w - 24, cell_h - 34), Image.Resampling.LANCZOS)
        x = c * cell_w + (cell_w - thumb.width) // 2
        y = r * cell_h + 22 + (cell_h - 26 - thumb.height) // 2
        sheet.alpha_composite(thumb, (x, y))
        preview.paste(thumb, (x, y), thumb)
        draw.text((c * cell_w + 8, r * cell_h + 6), name, fill=(25, 25, 25), font=font)
    sheet.save(OUT / "all-walls-transparent-sheet.png")
    preview.save(OUT / "all-walls-preview.png", quality=95)

    metadata = {
        "projection": "2:1 dimetric orthographic",
        "camera": {"elevation_degrees": 30, "azimuth_degrees": 45},
        "ground_basis_px": {"u": U.tolist(), "v": V.tolist()},
        "vertical_basis_px": Z.tolist(),
        "ground_angles_degrees": [26.565051177, -26.565051177],
        "perspective": False,
        "vanishing_point": False,
        "modules": [name for name, _ in rendered],
    }
    (OUT / "wall-kit.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
