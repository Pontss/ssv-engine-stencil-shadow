"""Deterministic procedural material maps; no external asset downloads required."""
from __future__ import annotations
import numpy as np
import moderngl


def _noise(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((size, size), dtype=np.float32)


def make_material_pixels(kind: str, size: int = 256) -> np.ndarray:
    if kind not in {"floor", "crate", "stone", "wood"}:
        raise ValueError("Material kind must be floor, crate, stone or wood")
    if size < 16 or size % 4:
        raise ValueError("Material texture size must be at least 16 and divisible by 4")
    yy, xx = np.mgrid[0:size, 0:size]
    seed = {"floor": 19, "crate": 41, "stone": 73, "wood": 97}[kind]
    noise = _noise(size, seed)
    grain = _noise(size, seed + 1)
    fine_grain = _noise(size, seed + 2)
    if kind == "floor":
        # Sun-warmed travertine paving: four broad slabs per repeat, with
        # readable recessed joints, alternating stone tones and fine pores.
        cell = max(32, size // 2)
        local_x, local_y = xx % cell, yy % cell
        edge = (local_x < 4) | (local_y < 4) | (local_x > cell-5) | (local_y > cell-5)
        slab_id = ((xx // cell) + (yy // cell)) % 2
        variation = ((noise - .5) * 17.0 + (grain - .5) * 8.0
                     + (fine_grain - .5) * 5.0
                     + 2.0 * np.sin(xx * .11) * np.sin(yy * .09))
        warmth = slab_id * 7.0
        base = np.stack((184 + variation + warmth, 164 + variation * .94 + warmth,
                         132 + variation * .82 + warmth * .75), axis=-1)
        base[edge] = np.array([105, 87, 66])
        flecks = (noise > .991) & ~edge
        base[flecks] = np.array([211, 190, 153])
    elif kind == "crate":
        # Painted industrial alloy panels with dark seams, amber safety marks and rivets.
        panel_x, panel_y = xx % 128, yy % 128
        seam = (panel_x < 4) | (panel_y < 4) | (panel_x > 123) | (panel_y > 123)
        stripe = ((panel_x > 12) & (panel_x < 19) & ((panel_y // 16) % 2 == 0))
        rivet = (((panel_x-10)**2 + (panel_y-10)**2) < 7) | (((panel_x-118)**2 + (panel_y-118)**2) < 7)
        brushed = ((noise - .5) * 18.0 + (grain - .5) * 8.0
                   + (fine_grain - .5) * 4.0 + 2.8 * np.sin(yy * .65))
        base = np.stack((89 + brushed * .90, 109 + brushed, 127 + brushed * 1.08), axis=-1)
        bevel = ((panel_x == 4) | (panel_y == 4) | (panel_x == 123) | (panel_y == 123))
        base[seam] = np.array([27, 38, 50])
        base[bevel & ~seam] = np.array([126, 143, 155])
        base[stripe] = np.array([205, 119, 43])
        base[rivet] = np.array([174, 187, 194])
    elif kind == "wood":
        # Clean, warm oak planks: long restrained grain, no scratches or seams
        # crossing the bench surface. UVs repeat per face, so grain stays calm.
        flow = yy.astype(np.float32)
        wave = 5.0 * np.sin(flow * 0.075 + 2.0 * np.sin(xx * 0.018))
        grain_lines = np.abs(np.sin(flow * 0.31 + np.sin(xx * 0.025) * 1.4)) < 0.045
        variation = (noise - .5) * 7.0 + (grain - .5) * 4.0 + wave
        base = np.stack((126 + variation, 78 + variation * .78, 43 + variation * .55), axis=-1)
        base[grain_lines] *= 0.91
    else:
        # Warm limestone / travertine with subtle block joints and fine pores.
        cell = max(32, size // 2)
        local_x, local_y = xx % cell, yy % cell
        joints = (local_x < 3) | (local_y < 3)
        broad = 10.0 * np.sin(xx * 0.055 + np.sin(yy * 0.031) * 2.0)
        pores = (fine_grain > 0.992)
        vein = (np.abs(np.sin(xx * 0.085 + np.sin(yy * 0.04) * 2.2)) < 0.055)
        variation = (noise - .5) * 18.0 + (grain - .5) * 12.0 + (fine_grain - .5) * 7.0 + broad
        base = np.stack((157 + variation, 132 + variation * .92, 98 + variation * .78), axis=-1)
        base[vein & ~joints] *= 0.94
        base[pores & ~joints] *= 0.96
        base[joints] = np.array([126, 108, 83])
    rgb = np.clip(base, 0, 255).astype(np.uint8)
    # Alpha stores a subtle height field used by the fragment shader for bump detail.
    height = np.clip(128 + (grain - .5) * 110 + (fine_grain - .5) * 70, 0, 255).astype(np.uint8)
    if kind == "floor":
        height[edge] = 48
    elif kind == "crate":
        height[seam] = 45
        height[rivet] = 210
    elif kind == "stone":
        height[joints] = 72
        height[pores] = 88
    elif kind == "wood":
        height = np.clip(128 + (grain - .5) * 24 + (fine_grain - .5) * 12, 0, 255).astype(np.uint8)
    rgba = np.concatenate((rgb, height[:, :, None]), axis=2)
    return rgba


def make_material_texture(context: moderngl.Context, kind: str, size: int = 256):
    rgba = make_material_pixels(kind, size)
    texture = context.texture((size,size),4,rgba.tobytes(),alignment=1)
    texture.repeat_x = True
    texture.repeat_y = True
    texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    texture.build_mipmaps()
    return texture
