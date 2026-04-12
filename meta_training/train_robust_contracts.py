"""Robust meta-training with parameterized abstract visual patterns.

This intentionally avoids literal, named "contract icon" classes. It combines:
1) original procedural episodes (for basic shape recognition continuity)
2) parameterized abstract episodes (for richer invariances)

Recommended:
    python meta_training/train_robust_contracts.py --hidden 192
"""

from __future__ import annotations

import argparse
import colorsys
import math
import os
import random
import sys
import time
from dataclasses import dataclass

import learn2learn as l2l
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.conv4 import Conv4WithHead
from meta_training.procedural_meta import generate_episode as generate_legacy_episode

IMG_SIZE = 84
CENTER = IMG_SIZE // 2
CKPT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "app", "models", "checkpoints"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _random_bg() -> tuple[int, int, int]:
    v = random.randint(228, 255)
    return (v, v, v)


def _random_color(
    hue_range: tuple[float, float] | None = None,
    sat_range: tuple[float, float] = (0.45, 1.0),
    val_range: tuple[float, float] = (0.45, 0.95),
) -> tuple[int, int, int]:
    h = random.uniform(*hue_range) if hue_range else random.random()
    s = random.uniform(*sat_range)
    v = random.uniform(*val_range)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _rotate_points(
    points: list[tuple[float, float]],
    angle_deg: float,
    cx: float,
    cy: float,
) -> list[tuple[float, float]]:
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    return [
        (
            cx + (x - cx) * cos_a - (y - cy) * sin_a,
            cy + (x - cx) * sin_a + (y - cy) * cos_a,
        )
        for x, y in points
    ]


def _superellipse_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    exponent: float,
    lobe_count: int = 0,
    lobe_depth: float = 0.0,
    phase: float = 0.0,
    n_points: int = 160,
) -> list[tuple[float, float]]:
    p = max(0.25, exponent)
    points: list[tuple[float, float]] = []
    for i in range(n_points):
        t = 2.0 * math.pi * i / n_points
        ct = math.cos(t)
        st = math.sin(t)
        x = math.copysign(abs(ct) ** (2.0 / p), ct)
        y = math.copysign(abs(st) ** (2.0 / p), st)
        if lobe_count > 0:
            mod = 1.0 + lobe_depth * math.sin(lobe_count * t + phase)
        else:
            mod = 1.0
        points.append((cx + rx * x * mod, cy + ry * y * mod))
    return points


def _draw_poly_outline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int,
):
    draw.polygon(points, outline=color)
    if width > 1:
        for i in range(len(points)):
            draw.line([points[i], points[(i + 1) % len(points)]], fill=color, width=width)


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    style: dict,
):
    mode = style["mode"]
    fg = style["fg_color"]
    accent = style["accent_color"]
    lw = style["line_width"]

    if mode == "outline":
        _draw_poly_outline(draw, points, fg, lw)
    elif mode == "filled_outline":
        draw.polygon(points, fill=fg, outline=accent)
        if lw > 1:
            for i in range(len(points)):
                draw.line(
                    [points[i], points[(i + 1) % len(points)]],
                    fill=accent,
                    width=lw,
                )
    else:
        draw.polygon(points, fill=fg)


def _shuffle_pair(imgs: list[torch.Tensor], labels: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    indices = list(range(len(imgs)))
    random.shuffle(indices)
    stacked = torch.stack([imgs[i] for i in indices])
    labs = torch.tensor([labels[i] for i in indices], dtype=torch.long)
    return stacked, labs


def _normalize_bbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    min_extent: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return a Pillow-safe bbox with sorted coordinates and minimum extent."""
    left = min(x0, x1)
    right = max(x0, x1)
    top = min(y0, y1)
    bottom = max(y0, y1)

    if right - left < min_extent:
        cx = (left + right) * 0.5
        half = min_extent * 0.5
        left, right = cx - half, cx + half
    if bottom - top < min_extent:
        cy = (top + bottom) * 0.5
        half = min_extent * 0.5
        top, bottom = cy - half, cy + half

    return left, top, right, bottom


def _sample_style() -> dict:
    fg = _random_color()
    accent = tuple(max(0, min(255, c + random.randint(-60, 40))) for c in fg)
    return {
        "bg_color": _random_bg(),
        "fg_color": fg,
        "accent_color": accent,
        "size": random.randint(14, 32),
        "rotation": random.uniform(0.0, 360.0),
        "offset_x": random.randint(-8, 8),
        "offset_y": random.randint(-8, 8),
        "line_width": random.randint(2, 5),
        "mode": random.choice(["fill", "fill", "fill", "filled_outline", "outline"]),
    }


# ---------------------------------------------------------------------------
# Pattern recipes
# ---------------------------------------------------------------------------


@dataclass
class PatternRecipe:
    family: str
    params: dict


FAMILY_SUPERELLIPSE = "superellipse"
FAMILY_WAVY_RIBBON = "wavy_ribbon"
FAMILY_PERFORATED = "perforated_region"
FAMILY_SPOKE_WEB = "spoke_web"
FAMILY_NESTED_BANDS = "nested_bands"
FAMILY_CURVY_STRIPES = "curvy_stripes"

ALL_FAMILIES = [
    FAMILY_SUPERELLIPSE,
    FAMILY_WAVY_RIBBON,
    FAMILY_PERFORATED,
    FAMILY_SPOKE_WEB,
    FAMILY_NESTED_BANDS,
    FAMILY_CURVY_STRIPES,
]


def _sample_superellipse_params() -> dict:
    return {
        "exponent": random.uniform(0.7, 4.5),           # round-ish <-> boxy
        "anisotropy": random.uniform(0.6, 1.5),         # x/y stretch ratio
        "lobe_count": random.randint(0, 8),             # number of radial bends
        "lobe_depth": random.uniform(0.0, 0.35),        # how curvy / dented
        "phase": random.uniform(0.0, 2.0 * math.pi),
    }


def _sample_wavy_ribbon_params() -> dict:
    return {
        "bend_count": random.randint(2, 8),             # number of bends
        "curviness": random.uniform(0.08, 0.55),        # bend amplitude
        "secondary_curviness": random.uniform(0.0, 0.2),
        "frequency_skew": random.uniform(0.7, 1.5),
        "stroke_width": random.randint(2, 7),
        "tail_connection": random.choice([False, False, True]),  # open/near-closed
    }


def _sample_perforated_params() -> dict:
    return {
        "outer_exponent": random.uniform(0.8, 3.8),
        "outer_lobe_count": random.randint(0, 6),
        "outer_lobe_depth": random.uniform(0.0, 0.25),
        "hole_count": random.randint(1, 7),             # holes few <-> many
        "hole_radius": random.uniform(0.10, 0.28),      # radius relative to size
        "hole_roundness": random.uniform(0.05, 1.0),    # corner radius factor
        "hole_aspect": random.uniform(0.6, 1.5),        # circular <-> elongated
        "hole_jitter": random.uniform(0.05, 0.35),      # irregular placement
        "phase": random.uniform(0.0, 2.0 * math.pi),
    }


def _sample_spoke_web_params() -> dict:
    return {
        "spoke_count": random.randint(4, 16),           # radial complexity
        "curvature": random.uniform(0.0, 0.55),         # spoke bow/curviness
        "inner_radius": random.uniform(0.08, 0.35),
        "ring_count": random.randint(0, 4),             # connector rings
        "length_jitter": random.uniform(0.0, 0.3),
        "stroke_width": random.randint(1, 4),
        "phase": random.uniform(0.0, 2.0 * math.pi),
    }


def _sample_nested_bands_params() -> dict:
    return {
        "band_count": random.randint(2, 8),             # sparse <-> dense
        "spacing_power": random.uniform(0.6, 1.8),      # linear <-> nonlinear spacing
        "eccentricity": random.uniform(0.55, 1.35),     # circle <-> ellipse
        "exponent": random.uniform(0.8, 4.5),           # band shape squareness
        "twist_per_band": random.uniform(-18.0, 18.0),  # progressive twist
        "band_width": random.randint(1, 4),
        "lobe_count": random.randint(0, 5),
        "lobe_depth": random.uniform(0.0, 0.15),
        "phase": random.uniform(0.0, 2.0 * math.pi),
    }


def _sample_curvy_stripes_params() -> dict:
    return {
        "stripe_spacing": random.randint(5, 14),        # sparse <-> dense
        "stripe_thickness": random.randint(1, 4),
        "bend_count": random.randint(2, 8),             # periodic complexity
        "curviness": random.uniform(0.08, 0.55),        # stripe waviness
        "crosshatch": random.choice([False, False, True]),
        "mask_exponent": random.uniform(0.8, 3.5),
        "mask_lobe_count": random.randint(0, 6),
        "mask_lobe_depth": random.uniform(0.0, 0.25),
        "phase": random.uniform(0.0, 2.0 * math.pi),
    }


FAMILY_SAMPLERS = {
    FAMILY_SUPERELLIPSE: _sample_superellipse_params,
    FAMILY_WAVY_RIBBON: _sample_wavy_ribbon_params,
    FAMILY_PERFORATED: _sample_perforated_params,
    FAMILY_SPOKE_WEB: _sample_spoke_web_params,
    FAMILY_NESTED_BANDS: _sample_nested_bands_params,
    FAMILY_CURVY_STRIPES: _sample_curvy_stripes_params,
}


def sample_recipe(family: str | None = None, overrides: dict | None = None) -> PatternRecipe:
    fam = family or random.choice(ALL_FAMILIES)
    params = FAMILY_SAMPLERS[fam]()
    if overrides:
        params.update(overrides)
    return PatternRecipe(family=fam, params=params)


def _recipe_signature(recipe: PatternRecipe) -> tuple:
    items = []
    for k, v in sorted(recipe.params.items()):
        if isinstance(v, float):
            items.append((k, round(v, 4)))
        else:
            items.append((k, v))
    return (recipe.family, tuple(items))


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_superellipse(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    style: dict,
    params: dict,
):
    anis = max(0.25, float(params["anisotropy"]))
    sx = math.sqrt(anis)
    sy = 1.0 / sx
    rx = size * sx
    ry = size * sy

    pts = _superellipse_points(
        cx=cx,
        cy=cy,
        rx=rx,
        ry=ry,
        exponent=float(params["exponent"]),
        lobe_count=int(params["lobe_count"]),
        lobe_depth=float(params["lobe_depth"]),
        phase=float(params["phase"]),
    )
    pts = _rotate_points(pts, style["rotation"], cx, cy)
    _draw_polygon(draw, pts, style)


def _render_wavy_ribbon(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    style: dict,
    params: dict,
):
    bends = max(1, int(params["bend_count"]))
    curviness = float(params["curviness"])
    secondary = float(params["secondary_curviness"])
    skew = float(params["frequency_skew"])
    phase = random.uniform(0.0, 2.0 * math.pi)
    phase2 = random.uniform(0.0, 2.0 * math.pi)

    points = []
    n = 120
    for i in range(n):
        t = i / max(1, n - 1)
        x = cx - size + 2.0 * size * t
        y = (
            cy
            + size * curviness * math.sin(2.0 * math.pi * bends * t + phase)
            + size * secondary * math.sin(2.0 * math.pi * (bends * skew + 1.0) * t + phase2)
        )
        points.append((x, y))

    points = _rotate_points(points, style["rotation"], cx, cy)
    width = max(1, int(params["stroke_width"]))
    draw.line(points, fill=style["fg_color"], width=width)

    if bool(params["tail_connection"]):
        draw.line([points[0], points[-1]], fill=style["accent_color"], width=max(1, width // 2))


def _render_perforated(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    style: dict,
    params: dict,
):
    outer_pts = _superellipse_points(
        cx=cx,
        cy=cy,
        rx=size,
        ry=size,
        exponent=float(params["outer_exponent"]),
        lobe_count=int(params["outer_lobe_count"]),
        lobe_depth=float(params["outer_lobe_depth"]),
        phase=float(params["phase"]),
    )
    outer_pts = _rotate_points(outer_pts, style["rotation"], cx, cy)
    draw.polygon(outer_pts, fill=style["fg_color"], outline=style["accent_color"])

    holes = max(0, int(params["hole_count"]))
    base_r = float(params["hole_radius"]) * size
    roundness = float(params["hole_roundness"])
    aspect = float(params["hole_aspect"])
    jitter = float(params["hole_jitter"])
    bg = style["bg_color"]

    centers: list[tuple[float, float]] = []
    attempts = 0
    while len(centers) < holes and attempts < holes * 35:
        attempts += 1
        a = random.uniform(0.0, 2.0 * math.pi)
        rad = random.uniform(0.05, 0.70) * (size - base_r * 1.3)
        hx = cx + rad * math.cos(a) + random.uniform(-size * jitter, size * jitter) * 0.25
        hy = cy + rad * math.sin(a) + random.uniform(-size * jitter, size * jitter) * 0.25

        ok = True
        for ox, oy in centers:
            if math.hypot(hx - ox, hy - oy) < base_r * 1.6:
                ok = False
                break
        if ok:
            centers.append((hx, hy))

    for hx, hy in centers:
        hw = base_r * random.uniform(0.75, 1.25)
        hh = hw * aspect * random.uniform(0.85, 1.15)
        x0, y0, x1, y1 = _normalize_bbox(hx - hw, hy - hh, hx + hw, hy + hh, min_extent=1.0)
        bbox = [x0, y0, x1, y1]
        if roundness > 0.92:
            draw.ellipse(bbox, fill=bg)
        elif roundness < 0.12:
            draw.rectangle(bbox, fill=bg)
        else:
            # Avoid Pillow rounded_rectangle edge cases on tiny bboxes:
            # use a superellipse approximation for intermediate roundness.
            w = max(1.0, x1 - x0)
            h = max(1.0, y1 - y0)
            cx_h = (x0 + x1) * 0.5
            cy_h = (y0 + y1) * 0.5
            exp = 2.0 + (1.0 - roundness) * 5.0  # round -> boxy
            hole_pts = _superellipse_points(
                cx=cx_h,
                cy=cy_h,
                rx=w * 0.5,
                ry=h * 0.5,
                exponent=exp,
                lobe_count=0,
                lobe_depth=0.0,
                phase=0.0,
                n_points=64,
            )
            draw.polygon(hole_pts, fill=bg)


def _render_spoke_web(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    style: dict,
    params: dict,
):
    spokes = max(1, int(params["spoke_count"]))
    curvature = float(params["curvature"])
    inner = float(params["inner_radius"]) * size
    outer = size * 0.95
    ring_count = max(0, int(params["ring_count"]))
    jitter = float(params["length_jitter"])
    width = max(1, int(params["stroke_width"]))
    phase = float(params["phase"])

    for i in range(spokes):
        theta = 2.0 * math.pi * i / spokes
        end_r = outer * (1.0 - jitter * random.uniform(0.0, 1.0))

        sx = cx + inner * math.cos(theta)
        sy = cy + inner * math.sin(theta)
        ex = cx + end_r * math.cos(theta)
        ey = cy + end_r * math.sin(theta)

        mid_r = (inner + end_r) * 0.55
        px = -math.sin(theta)
        py = math.cos(theta)
        bend = size * curvature * math.sin(i * 1.7 + phase)
        mx = cx + mid_r * math.cos(theta) + px * bend
        my = cy + mid_r * math.sin(theta) + py * bend

        pts = _rotate_points([(sx, sy), (mx, my), (ex, ey)], style["rotation"], cx, cy)
        draw.line(pts, fill=style["fg_color"], width=width)

    for r_idx in range(1, ring_count + 1):
        frac = r_idx / (ring_count + 1)
        rr = inner + (outer - inner) * frac
        ring_pts = []
        for i in range(spokes):
            theta = 2.0 * math.pi * i / spokes
            wobble = size * curvature * 0.12 * math.sin(theta * 2.0 + phase)
            ring_pts.append((cx + (rr + wobble) * math.cos(theta), cy + (rr + wobble) * math.sin(theta)))
        ring_pts = _rotate_points(ring_pts, style["rotation"], cx, cy)
        _draw_poly_outline(draw, ring_pts, style["accent_color"], 1)


def _render_nested_bands(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    style: dict,
    params: dict,
):
    bands = max(1, int(params["band_count"]))
    spacing_power = float(params["spacing_power"])
    ecc = max(0.25, float(params["eccentricity"]))
    exp = float(params["exponent"])
    twist = float(params["twist_per_band"])
    width = max(1, int(params["band_width"]))
    lobe_count = int(params["lobe_count"])
    lobe_depth = float(params["lobe_depth"])
    phase = float(params["phase"])

    for i in range(bands):
        frac = ((bands - i) / (bands + 0.25)) ** spacing_power
        rx = size * frac * math.sqrt(ecc)
        ry = size * frac / math.sqrt(ecc)
        pts = _superellipse_points(
            cx=cx,
            cy=cy,
            rx=rx,
            ry=ry,
            exponent=exp,
            lobe_count=lobe_count,
            lobe_depth=lobe_depth,
            phase=phase + i * 0.37,
        )
        pts = _rotate_points(pts, style["rotation"] + i * twist, cx, cy)
        _draw_poly_outline(draw, pts, style["fg_color"], width)


def _render_curvy_stripes(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: float,
    style: dict,
    params: dict,
):
    spacing = max(1, int(params["stripe_spacing"]))
    thickness = max(1, int(params["stripe_thickness"]))
    bends = max(1, int(params["bend_count"]))
    curviness = float(params["curviness"])
    cross = bool(params["crosshatch"])
    phase = float(params["phase"])

    pattern = Image.new("RGB", (IMG_SIZE, IMG_SIZE), style["bg_color"])
    pdraw = ImageDraw.Draw(pattern)

    amp = size * curviness
    for base in range(-IMG_SIZE, IMG_SIZE * 2, spacing):
        pts = []
        for x in range(-8, IMG_SIZE + 8, 2):
            t = x / max(1, IMG_SIZE - 1)
            y = base + amp * math.sin(2.0 * math.pi * bends * t + phase + base * 0.035)
            pts.append((x, y))
        pts = _rotate_points(pts, style["rotation"], cx, cy)
        pdraw.line(pts, fill=style["fg_color"], width=thickness)

    if cross:
        for base in range(-IMG_SIZE, IMG_SIZE * 2, spacing * 2):
            pts = []
            for y in range(-8, IMG_SIZE + 8, 2):
                t = y / max(1, IMG_SIZE - 1)
                x = base + amp * math.sin(2.0 * math.pi * bends * t + 1.7 * phase + base * 0.04)
                pts.append((x, y))
            pts = _rotate_points(pts, style["rotation"], cx, cy)
            pdraw.line(pts, fill=style["accent_color"], width=max(1, thickness - 1))

    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    mdraw = ImageDraw.Draw(mask)
    mask_pts = _superellipse_points(
        cx=cx,
        cy=cy,
        rx=size * 0.95,
        ry=size * 0.95,
        exponent=float(params["mask_exponent"]),
        lobe_count=int(params["mask_lobe_count"]),
        lobe_depth=float(params["mask_lobe_depth"]),
        phase=phase * 0.7,
    )
    mask_pts = _rotate_points(mask_pts, style["rotation"], cx, cy)
    mdraw.polygon(mask_pts, fill=255)

    img.paste(pattern, (0, 0), mask)
    _draw_poly_outline(draw, mask_pts, style["accent_color"], 1)


FAMILY_RENDERERS = {
    FAMILY_SUPERELLIPSE: _render_superellipse,
    FAMILY_WAVY_RIBBON: _render_wavy_ribbon,
    FAMILY_PERFORATED: _render_perforated,
    FAMILY_SPOKE_WEB: _render_spoke_web,
    FAMILY_NESTED_BANDS: _render_nested_bands,
    FAMILY_CURVY_STRIPES: _render_curvy_stripes,
}


def render_recipe(recipe: PatternRecipe, style_override: dict | None = None) -> torch.Tensor:
    style = _sample_style()
    if style_override:
        style.update(style_override)

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), style["bg_color"])
    draw = ImageDraw.Draw(img)

    cx = CENTER + style["offset_x"]
    cy = CENTER + style["offset_y"]
    size = style["size"]

    FAMILY_RENDERERS[recipe.family](img, draw, cx, cy, size, style, recipe.params)
    return _to_tensor(img)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


def _build_episode(
    class_0_fn,
    class_1_fn,
    n_support: int,
    n_query: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_imgs: list[torch.Tensor] = []
    s_labs: list[int] = []
    q_imgs: list[torch.Tensor] = []
    q_labs: list[int] = []

    for _ in range(n_support):
        s_imgs.append(class_0_fn())
        s_labs.append(0)
        s_imgs.append(class_1_fn())
        s_labs.append(1)

    for _ in range(n_query):
        q_imgs.append(class_0_fn())
        q_labs.append(0)
        q_imgs.append(class_1_fn())
        q_labs.append(1)

    s_img, s_lab = _shuffle_pair(s_imgs, s_labs)
    q_img, q_lab = _shuffle_pair(q_imgs, q_labs)
    return s_img, s_lab, q_img, q_lab


def generate_instance_episode(
    n_support: int = 8,
    n_query: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    recipe_0 = sample_recipe()
    recipe_1 = sample_recipe()

    # Ensure structural difference
    for _ in range(16):
        if _recipe_signature(recipe_0) != _recipe_signature(recipe_1):
            break
        recipe_1 = sample_recipe()

    def class_0():
        return render_recipe(recipe_0)

    def class_1():
        return render_recipe(recipe_1)

    return _build_episode(class_0, class_1, n_support, n_query)


def concept_curviness_low_vs_high():
    families = [FAMILY_SUPERELLIPSE, FAMILY_WAVY_RIBBON, FAMILY_CURVY_STRIPES]

    def class_0():
        fam = random.choice(families)
        if fam == FAMILY_SUPERELLIPSE:
            r = sample_recipe(fam, {"lobe_depth": random.uniform(0.00, 0.08)})
        elif fam == FAMILY_WAVY_RIBBON:
            r = sample_recipe(fam, {"curviness": random.uniform(0.08, 0.20)})
        else:
            r = sample_recipe(fam, {"curviness": random.uniform(0.08, 0.18)})
        return render_recipe(r)

    def class_1():
        fam = random.choice(families)
        if fam == FAMILY_SUPERELLIPSE:
            r = sample_recipe(fam, {"lobe_depth": random.uniform(0.20, 0.45)})
        elif fam == FAMILY_WAVY_RIBBON:
            r = sample_recipe(fam, {"curviness": random.uniform(0.30, 0.60)})
        else:
            r = sample_recipe(fam, {"curviness": random.uniform(0.30, 0.60)})
        return render_recipe(r)

    return class_0, class_1


def concept_bends_few_vs_many():
    families = [FAMILY_WAVY_RIBBON, FAMILY_CURVY_STRIPES]

    def class_0():
        fam = random.choice(families)
        r = sample_recipe(fam, {"bend_count": random.randint(2, 3)})
        return render_recipe(r)

    def class_1():
        fam = random.choice(families)
        r = sample_recipe(fam, {"bend_count": random.randint(6, 9)})
        return render_recipe(r)

    return class_0, class_1


def concept_holes_few_vs_many():
    def class_0():
        r = sample_recipe(FAMILY_PERFORATED, {"hole_count": random.randint(1, 2)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_PERFORATED, {"hole_count": random.randint(5, 8)})
        return render_recipe(r)

    return class_0, class_1


def concept_hole_roundness_sharp_vs_round():
    def class_0():
        r = sample_recipe(FAMILY_PERFORATED, {"hole_roundness": random.uniform(0.05, 0.25)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_PERFORATED, {"hole_roundness": random.uniform(0.70, 1.0)})
        return render_recipe(r)

    return class_0, class_1


def concept_spokes_few_vs_many():
    def class_0():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"spoke_count": random.randint(4, 7)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"spoke_count": random.randint(11, 16)})
        return render_recipe(r)

    return class_0, class_1


def concept_band_density_sparse_vs_dense():
    def class_0():
        r = sample_recipe(FAMILY_NESTED_BANDS, {"band_count": random.randint(2, 3)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_NESTED_BANDS, {"band_count": random.randint(6, 9)})
        return render_recipe(r)

    return class_0, class_1


def concept_squarish_vs_roundish():
    def class_0():
        # Round-ish superellipse
        r = sample_recipe(FAMILY_SUPERELLIPSE, {"exponent": random.uniform(1.6, 2.4)})
        return render_recipe(r)

    def class_1():
        # Boxier corners
        r = sample_recipe(FAMILY_SUPERELLIPSE, {"exponent": random.uniform(3.5, 6.0)})
        return render_recipe(r)

    return class_0, class_1


def concept_open_vs_closed():
    open_families = [FAMILY_WAVY_RIBBON, FAMILY_SPOKE_WEB]
    closed_families = [FAMILY_SUPERELLIPSE, FAMILY_PERFORATED, FAMILY_NESTED_BANDS]

    def class_0():
        fam = random.choice(open_families)
        if fam == FAMILY_SPOKE_WEB:
            r = sample_recipe(fam, {"ring_count": 0})
        else:
            r = sample_recipe(fam, {"tail_connection": False})
        return render_recipe(r, {"mode": "outline"})

    def class_1():
        fam = random.choice(closed_families)
        r = sample_recipe(fam)
        return render_recipe(r, {"mode": "fill"})

    return class_0, class_1


def concept_sparse_vs_dense_stripes():
    def class_0():
        r = sample_recipe(FAMILY_CURVY_STRIPES, {"stripe_spacing": random.randint(11, 16)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_CURVY_STRIPES, {"stripe_spacing": random.randint(4, 7)})
        return render_recipe(r)

    return class_0, class_1


def concept_random_family_pair():
    fam_0, fam_1 = random.sample(ALL_FAMILIES, 2)

    def class_0():
        return render_recipe(sample_recipe(fam_0))

    def class_1():
        return render_recipe(sample_recipe(fam_1))

    return class_0, class_1


def concept_anisotropy_tall_vs_wide():
    def class_0():
        r = sample_recipe(FAMILY_SUPERELLIPSE, {"anisotropy": random.uniform(0.45, 0.75)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_SUPERELLIPSE, {"anisotropy": random.uniform(1.35, 1.95)})
        return render_recipe(r)

    return class_0, class_1


def concept_lobes_few_vs_many():
    def class_0():
        r = sample_recipe(FAMILY_SUPERELLIPSE, {"lobe_count": random.randint(0, 2)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_SUPERELLIPSE, {"lobe_count": random.randint(6, 10)})
        return render_recipe(r)

    return class_0, class_1


def concept_hole_size_small_vs_large():
    def class_0():
        r = sample_recipe(FAMILY_PERFORATED, {"hole_radius": random.uniform(0.08, 0.14)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_PERFORATED, {"hole_radius": random.uniform(0.20, 0.32)})
        return render_recipe(r)

    return class_0, class_1


def concept_hole_shape_round_vs_elongated():
    def class_0():
        r = sample_recipe(
            FAMILY_PERFORATED,
            {"hole_roundness": random.uniform(0.75, 1.0), "hole_aspect": random.uniform(0.90, 1.10)},
        )
        return render_recipe(r)

    def class_1():
        r = sample_recipe(
            FAMILY_PERFORATED,
            {"hole_roundness": random.uniform(0.25, 0.65), "hole_aspect": random.uniform(1.35, 2.10)},
        )
        return render_recipe(r)

    return class_0, class_1


def concept_hole_layout_regular_vs_irregular():
    def class_0():
        r = sample_recipe(FAMILY_PERFORATED, {"hole_jitter": random.uniform(0.03, 0.10)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_PERFORATED, {"hole_jitter": random.uniform(0.25, 0.45)})
        return render_recipe(r)

    return class_0, class_1


def concept_spoke_curvature_low_vs_high():
    def class_0():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"curvature": random.uniform(0.00, 0.12)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"curvature": random.uniform(0.30, 0.60)})
        return render_recipe(r)

    return class_0, class_1


def concept_spoke_ringless_vs_ringed():
    def class_0():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"ring_count": 0})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"ring_count": random.randint(2, 5)})
        return render_recipe(r)

    return class_0, class_1


def concept_spoke_inner_hub_small_vs_large():
    def class_0():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"inner_radius": random.uniform(0.06, 0.14)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"inner_radius": random.uniform(0.24, 0.42)})
        return render_recipe(r)

    return class_0, class_1


def concept_spoke_length_regular_vs_jittered():
    def class_0():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"length_jitter": random.uniform(0.00, 0.08)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_SPOKE_WEB, {"length_jitter": random.uniform(0.20, 0.40)})
        return render_recipe(r)

    return class_0, class_1


def concept_band_twist_low_vs_high():
    def class_0():
        r = sample_recipe(FAMILY_NESTED_BANDS, {"twist_per_band": random.uniform(-4.0, 4.0)})
        return render_recipe(r)

    def class_1():
        sign = random.choice([-1.0, 1.0])
        r = sample_recipe(FAMILY_NESTED_BANDS, {"twist_per_band": sign * random.uniform(14.0, 24.0)})
        return render_recipe(r)

    return class_0, class_1


def concept_band_spacing_linear_vs_nonlinear():
    def class_0():
        r = sample_recipe(FAMILY_NESTED_BANDS, {"spacing_power": random.uniform(0.90, 1.10)})
        return render_recipe(r)

    def class_1():
        if random.random() < 0.5:
            power = random.uniform(0.45, 0.70)
        else:
            power = random.uniform(1.40, 2.00)
        r = sample_recipe(FAMILY_NESTED_BANDS, {"spacing_power": power})
        return render_recipe(r)

    return class_0, class_1


def concept_band_eccentricity_near_circle_vs_elongated():
    def class_0():
        r = sample_recipe(FAMILY_NESTED_BANDS, {"eccentricity": random.uniform(0.90, 1.10)})
        return render_recipe(r)

    def class_1():
        if random.random() < 0.5:
            ecc = random.uniform(0.45, 0.70)
        else:
            ecc = random.uniform(1.35, 1.90)
        r = sample_recipe(FAMILY_NESTED_BANDS, {"eccentricity": ecc})
        return render_recipe(r)

    return class_0, class_1


def concept_band_width_thin_vs_thick():
    def class_0():
        r = sample_recipe(FAMILY_NESTED_BANDS, {"band_width": random.randint(1, 2)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_NESTED_BANDS, {"band_width": random.randint(4, 6)})
        return render_recipe(r)

    return class_0, class_1


def concept_stripe_thin_vs_thick():
    def class_0():
        r = sample_recipe(FAMILY_CURVY_STRIPES, {"stripe_thickness": random.randint(1, 2)})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_CURVY_STRIPES, {"stripe_thickness": random.randint(4, 6)})
        return render_recipe(r)

    return class_0, class_1


def concept_stripe_crosshatch_off_vs_on():
    def class_0():
        r = sample_recipe(FAMILY_CURVY_STRIPES, {"crosshatch": False})
        return render_recipe(r)

    def class_1():
        r = sample_recipe(FAMILY_CURVY_STRIPES, {"crosshatch": True})
        return render_recipe(r)

    return class_0, class_1


def concept_stripe_mask_smooth_vs_lobed():
    def class_0():
        r = sample_recipe(
            FAMILY_CURVY_STRIPES,
            {"mask_lobe_count": random.randint(0, 1), "mask_lobe_depth": random.uniform(0.0, 0.05)},
        )
        return render_recipe(r)

    def class_1():
        r = sample_recipe(
            FAMILY_CURVY_STRIPES,
            {"mask_lobe_count": random.randint(4, 8), "mask_lobe_depth": random.uniform(0.12, 0.28)},
        )
        return render_recipe(r)

    return class_0, class_1


def concept_fill_vs_outline():
    def class_0():
        return render_recipe(sample_recipe(), {"mode": "fill"})

    def class_1():
        return render_recipe(sample_recipe(), {"mode": "outline"})

    return class_0, class_1


def concept_stroke_width_thin_vs_thick():
    def class_0():
        return render_recipe(sample_recipe(), {"line_width": random.randint(1, 2), "mode": "outline"})

    def class_1():
        return render_recipe(sample_recipe(), {"line_width": random.randint(4, 7), "mode": "outline"})

    return class_0, class_1


ALL_CONCEPTS = [
    concept_curviness_low_vs_high,
    concept_bends_few_vs_many,
    concept_holes_few_vs_many,
    concept_hole_roundness_sharp_vs_round,
    concept_spokes_few_vs_many,
    concept_band_density_sparse_vs_dense,
    concept_squarish_vs_roundish,
    concept_open_vs_closed,
    concept_sparse_vs_dense_stripes,
    concept_anisotropy_tall_vs_wide,
    concept_lobes_few_vs_many,
    concept_hole_size_small_vs_large,
    concept_hole_shape_round_vs_elongated,
    concept_hole_layout_regular_vs_irregular,
    concept_spoke_curvature_low_vs_high,
    concept_spoke_ringless_vs_ringed,
    concept_spoke_inner_hub_small_vs_large,
    concept_spoke_length_regular_vs_jittered,
    concept_band_twist_low_vs_high,
    concept_band_spacing_linear_vs_nonlinear,
    concept_band_eccentricity_near_circle_vs_elongated,
    concept_band_width_thin_vs_thick,
    concept_stripe_thin_vs_thick,
    concept_stripe_crosshatch_off_vs_on,
    concept_stripe_mask_smooth_vs_lobed,
    concept_fill_vs_outline,
    concept_stroke_width_thin_vs_thick,
    concept_random_family_pair,
]


def generate_concept_episode(
    n_support: int = 8,
    n_query: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    class_0_fn, class_1_fn = random.choice(ALL_CONCEPTS)()
    return _build_episode(class_0_fn, class_1_fn, n_support, n_query)


def generate_episode(
    n_support: int = 8,
    n_query: int = 8,
    instance_ratio: float = 0.70,
    legacy_shape_ratio: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Keep a substantial lane of original procedural tasks so the new model
    # stays strong on basic geometric categories.
    if random.random() < legacy_shape_ratio:
        return generate_legacy_episode(n_support=n_support, n_query=n_query)

    if random.random() < instance_ratio:
        return generate_instance_episode(n_support, n_query)
    return generate_concept_episode(n_support, n_query)


# ---------------------------------------------------------------------------
# Augmentation (robustness)
# ---------------------------------------------------------------------------


def _augment_batch(images: torch.Tensor) -> torch.Tensor:
    x = images.clone()
    n = x.size(0)

    # Brightness jitter
    if random.random() < 0.85:
        gains = torch.empty((n, 1, 1, 1), dtype=x.dtype).uniform_(0.65, 1.35)
        x = x * gains

    # Additive Gaussian noise
    if random.random() < 0.80:
        noise_std = random.uniform(0.010, 0.060)
        x = x + torch.randn_like(x) * noise_std

    # Local occlusion / cutout
    if random.random() < 0.45:
        for i in range(n):
            h = random.randint(6, 20)
            w = random.randint(6, 20)
            y0 = random.randint(0, IMG_SIZE - h)
            x0 = random.randint(0, IMG_SIZE - w)
            patch = x[i, :, y0 : y0 + h, x0 : x0 + w]
            x[i, :, y0 : y0 + h, x0 : x0 + w] = patch.mean()

    return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=192, help="Conv4 hidden channels")
    parser.add_argument("--iterations", type=int, default=4000, help="Meta-training iterations")
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--outer-lr", type=float, default=8e-4)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--tasks-per-batch", type=int, default=8)
    parser.add_argument("--support", type=int, default=8, help="Support examples per class")
    parser.add_argument("--query", type=int, default=8, help="Query examples per class")
    parser.add_argument(
        "--instance-ratio",
        type=float,
        default=0.70,
        help="Within abstract lane, fraction of episodes that are instance-discrimination vs concept episodes",
    )
    parser.add_argument(
        "--legacy-shape-ratio",
        type=float,
        default=0.35,
        help="Fraction of episodes sampled from original procedural_meta generator",
    )
    parser.add_argument("--augment-prob", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument(
        "--save-threshold",
        type=float,
        default=0.80,
        help="Save extra checkpoints when accuracy is >= this value",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Check/save threshold checkpoint every N iterations",
    )
    # Backward-compatible no-op flag from earlier version of this script.
    parser.add_argument("--contract-mix", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.contract_mix is not None:
        print(
            "Note: --contract-mix is deprecated and ignored. "
            "Literal contract shapes were removed."
        )

    instance_ratio = max(0.05, min(0.95, args.instance_ratio))
    legacy_shape_ratio = max(0.0, min(0.90, args.legacy_shape_ratio))
    augment_prob = max(0.0, min(1.0, args.augment_prob))
    save_threshold = max(0.0, min(1.0, args.save_threshold))
    save_interval = max(1, int(args.save_interval))

    print(f"Device: {device}")
    print(f"Backbone: Conv4-{args.hidden} ({args.hidden * 5 * 5}-dim features)")
    print("Data: parameterized abstract patterns only (no literal icon classes)")

    model = Conv4WithHead(num_classes=2, hidden=args.hidden).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    maml = l2l.algorithms.MAML(model, lr=args.inner_lr, first_order=True)
    optimizer = torch.optim.AdamW(maml.parameters(), lr=args.outer_lr, weight_decay=1e-4)
    os.makedirs(CKPT_DIR, exist_ok=True)
    save_name = args.save_name or f"general_conv4_{args.hidden}_robust.pt"
    base_ckpt_path = os.path.join(CKPT_DIR, save_name)
    stem, ext = os.path.splitext(save_name)
    if not ext:
        ext = ".pt"

    print(
        f"\nTraining {args.iterations} iterations | "
        f"{args.tasks_per_batch} tasks/batch | "
        f"{args.support}-shot/{args.query}-query (per class) | "
        f"legacy_shape_ratio={legacy_shape_ratio:.2f} | "
        f"instance_ratio={instance_ratio:.2f} | augment_prob={augment_prob:.2f} | "
        f"save>= {save_threshold:.0%} every {save_interval} iters\n"
    )

    t0 = time.time()
    best_acc = 0.0
    best_state = None

    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        meta_loss = 0.0
        meta_acc = 0.0

        for _ in range(args.tasks_per_batch):
            learner = maml.clone()
            s_img, s_lab, q_img, q_lab = generate_episode(
                n_support=args.support,
                n_query=args.query,
                instance_ratio=instance_ratio,
                legacy_shape_ratio=legacy_shape_ratio,
            )

            if random.random() < augment_prob:
                s_img = _augment_batch(s_img)
                q_img = _augment_batch(q_img)

            s_img, s_lab = s_img.to(device), s_lab.to(device)
            q_img, q_lab = q_img.to(device), q_lab.to(device)

            for _ in range(args.inner_steps):
                preds = learner(s_img)
                learner.adapt(F.cross_entropy(preds, s_lab))

            preds = learner(q_img)
            task_loss = F.cross_entropy(preds, q_lab)
            task_acc = (preds.argmax(1) == q_lab).float().mean().item()

            task_loss.backward()
            meta_loss += task_loss.item()
            meta_acc += task_acc

        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.div_(args.tasks_per_batch)
        torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=5.0)
        optimizer.step()

        avg_loss = meta_loss / args.tasks_per_batch
        avg_acc = meta_acc / args.tasks_per_batch
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_state = {k: v.detach().cpu().clone() for k, v in maml.state_dict().items()}

        if iteration % save_interval == 0 and avg_acc >= save_threshold:
            acc_pct = avg_acc * 100.0
            threshold_ckpt_path = os.path.join(
                CKPT_DIR,
                f"{stem}_iter{iteration:05d}_acc{acc_pct:05.2f}{ext}",
            )
            torch.save(maml.state_dict(), threshold_ckpt_path)
            print(f"Saved threshold checkpoint: {threshold_ckpt_path}", flush=True)

        if iteration % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"Iter {iteration:>5}/{args.iterations} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {avg_acc:.2%} | "
                f"Best: {best_acc:.2%} | "
                f"Time: {elapsed:.0f}s",
                flush=True,
            )

    torch.save(best_state if best_state is not None else maml.state_dict(), base_ckpt_path)

    print(f"\nSaved: {base_ckpt_path}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print(f"Best meta-batch acc: {best_acc:.2%}")


if __name__ == "__main__":
    main()
