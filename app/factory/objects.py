"""Procedural object generator for the factory idle game.

Generates 84x84 RGB images of geometric shapes with high per-instance
variation so that classification is genuinely challenging.  Uses Pillow
for headless rendering -- no pygame dependency.
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS_SIZE = 84
CENTER = CANVAS_SIZE // 2  # 42

SHAPE_FAMILIES: dict[str, list[str]] = {
    "pointy": ["star_4", "star_5", "star_6", "arrow", "lightning"],
    "angular": ["triangle", "right_triangle", "diamond", "parallelogram", "trapezoid"],
    "rounded": ["circle", "oval", "semicircle"],
    "boxy": ["square", "rectangle", "cross"],
    "organic": ["heart", "crescent"],
}

# Ordered so early categories span different families (easier to distinguish)
ALL_CATEGORIES: list[str] = [
    "circle", "triangle", "star_5", "square",       # 4 starting: one from each family
    "diamond", "oval", "cross", "heart",             # 4 more, diverse
    "arrow", "rectangle", "semicircle", "crescent",  # varied
    "star_4", "right_triangle", "parallelogram",     # harder
    "star_6", "trapezoid", "lightning",              # hardest
]

_CATEGORY_TO_FAMILY: dict[str, str] = {
    cat: fam for fam, cats in SHAPE_FAMILIES.items() for cat in cats
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class FactoryObject:
    tensor: torch.Tensor          # (3, 84, 84) normalised float image
    category: str                 # ground-truth category name
    attributes: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Random attribute sampling
# ---------------------------------------------------------------------------

def _sample_attributes(difficulty: float = 0.0) -> dict:
    """Sample all continuous variation axes for one object.

    *difficulty* in [0, 1] controls how much variation is applied.
    0 = clean, centered shapes (easy for ML).  1 = maximum variation.
    """
    d = max(0.0, min(1.0, difficulty))

    hue = random.random()
    saturation = random.uniform(0.6, 1.0)
    value = random.uniform(0.6, 0.9)
    size = random.randint(20, 38)
    rotation_deg = random.uniform(0.0, 45.0 * d)     # no rotation at d=0
    outline_width = random.choice([0, 0, 0, 2])       # mostly filled
    offset_x = random.randint(int(-8 * d), int(8 * d))
    offset_y = random.randint(int(-8 * d), int(8 * d))
    deformation = random.uniform(0.0, 0.12 * d)       # no deformation at d=0
    bg_tint_r = random.randint(240, 255)
    bg_tint_g = random.randint(240, 255)
    bg_tint_b = random.randint(240, 255)

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    fill_color = (int(r * 255), int(g * 255), int(b * 255))

    return {
        "hue": round(hue * 360, 1),
        "saturation": round(saturation, 3),
        "value": round(value, 3),
        "fill_color": fill_color,
        "size": size,
        "rotation_deg": round(rotation_deg, 1),
        "outline_width": outline_width,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "deformation": round(deformation, 4),
        "bg_color": (bg_tint_r, bg_tint_g, bg_tint_b),
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotate(points: list[tuple[float, float]], angle_deg: float,
            cx: float, cy: float) -> list[tuple[float, float]]:
    """Rotate *points* around (cx, cy) by *angle_deg* degrees."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    out = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        out.append((cx + dx * cos_a - dy * sin_a,
                     cy + dx * sin_a + dy * cos_a))
    return out


def _deform(points: list[tuple[float, float]], strength: float,
            size: float) -> list[tuple[float, float]]:
    """Add random jitter to each vertex proportional to *size*."""
    if strength <= 0:
        return points
    max_offset = strength * size
    return [(x + random.uniform(-max_offset, max_offset),
             y + random.uniform(-max_offset, max_offset))
            for x, y in points]


def _transform(points: list[tuple[float, float]], attrs: dict,
               cx: float, cy: float) -> list[tuple[float, float]]:
    """Apply rotation then deformation to a vertex list."""
    pts = _rotate(points, attrs["rotation_deg"], cx, cy)
    pts = _deform(pts, attrs["deformation"], attrs["size"])
    return pts


# ---------------------------------------------------------------------------
# Shape vertex generators
#
# Each returns raw vertices centered at (cx, cy) with the given *size*
# (half-extent), **before** rotation / deformation.
# ---------------------------------------------------------------------------

def _star_vertices(cx: float, cy: float, size: float,
                   n_points: int) -> list[tuple[float, float]]:
    outer_r = size
    inner_r = size * random.uniform(0.35, 0.55)
    pts: list[tuple[float, float]] = []
    for i in range(n_points * 2):
        angle = math.pi / 2 + i * math.pi / n_points
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    return pts


def _triangle_vertices(cx: float, cy: float,
                       size: float) -> list[tuple[float, float]]:
    # Equilateral-ish triangle
    h = size * math.sqrt(3) / 2
    return [
        (cx, cy - size),
        (cx - h, cy + size * 0.5),
        (cx + h, cy + size * 0.5),
    ]


def _right_triangle_vertices(cx: float, cy: float,
                              size: float) -> list[tuple[float, float]]:
    return [
        (cx - size, cy + size),
        (cx - size, cy - size),
        (cx + size, cy + size),
    ]


def _diamond_vertices(cx: float, cy: float,
                      size: float) -> list[tuple[float, float]]:
    stretch = random.uniform(0.6, 0.9)
    return [
        (cx, cy - size),
        (cx + size * stretch, cy),
        (cx, cy + size),
        (cx - size * stretch, cy),
    ]


def _parallelogram_vertices(cx: float, cy: float,
                             size: float) -> list[tuple[float, float]]:
    skew = size * random.uniform(0.3, 0.6)
    h = size * random.uniform(0.5, 0.9)
    return [
        (cx - size + skew, cy - h),
        (cx + size + skew, cy - h),
        (cx + size - skew, cy + h),
        (cx - size - skew, cy + h),
    ]


def _trapezoid_vertices(cx: float, cy: float,
                        size: float) -> list[tuple[float, float]]:
    top_half = size * random.uniform(0.35, 0.7)
    h = size * random.uniform(0.6, 1.0)
    return [
        (cx - top_half, cy - h),
        (cx + top_half, cy - h),
        (cx + size, cy + h),
        (cx - size, cy + h),
    ]


def _square_vertices(cx: float, cy: float,
                     size: float) -> list[tuple[float, float]]:
    return [
        (cx - size, cy - size),
        (cx + size, cy - size),
        (cx + size, cy + size),
        (cx - size, cy + size),
    ]


def _rectangle_vertices(cx: float, cy: float,
                        size: float) -> list[tuple[float, float]]:
    aspect = random.uniform(0.4, 0.7)
    w, h = size, size * aspect
    # Randomly orient tall vs wide
    if random.random() < 0.5:
        w, h = h, w
    return [
        (cx - w, cy - h),
        (cx + w, cy - h),
        (cx + w, cy + h),
        (cx - w, cy + h),
    ]


def _cross_vertices(cx: float, cy: float,
                    size: float) -> list[tuple[float, float]]:
    arm = size * random.uniform(0.25, 0.45)
    return [
        (cx - arm, cy - size),
        (cx + arm, cy - size),
        (cx + arm, cy - arm),
        (cx + size, cy - arm),
        (cx + size, cy + arm),
        (cx + arm, cy + arm),
        (cx + arm, cy + size),
        (cx - arm, cy + size),
        (cx - arm, cy + arm),
        (cx - size, cy + arm),
        (cx - size, cy - arm),
        (cx - arm, cy - arm),
    ]


def _arrow_vertices(cx: float, cy: float,
                    size: float) -> list[tuple[float, float]]:
    shaft_w = size * random.uniform(0.25, 0.45)
    head_w = size * random.uniform(0.7, 1.0)
    head_len = size * random.uniform(0.4, 0.65)
    return [
        (cx, cy - size),                   # tip
        (cx + head_w, cy - size + head_len),
        (cx + shaft_w, cy - size + head_len),
        (cx + shaft_w, cy + size),
        (cx - shaft_w, cy + size),
        (cx - shaft_w, cy - size + head_len),
        (cx - head_w, cy - size + head_len),
    ]


def _lightning_vertices(cx: float, cy: float,
                        size: float) -> list[tuple[float, float]]:
    w = size * random.uniform(0.5, 0.8)
    jag = size * random.uniform(0.15, 0.35)
    return [
        (cx + w * 0.2, cy - size),
        (cx + w, cy - size * 0.3),
        (cx + jag, cy - size * 0.1),
        (cx + w * 0.6, cy + size * 0.5),
        (cx - w * 0.2, cy + size * 0.1),
        (cx + jag * 0.5, cy + size),
        (cx - w, cy + size * 0.15),
        (cx - jag, cy + size * 0.05),
        (cx - w * 0.6, cy - size * 0.4),
    ]


# ---------------------------------------------------------------------------
# Parametric / arc-based shapes (drawn directly, not vertex-based)
# ---------------------------------------------------------------------------

def _draw_heart(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                size: float, attrs: dict):
    """Parametric heart curve rendered as a filled/outlined polygon."""
    n_samples = 80
    pts: list[tuple[float, float]] = []
    for i in range(n_samples):
        t = 2.0 * math.pi * i / n_samples
        x = 16.0 * math.sin(t) ** 3
        y = -(13.0 * math.cos(t) - 5.0 * math.cos(2 * t)
              - 2.0 * math.cos(3 * t) - math.cos(4 * t))
        # Scale into size (raw curve spans ~-16..16 in x, ~-17..15 in y)
        pts.append((cx + x * size / 17.0, cy + y * size / 17.0))
    pts = _transform(pts, attrs, cx, cy)
    _draw_poly(draw, pts, attrs)


def _draw_crescent(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                   size: float, attrs: dict):
    """Two overlapping ellipses with background fill to cut out the bite."""
    # Build a crescent as a polygon by sampling the outer circle minus inner
    n = 80
    outer_r = size
    inner_r = size * random.uniform(0.65, 0.85)
    inner_offset_x = size * random.uniform(0.25, 0.5)
    inner_offset_y = size * random.uniform(-0.15, 0.15)

    # Outer arc  (full circle)
    outer_pts = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        outer_pts.append((cx + outer_r * math.cos(a),
                          cy + outer_r * math.sin(a)))

    # Inner arc (full circle, shifted) -- we'll subtract
    inner_cx = cx + inner_offset_x
    inner_cy = cy + inner_offset_y
    inner_pts = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        inner_pts.append((inner_cx + inner_r * math.cos(a),
                          inner_cy + inner_r * math.sin(a)))

    # Keep only outer points that are NOT inside the inner circle
    crescent: list[tuple[float, float]] = []
    for px, py in outer_pts:
        dist = math.hypot(px - inner_cx, py - inner_cy)
        if dist >= inner_r * 0.95:
            crescent.append((px, py))

    if len(crescent) < 4:
        crescent = outer_pts  # fallback

    crescent = _transform(crescent, attrs, cx, cy)
    _draw_poly(draw, crescent, attrs)


def _draw_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                 size: float, attrs: dict):
    """Draw a filled / outlined circle (no vertices to transform)."""
    # Apply slight deformation via random per-sample jitter
    n = 64
    pts: list[tuple[float, float]] = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        r = size + random.uniform(-size * attrs["deformation"],
                                   size * attrs["deformation"])
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts = _rotate(pts, attrs["rotation_deg"], cx, cy)
    _draw_poly(draw, pts, attrs)


def _draw_oval(draw: ImageDraw.ImageDraw, cx: float, cy: float,
               size: float, attrs: dict):
    aspect = random.uniform(0.45, 0.75)
    rx, ry = size, size * aspect
    if random.random() < 0.5:
        rx, ry = ry, rx
    n = 64
    pts: list[tuple[float, float]] = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    pts = _transform(pts, attrs, cx, cy)
    _draw_poly(draw, pts, attrs)


def _draw_semicircle(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                     size: float, attrs: dict):
    n = 40
    pts: list[tuple[float, float]] = []
    # Half-circle arc (top half)
    for i in range(n + 1):
        a = math.pi * i / n
        pts.append((cx + size * math.cos(a), cy - size * math.sin(a)))
    # Close with the diameter line
    pts.append((cx + size, cy))
    pts.append((cx - size, cy))
    pts = _transform(pts, attrs, cx, cy)
    _draw_poly(draw, pts, attrs)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_poly(draw: ImageDraw.ImageDraw,
               pts: list[tuple[float, float]], attrs: dict):
    """Draw a polygon either filled or with an outline."""
    coords = [(round(x), round(y)) for x, y in pts]
    outline_w = attrs["outline_width"]
    fill = attrs["fill_color"]
    if outline_w == 0:
        draw.polygon(coords, fill=fill)
    else:
        # Darker outline derived from fill
        outline_color = tuple(max(0, c - 60) for c in fill)
        draw.polygon(coords, fill=fill, outline=outline_color)
        # Pillow polygon outline is always 1px; fake wider with lines
        if outline_w > 1:
            for i in range(len(coords)):
                draw.line([coords[i], coords[(i + 1) % len(coords)]],
                          fill=outline_color, width=outline_w)


# ---------------------------------------------------------------------------
# Master render dispatch
# ---------------------------------------------------------------------------

_POLYGON_SHAPES: dict[str, object] = {
    "star_4": lambda cx, cy, s: _star_vertices(cx, cy, s, 4),
    "star_5": lambda cx, cy, s: _star_vertices(cx, cy, s, 5),
    "star_6": lambda cx, cy, s: _star_vertices(cx, cy, s, 6),
    "triangle": _triangle_vertices,
    "right_triangle": _right_triangle_vertices,
    "diamond": _diamond_vertices,
    "parallelogram": _parallelogram_vertices,
    "trapezoid": _trapezoid_vertices,
    "square": _square_vertices,
    "rectangle": _rectangle_vertices,
    "cross": _cross_vertices,
    "arrow": _arrow_vertices,
    "lightning": _lightning_vertices,
}

_DIRECT_DRAW_SHAPES: dict[str, object] = {
    "heart": _draw_heart,
    "crescent": _draw_crescent,
    "circle": _draw_circle,
    "oval": _draw_oval,
    "semicircle": _draw_semicircle,
}


def _render(category: str, attrs: dict) -> Image.Image:
    """Render one shape onto an 84x84 RGB image."""
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), attrs["bg_color"])
    draw = ImageDraw.Draw(img)

    cx = CENTER + attrs["offset_x"]
    cy = CENTER + attrs["offset_y"]
    size = attrs["size"]

    if category in _POLYGON_SHAPES:
        raw_pts = _POLYGON_SHAPES[category](cx, cy, size)
        pts = _transform(raw_pts, attrs, cx, cy)
        _draw_poly(draw, pts, attrs)
    elif category in _DIRECT_DRAW_SHAPES:
        _DIRECT_DRAW_SHAPES[category](draw, cx, cy, size, attrs)
    else:
        raise ValueError(f"Unknown category: {category!r}")

    return img


# ---------------------------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------------------------

def _image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a (3, H, W) float tensor in [0, 1]."""
    arr = np.array(img, dtype=np.float32)       # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1) / 255.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ObjectGenerator:
    """Procedural generator for factory objects with high per-instance variety."""

    ALL_CATEGORIES: list[str] = ALL_CATEGORIES
    SHAPE_FAMILIES: dict[str, list[str]] = SHAPE_FAMILIES

    def __init__(self, difficulty: float = 0.0):
        self.difficulty = difficulty

    def generate(self, category: str | None = None) -> FactoryObject:
        """Generate one random object.

        If *category* is ``None``, a category is chosen uniformly at random.
        """
        if category is None:
            category = random.choice(self.ALL_CATEGORIES)
        if category not in _CATEGORY_TO_FAMILY:
            raise ValueError(
                f"Unknown category {category!r}. "
                f"Valid categories: {self.ALL_CATEGORIES}"
            )

        attrs = _sample_attributes(self.difficulty)
        attrs["shape_family"] = _CATEGORY_TO_FAMILY[category]

        img = _render(category, attrs)
        tensor = _image_to_tensor(img)

        return FactoryObject(
            tensor=tensor,
            category=category,
            attributes=attrs,
        )

    def generate_batch(
        self,
        n: int,
        categories: list[str] | None = None,
    ) -> list[FactoryObject]:
        """Generate *n* objects, optionally restricted to *categories*."""
        pool = categories if categories else self.ALL_CATEGORIES
        return [self.generate(random.choice(pool)) for _ in range(n)]

    def generate_balanced_batch(
        self,
        per_class: int,
        categories: list[str] | None = None,
    ) -> list[FactoryObject]:
        """Generate exactly *per_class* objects for every category.

        Returns a list of length ``per_class * len(categories)`` in
        shuffled order.
        """
        cats = categories if categories else self.ALL_CATEGORIES
        objects: list[FactoryObject] = []
        for cat in cats:
            for _ in range(per_class):
                objects.append(self.generate(cat))
        random.shuffle(objects)
        return objects
