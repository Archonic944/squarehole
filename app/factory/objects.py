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
    "organic": ["heart", "crescent", "cloud", "teardrop"],
    "holed": ["donut", "picture_frame", "key", "gear"],
    "composite": ["mushroom", "tree", "flower", "candy_cane", "rainbow"],
}

# Ordered so early categories span different families (easier to distinguish)
ALL_CATEGORIES: list[str] = [
    "circle", "triangle", "star_5", "square",       # 4 starting: one from each family
    "diamond", "oval", "cross", "heart",             # 4 more, diverse
    "arrow", "rectangle", "semicircle", "crescent",  # varied
    "star_4", "right_triangle", "parallelogram",     # harder
    "star_6", "trapezoid", "lightning",              # hardest
    # New packs
    "cloud", "teardrop",
    "donut", "picture_frame", "key", "gear",
    "mushroom", "tree", "flower", "candy_cane", "rainbow",
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


def _cloud_vertices(cx: float, cy: float,
                    size: float) -> list[tuple[float, float]]:
    """Bumpy round blob built from several overlapping circular lobes."""
    # A cloud is approximated by sampling the union of 4-5 circles
    # whose centers sit along a horizontal axis at varying heights.
    n_lobes = random.randint(4, 5)
    lobes: list[tuple[float, float, float]] = []  # (cx, cy, r)
    # Spread lobes horizontally across roughly 1.4*size, keeping overall
    # extent within +/- size of the center in both axes.
    span = size * 0.75
    for i in range(n_lobes):
        t = i / max(1, n_lobes - 1)  # 0..1
        lx = cx - span + t * (2 * span)
        # Middle lobes sit higher, edge lobes sit lower => cloud silhouette
        bump = math.sin(t * math.pi)   # 0 at edges, 1 in middle
        ly = cy - bump * size * 0.15 + random.uniform(-size * 0.04, size * 0.04)
        lr = size * random.uniform(0.42, 0.55)
        # Middle lobes slightly bigger
        lr *= 0.85 + bump * 0.25
        lobes.append((lx, ly, lr))

    # Sample points along the outer perimeter of the union of lobes.
    n_samples = 96
    pts: list[tuple[float, float]] = []
    for i in range(n_samples):
        a = 2.0 * math.pi * i / n_samples
        dir_x, dir_y = math.cos(a), math.sin(a)
        # For each angle, find the farthest point from cx,cy that lies
        # within any of the lobes.
        best = 0.0
        for lx, ly, lr in lobes:
            # Intersect ray from (cx,cy) in direction (dir_x,dir_y) with
            # lobe circle. Parametrize as (cx,cy) + t*(dir_x,dir_y).
            fx = cx - lx
            fy = cy - ly
            b = 2.0 * (dir_x * fx + dir_y * fy)
            c = fx * fx + fy * fy - lr * lr
            disc = b * b - 4 * c
            if disc < 0:
                continue
            sq = math.sqrt(disc)
            t1 = (-b + sq) * 0.5
            if t1 > best:
                best = t1
        if best <= 0:
            best = size * 0.3
        pts.append((cx + dir_x * best, cy + dir_y * best))
    # Flatten bottom slightly so the cloud looks right-side-up
    flat_y = cy + size * 0.35
    pts = [(x, min(y, flat_y)) for x, y in pts]
    return pts


def _teardrop_vertices(cx: float, cy: float,
                       size: float) -> list[tuple[float, float]]:
    """Round at bottom, pointed at top (default orientation)."""
    n = 64
    pts: list[tuple[float, float]] = []
    # Parametrize by angle around the center.  The radius decreases as
    # we approach the top (negative-y direction) so the shape tapers.
    for i in range(n):
        a = 2.0 * math.pi * i / n
        # angle 0 = right, pi/2 = down, -pi/2 = up (our top)
        sin_a = math.sin(a)   # +1 at top (up), -1 at bottom
        # factor is 1 at bottom, 0 near the top point
        taper = (1.0 - sin_a) * 0.5  # in [0,1]
        # smooth ease + small power -> pointier tip
        taper = taper ** 0.85
        r_x = size * 0.85
        r_y = size
        x = cx + r_x * math.cos(a) * taper
        y = cy - r_y * sin_a
        pts.append((x, y))
    return pts


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
    """Moon-shape crescent from two overlapping circles.

    Builds a proper polygon by walking the outer arc on the far side of the
    inner circle, then the inner arc on the near side back to the start.
    """
    R = size
    r = size * random.uniform(0.85, 1.0)
    dx = size * random.uniform(0.45, 0.70)
    dy = size * random.uniform(-0.12, 0.12)
    d = math.hypot(dx, dy)

    # Circles must properly overlap (two intersections) — fall back otherwise.
    if d >= R + r or d <= abs(R - r) or d < 1e-3:
        pts = [
            (cx + R * math.cos(2 * math.pi * i / 80),
             cy + R * math.sin(2 * math.pi * i / 80))
            for i in range(80)
        ]
        pts = _transform(pts, attrs, cx, cy)
        _draw_poly(draw, pts, attrs)
        return

    icx = cx + dx
    icy = cy + dy
    angle_to_inner = math.atan2(dy, dx)

    # Angles (at outer center) to the two intersection points.
    cos_phi = (R * R + d * d - r * r) / (2 * R * d)
    cos_phi = max(-1.0, min(1.0, cos_phi))
    phi = math.acos(cos_phi)

    # Outer arc goes the long way AROUND, away from the inner circle.
    steps = 60
    outer_arc: list[tuple[float, float]] = []
    a_start = angle_to_inner + phi
    a_end = angle_to_inner - phi + 2 * math.pi
    for i in range(steps + 1):
        a = a_start + (a_end - a_start) * i / steps
        outer_arc.append((cx + R * math.cos(a), cy + R * math.sin(a)))

    # Inner arc: from inner-circle center, the arc facing the outer center.
    cos_psi = (r * r + d * d - R * R) / (2 * r * d)
    cos_psi = max(-1.0, min(1.0, cos_psi))
    psi = math.acos(cos_psi)
    angle_to_outer = angle_to_inner + math.pi

    inner_arc: list[tuple[float, float]] = []
    i_start = angle_to_outer + psi
    i_end = angle_to_outer - psi
    for i in range(steps + 1):
        a = i_start + (i_end - i_start) * i / steps
        inner_arc.append((icx + r * math.cos(a), icy + r * math.sin(a)))

    crescent = outer_arc + inner_arc
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
# Multi-color / holed shapes (direct draw)
# ---------------------------------------------------------------------------

def _secondary_color(primary: tuple[int, int, int],
                     hue_shift: float = 0.4,
                     sat: float = 0.75,
                     val: float = 0.8) -> tuple[int, int, int]:
    """Generate a contrasting color by shifting the primary color's hue."""
    r, g, b = primary[0] / 255.0, primary[1] / 255.0, primary[2] / 255.0
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    new_h = (h + hue_shift) % 1.0
    nr, ng, nb = colorsys.hsv_to_rgb(new_h, sat, val)
    return (int(nr * 255), int(ng * 255), int(nb * 255))


def _hsv(hue: float, sat: float = 0.85,
         val: float = 0.85) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def _circle_points(cx: float, cy: float, r: float,
                   n: int = 48) -> list[tuple[float, float]]:
    return [
        (cx + r * math.cos(2.0 * math.pi * i / n),
         cy + r * math.sin(2.0 * math.pi * i / n))
        for i in range(n)
    ]


def _draw_donut(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                size: float, attrs: dict):
    """A ring: filled outer circle with a concentric hole."""
    outer_r = size
    inner_r = size * random.uniform(0.38, 0.55)
    outer = _circle_points(cx, cy, outer_r, n=64)
    outer = _transform(outer, attrs, cx, cy)
    _draw_poly(draw, outer, attrs)
    # Punch hole using bg color
    bg = attrs["bg_color"]
    hole = _circle_points(cx, cy, inner_r, n=48)
    coords = [(round(x), round(y)) for x, y in hole]
    draw.polygon(coords, fill=bg)


def _draw_picture_frame(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                        size: float, attrs: dict):
    """A square outline: filled square with a concentric square hole."""
    outer = _square_vertices(cx, cy, size)
    outer = _transform(outer, attrs, cx, cy)
    _draw_poly(draw, outer, attrs)
    # Hole
    inner_size = size * random.uniform(0.55, 0.72)
    inner = _square_vertices(cx, cy, inner_size)
    inner = _rotate(inner, attrs["rotation_deg"], cx, cy)
    bg = attrs["bg_color"]
    coords = [(round(x), round(y)) for x, y in inner]
    draw.polygon(coords, fill=bg)


def _draw_key(draw: ImageDraw.ImageDraw, cx: float, cy: float,
              size: float, attrs: dict):
    """Circular handle on the left + rectangular shaft on the right with notches."""
    fill = attrs["fill_color"]
    bg = attrs["bg_color"]
    rot = attrs["rotation_deg"]

    # Geometry in local (pre-rotation) space
    handle_cx = cx - size * 0.55
    handle_cy = cy
    handle_r = size * 0.55
    shaft_left = cx - size * 0.05
    shaft_right = cx + size * 0.95
    shaft_top = cy - size * 0.16
    shaft_bottom = cy + size * 0.16

    # Handle (outer circle)
    handle_pts = _circle_points(handle_cx, handle_cy, handle_r, n=48)
    handle_pts = _rotate(handle_pts, rot, cx, cy)
    _draw_poly(draw, handle_pts, attrs)

    # Shaft rectangle + notches on the far (right) end
    notch_h = size * 0.22
    notch_w = size * 0.18
    # Build a single polygon that forms the shaft with 1-2 teeth on the bottom
    n_notches = random.randint(1, 2)
    shaft_pts: list[tuple[float, float]] = []
    # Start top-left, go clockwise
    shaft_pts.append((shaft_left, shaft_top))
    shaft_pts.append((shaft_right, shaft_top))
    shaft_pts.append((shaft_right, shaft_bottom))
    # Walk back along the bottom, adding notches (teeth extending downward)
    x_cursor = shaft_right
    step = size * 0.28
    for i in range(n_notches):
        notch_right = x_cursor - step * i - size * 0.05
        notch_left = notch_right - notch_w
        if notch_left <= shaft_left + size * 0.1:
            break
        shaft_pts.append((notch_right, shaft_bottom))
        shaft_pts.append((notch_right, shaft_bottom + notch_h))
        shaft_pts.append((notch_left, shaft_bottom + notch_h))
        shaft_pts.append((notch_left, shaft_bottom))
    shaft_pts.append((shaft_left, shaft_bottom))
    shaft_pts = _rotate(shaft_pts, rot, cx, cy)
    _draw_poly(draw, shaft_pts, attrs)

    # Hole in the handle (bg color)
    hole_r = handle_r * random.uniform(0.35, 0.5)
    hole_pts = _circle_points(handle_cx, handle_cy, hole_r, n=36)
    hole_pts = _rotate(hole_pts, rot, cx, cy)
    coords = [(round(x), round(y)) for x, y in hole_pts]
    draw.polygon(coords, fill=bg)


def _draw_gear(draw: ImageDraw.ImageDraw, cx: float, cy: float,
               size: float, attrs: dict):
    """Circle with rectangular teeth around the perimeter and a center hole."""
    n_teeth = 8
    body_r = size * 0.78
    tooth_outer = size
    # Build the gear outline by alternating between body_r (between teeth)
    # and tooth_outer (tooth tip), with flat segments for the sides of
    # each tooth.
    pts: list[tuple[float, float]] = []
    # 2 points per tooth sector: (base_start, tip_start, tip_end, base_end)
    # We use n_teeth*4 points around the circle.
    n = n_teeth * 4
    for i in range(n):
        frac = i / n
        a = 2.0 * math.pi * frac - math.pi / 2
        phase = i % 4  # 0 base_start, 1 tip_start, 2 tip_end, 3 base_end
        if phase == 0 or phase == 3:
            r = body_r
        else:
            r = tooth_outer
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts = _rotate(pts, attrs["rotation_deg"], cx, cy)
    _draw_poly(draw, pts, attrs)

    # Center hole (bg)
    hole_r = size * random.uniform(0.22, 0.32)
    hole = _circle_points(cx, cy, hole_r, n=36)
    hole = _rotate(hole, attrs["rotation_deg"], cx, cy)
    bg = attrs["bg_color"]
    coords = [(round(x), round(y)) for x, y in hole]
    draw.polygon(coords, fill=bg)


def _draw_mushroom(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                   size: float, attrs: dict):
    """Dome cap on top of a rectangular stem (two colors)."""
    cap_color = attrs["fill_color"]
    stem_color = _secondary_color(cap_color, hue_shift=0.5, sat=0.35, val=0.9)
    rot = attrs["rotation_deg"]

    # Cap: top half of an ellipse wider than it is tall
    cap_rx = size
    cap_ry = size * 0.7
    cap_cy = cy - size * 0.15
    cap_pts: list[tuple[float, float]] = []
    n = 48
    for i in range(n + 1):
        a = math.pi * i / n  # 0..pi sweeps the top half
        cap_pts.append((cx + cap_rx * math.cos(a),
                        cap_cy - cap_ry * math.sin(a)))
    # Flat bottom of the cap
    cap_pts.append((cx - cap_rx, cap_cy))
    cap_pts = _rotate(cap_pts, rot, cx, cy)
    coords = [(round(x), round(y)) for x, y in cap_pts]
    draw.polygon(coords, fill=cap_color)

    # Stem: rectangle attached under the cap
    stem_w = size * 0.45
    stem_top = cap_cy
    stem_bottom = cy + size * 0.85
    stem_pts = [
        (cx - stem_w, stem_top),
        (cx + stem_w, stem_top),
        (cx + stem_w, stem_bottom),
        (cx - stem_w, stem_bottom),
    ]
    stem_pts = _rotate(stem_pts, rot, cx, cy)
    coords = [(round(x), round(y)) for x, y in stem_pts]
    draw.polygon(coords, fill=stem_color)


def _draw_tree(draw: ImageDraw.ImageDraw, cx: float, cy: float,
               size: float, attrs: dict):
    """Triangular green canopy on top of a brown rectangular trunk."""
    rot = attrs["rotation_deg"]
    # Green canopy (hue around 0.3), some variation in value/saturation
    canopy_color = _hsv(random.uniform(0.28, 0.38),
                        sat=random.uniform(0.65, 0.9),
                        val=random.uniform(0.55, 0.8))
    # Brown trunk (hue around 0.08)
    trunk_color = _hsv(random.uniform(0.06, 0.1),
                       sat=random.uniform(0.55, 0.8),
                       val=random.uniform(0.4, 0.6))

    # Canopy triangle occupies the top ~75% of the frame
    canopy_top = cy - size
    canopy_bottom = cy + size * 0.4
    canopy_half = size * 0.85
    canopy_pts = [
        (cx, canopy_top),
        (cx + canopy_half, canopy_bottom),
        (cx - canopy_half, canopy_bottom),
    ]
    canopy_pts = _rotate(canopy_pts, rot, cx, cy)
    coords = [(round(x), round(y)) for x, y in canopy_pts]
    draw.polygon(coords, fill=canopy_color)

    # Trunk rectangle below the canopy
    trunk_w = size * 0.25
    trunk_top = canopy_bottom
    trunk_bottom = cy + size
    trunk_pts = [
        (cx - trunk_w, trunk_top),
        (cx + trunk_w, trunk_top),
        (cx + trunk_w, trunk_bottom),
        (cx - trunk_w, trunk_bottom),
    ]
    trunk_pts = _rotate(trunk_pts, rot, cx, cy)
    coords = [(round(x), round(y)) for x, y in trunk_pts]
    draw.polygon(coords, fill=trunk_color)


def _draw_flower(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                 size: float, attrs: dict):
    """Multi-colored petals radiating from a central disc."""
    rot_base = attrs["rotation_deg"]
    n_petals = random.randint(5, 6)
    petal_color = attrs["fill_color"]
    # Center color is a contrasting hue (e.g., yellow if petals are pink)
    center_color = _secondary_color(petal_color, hue_shift=random.uniform(0.3, 0.6),
                                    sat=0.85, val=0.95)

    petal_r = size * 0.4  # petal half-length (each petal ~2*petal_r long)
    petal_dist = size * 0.55  # distance from center to petal center
    petal_w = size * 0.32

    for i in range(n_petals):
        a = 2.0 * math.pi * i / n_petals + math.radians(rot_base)
        pcx = cx + petal_dist * math.cos(a)
        pcy = cy + petal_dist * math.sin(a)
        # Build an ellipse oriented along the radial direction
        n = 24
        pts: list[tuple[float, float]] = []
        for j in range(n):
            t = 2.0 * math.pi * j / n
            # Ellipse with long axis along angle a
            ex = petal_r * math.cos(t)
            ey = petal_w * math.sin(t)
            cos_a, sin_a = math.cos(a), math.sin(a)
            px = pcx + ex * cos_a - ey * sin_a
            py = pcy + ex * sin_a + ey * cos_a
            pts.append((px, py))
        coords = [(round(x), round(y)) for x, y in pts]
        draw.polygon(coords, fill=petal_color)

    # Central disc
    center_r = size * 0.32
    center_pts = _circle_points(cx, cy, center_r, n=48)
    coords = [(round(x), round(y)) for x, y in center_pts]
    draw.polygon(coords, fill=center_color)


def _draw_candy_cane(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                     size: float, attrs: dict):
    """Red-and-white J-shaped cane with diagonal stripes."""
    rot = attrs["rotation_deg"]
    # Red-ish primary (could vary a bit)
    red = _hsv(random.uniform(0.97, 1.02) % 1.0, sat=0.85, val=0.9)
    white = (245, 245, 245)

    # Build the cane as a thick stroked path:
    # vertical shaft down the right, rounded hook at the top curving left.
    thickness = size * 0.34
    shaft_x = cx + size * 0.35
    shaft_top = cy - size * 0.25
    shaft_bottom = cy + size
    hook_center_x = shaft_x - size * 0.6
    hook_center_y = shaft_top
    hook_r = size * 0.6

    # Sample centerline: start at bottom of shaft, go up to top of shaft,
    # then arc left around the hook center.
    centerline: list[tuple[float, float]] = []
    n_shaft = 20
    for i in range(n_shaft + 1):
        t = i / n_shaft
        centerline.append((shaft_x, shaft_bottom + (shaft_top - shaft_bottom) * t))
    n_arc = 24
    for i in range(1, n_arc + 1):
        a = -math.pi / 2 * (1 - i / n_arc) + math.pi * (i / n_arc)
        # start angle: 0 (pointing right from hook_center -> shaft_top)
        # end angle: pi (pointing left)
        # Parametrize i=0 -> 0, i=n_arc -> pi
        angle = math.pi * i / n_arc
        centerline.append((hook_center_x + hook_r * math.cos(angle),
                           hook_center_y - hook_r * math.sin(angle)))

    # Build a thick polygon by offsetting the centerline left/right
    left_side: list[tuple[float, float]] = []
    right_side: list[tuple[float, float]] = []
    for i in range(len(centerline)):
        x, y = centerline[i]
        # Compute tangent
        if i == 0:
            nx_, ny_ = centerline[1]
            tx, ty = nx_ - x, ny_ - y
        elif i == len(centerline) - 1:
            px, py = centerline[i - 1]
            tx, ty = x - px, y - py
        else:
            nx_, ny_ = centerline[i + 1]
            px, py = centerline[i - 1]
            tx, ty = nx_ - px, ny_ - py
        length = math.hypot(tx, ty) or 1.0
        # Normal is perpendicular
        nx = -ty / length
        ny = tx / length
        left_side.append((x + nx * thickness * 0.5,
                          y + ny * thickness * 0.5))
        right_side.append((x - nx * thickness * 0.5,
                           y - ny * thickness * 0.5))

    outline_pts = left_side + list(reversed(right_side))
    outline_pts = _rotate(outline_pts, rot, cx, cy)
    coords = [(round(x), round(y)) for x, y in outline_pts]
    # Fill base with white, then overlay red stripes clipped to this polygon.
    draw.polygon(coords, fill=white)

    # Draw diagonal red stripes across the full canvas, clipped by the
    # cane polygon via a mask image.
    stripe_img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), white)
    stripe_draw = ImageDraw.Draw(stripe_img)
    stripe_spacing = max(6, int(size * 0.35))
    stripe_width = max(3, int(size * 0.18))
    # Diagonal stripes running 45 degrees
    for offset in range(-CANVAS_SIZE, CANVAS_SIZE * 2, stripe_spacing):
        stripe_draw.line(
            [(offset, 0), (offset + CANVAS_SIZE, CANVAS_SIZE)],
            fill=red, width=stripe_width,
        )

    mask = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(coords, fill=255)
    # Paste stripes onto the main image wherever the mask is set.
    base = draw._image  # type: ignore[attr-defined]
    base.paste(stripe_img, (0, 0), mask)


def _draw_rainbow(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                  size: float, attrs: dict):
    """An upward-arched set of nested colored arcs."""
    rot = attrs["rotation_deg"]
    colors = [
        _hsv(0.0, sat=0.85, val=0.9),   # red
        _hsv(0.08, sat=0.85, val=0.95),  # orange
        _hsv(0.15, sat=0.85, val=0.95),  # yellow
        _hsv(0.33, sat=0.8, val=0.75),   # green
        _hsv(0.6, sat=0.85, val=0.85),   # blue
    ]
    # Base of arch below center so the arch rises
    base_y = cy + size * 0.55
    outer_r = size
    # Step per band
    n_bands = len(colors)
    band_w = outer_r / (n_bands + 1) * 0.9

    for i, color in enumerate(colors):
        r_outer = outer_r - i * band_w
        r_inner = r_outer - band_w
        if r_inner <= 0:
            continue
        # Build annular sector: outer half-circle + inner half-circle (reversed)
        pts: list[tuple[float, float]] = []
        n = 40
        for j in range(n + 1):
            a = math.pi * j / n  # 0..pi sweeps left to right along top
            pts.append((cx + r_outer * math.cos(a),
                        base_y - r_outer * math.sin(a)))
        for j in range(n, -1, -1):
            a = math.pi * j / n
            pts.append((cx + r_inner * math.cos(a),
                        base_y - r_inner * math.sin(a)))
        pts = _rotate(pts, rot, cx, cy)
        coords = [(round(x), round(y)) for x, y in pts]
        draw.polygon(coords, fill=color)


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
    "cloud": _cloud_vertices,
    "teardrop": _teardrop_vertices,
}

_DIRECT_DRAW_SHAPES: dict[str, object] = {
    "heart": _draw_heart,
    "crescent": _draw_crescent,
    "circle": _draw_circle,
    "oval": _draw_oval,
    "semicircle": _draw_semicircle,
    "donut": _draw_donut,
    "picture_frame": _draw_picture_frame,
    "key": _draw_key,
    "gear": _draw_gear,
    "mushroom": _draw_mushroom,
    "tree": _draw_tree,
    "flower": _draw_flower,
    "candy_cane": _draw_candy_cane,
    "rainbow": _draw_rainbow,
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
