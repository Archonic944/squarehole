"""Procedural meta-training episode generator.

Generates a massive diversity of visual classification tasks so the
MAML backbone learns general-purpose visual features.  Each episode
picks a random visual concept (binary split) and generates support +
query images for both sides.

Building blocks (visual primitives the backbone must learn to detect):
  - Edge curvature (straight vs curved)
  - Vertex count and sharpness
  - Symmetry (bilateral, rotational, none)
  - Convexity / concavity
  - Open vs closed contours
  - Containment (nested shapes)
  - Fill (solid, outline, striped, dotted)
  - Aspect ratio (tall vs wide)
  - Complexity (simple vs compound)
  - Regularity (even vs irregular vertices)
  - Color properties (warm/cool, bright/dark, saturated/muted)
  - Size (large vs small on canvas)
  - Count (single vs multiple shapes)
"""

import math
import random
import colorsys

import numpy as np
import torch
from PIL import Image, ImageDraw

IMG_SIZE = 84


# ---------------------------------------------------------------------------
# Low-level drawing primitives
# ---------------------------------------------------------------------------

def _random_color(hue_range=None, sat_range=(0.4, 1.0), val_range=(0.4, 0.9)):
    """Return an RGB tuple from HSV ranges."""
    h = random.uniform(*hue_range) if hue_range else random.random()
    s = random.uniform(*sat_range)
    v = random.uniform(*val_range)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _random_bg():
    v = random.randint(230, 255)
    return (v, v, v)


def _rotate_points(points, angle_deg, cx, cy):
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    return [
        (cx + (x - cx) * cos_a - (y - cy) * sin_a,
         cy + (x - cx) * sin_a + (y - cy) * cos_a)
        for x, y in points
    ]


def _jitter_points(points, amount):
    return [(x + random.uniform(-amount, amount),
             y + random.uniform(-amount, amount)) for x, y in points]


# ---------------------------------------------------------------------------
# Shape generators — return (points_list, is_closed) or draw directly
# ---------------------------------------------------------------------------

def gen_regular_polygon(n_sides, cx, cy, r, rotation=0):
    """Regular n-gon."""
    pts = []
    for i in range(n_sides):
        angle = math.radians(360 * i / n_sides - 90 + rotation)
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def gen_star(n_points, cx, cy, r_outer, r_inner=None, rotation=0):
    """Star with n points."""
    if r_inner is None:
        r_inner = r_outer * random.uniform(0.3, 0.5)
    pts = []
    for i in range(n_points * 2):
        angle = math.radians(360 * i / (n_points * 2) - 90 + rotation)
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return pts


def gen_irregular_polygon(n_sides, cx, cy, r, jitter_frac=0.3):
    """Irregular polygon with random vertex perturbation."""
    pts = gen_regular_polygon(n_sides, cx, cy, r, random.uniform(0, 360))
    return _jitter_points(pts, r * jitter_frac)


def gen_spiral(cx, cy, r, turns=2.5, n_points=60):
    """Spiral curve (open contour)."""
    pts = []
    for i in range(n_points):
        t = i / n_points
        angle = t * turns * 2 * math.pi
        radius = r * 0.1 + r * 0.9 * t
        pts.append((cx + radius * math.cos(angle),
                     cy + radius * math.sin(angle)))
    return pts


def gen_wave(cx, cy, w, h, periods=2, n_points=40):
    """Sine wave (open contour)."""
    pts = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = cx - w / 2 + t * w
        y = cy + h / 2 * math.sin(t * periods * 2 * math.pi)
        pts.append((x, y))
    return pts


def gen_blob(cx, cy, r, n_bumps=5, bump_size=0.3):
    """Organic blob shape with random bumps."""
    pts = []
    n = 40
    for i in range(n):
        angle = 2 * math.pi * i / n
        noise = 1.0 + bump_size * math.sin(n_bumps * angle + random.uniform(-0.5, 0.5))
        rad = r * noise
        pts.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
    return pts


def gen_arrow(cx, cy, length, width, rotation=0):
    """Arrow shape pointing right, then rotated."""
    hw = width / 2
    hl = length / 2
    head_len = length * 0.35
    pts = [
        (cx - hl, cy - hw * 0.4),
        (cx + hl - head_len, cy - hw * 0.4),
        (cx + hl - head_len, cy - hw),
        (cx + hl, cy),
        (cx + hl - head_len, cy + hw),
        (cx + hl - head_len, cy + hw * 0.4),
        (cx - hl, cy + hw * 0.4),
    ]
    return _rotate_points(pts, rotation, cx, cy)


def gen_cross(cx, cy, r, thickness_frac=0.3):
    """Plus/cross shape."""
    t = r * thickness_frac
    return [
        (cx - t, cy - r), (cx + t, cy - r),
        (cx + t, cy - t), (cx + r, cy - t),
        (cx + r, cy + t), (cx + t, cy + t),
        (cx + t, cy + r), (cx - t, cy + r),
        (cx - t, cy + t), (cx - r, cy + t),
        (cx - r, cy - t), (cx - t, cy - t),
    ]


def gen_leaf(cx, cy, r, rotation=0):
    """Simple leaf shape using two arcs."""
    pts = []
    n = 30
    for i in range(n):
        t = i / (n - 1)
        angle = -math.pi / 2 + t * math.pi
        # Asymmetric radius for leaf shape
        rad = r * (0.5 + 0.5 * math.sin(t * math.pi))
        pts.append((cx + rad * math.cos(angle) * 0.5,
                     cy + r * (t - 0.5)))
    # Mirror
    for i in range(n - 1, -1, -1):
        t = i / (n - 1)
        angle = -math.pi / 2 + t * math.pi
        rad = r * (0.5 + 0.5 * math.sin(t * math.pi))
        pts.append((cx - rad * math.cos(angle) * 0.5,
                     cy + r * (t - 0.5)))
    return _rotate_points(pts, rotation, cx, cy)


def gen_crescent(cx, cy, r, offset=0.35):
    """Crescent moon shape."""
    pts = []
    n = 40
    for i in range(n):
        t = i / (n - 1)
        angle = -math.pi / 2 + t * math.pi * 2
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    # Inner circle offset
    for i in range(n - 1, -1, -1):
        t = i / (n - 1)
        angle = -math.pi / 2 + t * math.pi * 2
        inner_r = r * 0.85
        pts.append((cx + r * offset + inner_r * math.cos(angle),
                     cy + inner_r * math.sin(angle)))
    return pts


# ---------------------------------------------------------------------------
# Compound shape generators
# ---------------------------------------------------------------------------

def gen_nested(draw, cx, cy, r, color, bg):
    """Shape inside a shape (containment)."""
    outer_sides = random.choice([0, 3, 4, 5, 6])  # 0 = circle
    inner_sides = random.choice([0, 3, 4, 5, 6])
    if outer_sides == 0:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
    else:
        pts = gen_regular_polygon(outer_sides, cx, cy, r, random.uniform(0, 360))
        draw.polygon(pts, outline=color, width=2)
    inner_r = r * random.uniform(0.3, 0.6)
    if inner_sides == 0:
        draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
                      fill=color)
    else:
        pts2 = gen_regular_polygon(inner_sides, cx, cy, inner_r, random.uniform(0, 360))
        draw.polygon(pts2, fill=color)


def gen_multi_shapes(draw, cx, cy, r, color, count=None):
    """Multiple scattered shapes."""
    if count is None:
        count = random.randint(2, 5)
    for _ in range(count):
        sx = cx + random.uniform(-r * 0.6, r * 0.6)
        sy = cy + random.uniform(-r * 0.6, r * 0.6)
        sr = r * random.uniform(0.15, 0.35)
        sides = random.choice([0, 3, 4, 5])
        if sides == 0:
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=color)
        else:
            pts = gen_regular_polygon(sides, sx, sy, sr, random.uniform(0, 360))
            draw.polygon(pts, fill=color)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _draw_shape(draw, pts, color, fill_mode="solid", line_width=2):
    """Draw a polygon/polyline with various fill modes."""
    if fill_mode == "solid":
        draw.polygon(pts, fill=color)
    elif fill_mode == "outline":
        draw.polygon(pts, outline=color, width=line_width)
    elif fill_mode == "thick_outline":
        draw.polygon(pts, outline=color, width=max(3, line_width))
    elif fill_mode == "dotted":
        draw.polygon(pts, fill=color)
        # Add dots
        for i in range(0, len(pts), max(1, len(pts) // 5)):
            x, y = pts[i]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255))


def render_image(shape_fn, attrs):
    """Render a single image and return as (3, 84, 84) tensor."""
    bg = attrs.get("bg_color", _random_bg())
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)

    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    cx += attrs.get("offset_x", 0)
    cy += attrs.get("offset_y", 0)

    color = attrs.get("color", _random_color())
    r = attrs.get("size", random.randint(15, 35))
    fill = attrs.get("fill_mode", "solid")
    rotation = attrs.get("rotation", random.uniform(0, 360))

    shape_fn(draw, cx, cy, r, color, rotation, fill, attrs)

    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Shape function wrappers (unified interface for render_image)
# ---------------------------------------------------------------------------

def shape_polygon(draw, cx, cy, r, color, rotation, fill, attrs):
    n = attrs.get("n_sides", random.randint(3, 10))
    jitter = attrs.get("jitter", random.uniform(0, 0.2))
    pts = gen_irregular_polygon(n, cx, cy, r, jitter)
    pts = _rotate_points(pts, rotation, cx, cy)
    _draw_shape(draw, pts, color, fill)


def shape_star(draw, cx, cy, r, color, rotation, fill, attrs):
    n = attrs.get("n_points", random.randint(3, 8))
    pts = gen_star(n, cx, cy, r, rotation=rotation)
    _draw_shape(draw, pts, color, fill)


def shape_circle(draw, cx, cy, r, color, rotation, fill, attrs):
    if fill == "solid":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        w = 2 if fill == "outline" else 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=w)


def shape_ellipse(draw, cx, cy, r, color, rotation, fill, attrs):
    aspect = attrs.get("aspect", random.uniform(0.4, 0.8))
    rx, ry = r, r * aspect
    if fill == "solid":
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    else:
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=color, width=2)


def shape_spiral(draw, cx, cy, r, color, rotation, fill, attrs):
    turns = attrs.get("turns", random.uniform(1.5, 3.5))
    pts = gen_spiral(cx, cy, r, turns=turns)
    pts = _rotate_points(pts, rotation, cx, cy)
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=color, width=2)


def shape_wave(draw, cx, cy, r, color, rotation, fill, attrs):
    periods = attrs.get("periods", random.uniform(1.5, 4))
    pts = gen_wave(cx, cy, r * 1.5, r * 0.6, periods=periods)
    pts = _rotate_points(pts, rotation, cx, cy)
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=color, width=2)


def shape_blob(draw, cx, cy, r, color, rotation, fill, attrs):
    n_bumps = attrs.get("n_bumps", random.randint(3, 8))
    pts = gen_blob(cx, cy, r, n_bumps=n_bumps)
    _draw_shape(draw, pts, color, fill)


def shape_arrow(draw, cx, cy, r, color, rotation, fill, attrs):
    pts = gen_arrow(cx, cy, r * 1.8, r * 0.8, rotation=rotation)
    _draw_shape(draw, pts, color, fill)


def shape_cross(draw, cx, cy, r, color, rotation, fill, attrs):
    thickness = attrs.get("thickness_frac", random.uniform(0.2, 0.45))
    pts = gen_cross(cx, cy, r, thickness)
    pts = _rotate_points(pts, rotation, cx, cy)
    _draw_shape(draw, pts, color, fill)


def shape_leaf(draw, cx, cy, r, color, rotation, fill, attrs):
    pts = gen_leaf(cx, cy, r, rotation=rotation)
    _draw_shape(draw, pts, color, fill)


def shape_crescent(draw, cx, cy, r, color, rotation, fill, attrs):
    pts = gen_crescent(cx, cy, r)
    pts = _rotate_points(pts, rotation, cx, cy)
    _draw_shape(draw, pts, color, fill)


def shape_nested(draw, cx, cy, r, color, rotation, fill, attrs):
    gen_nested(draw, cx, cy, r, color, _random_bg())


def shape_multi(draw, cx, cy, r, color, rotation, fill, attrs):
    count = attrs.get("count", random.randint(2, 5))
    gen_multi_shapes(draw, cx, cy, r, color, count=count)


ALL_SHAPE_FNS = [
    shape_polygon, shape_star, shape_circle, shape_ellipse,
    shape_spiral, shape_wave, shape_blob, shape_arrow,
    shape_cross, shape_leaf, shape_crescent, shape_nested, shape_multi,
]


# ---------------------------------------------------------------------------
# Meta-training episode concepts (binary splits)
# ---------------------------------------------------------------------------

def _sample_random_attrs(**overrides):
    attrs = {
        "color": _random_color(),
        "bg_color": _random_bg(),
        "size": random.randint(14, 36),
        "rotation": random.uniform(0, 360),
        "offset_x": random.randint(-6, 6),
        "offset_y": random.randint(-6, 6),
        "fill_mode": random.choice(["solid", "solid", "solid", "outline", "thick_outline"]),
    }
    attrs.update(overrides)
    return attrs


# Each concept is (name, fn_class_0, fn_class_1) where each fn takes
# no args and returns (shape_fn, attrs_override).

def concept_curved_vs_straight():
    """Curved edges vs straight edges."""
    curved = [shape_circle, shape_ellipse, shape_blob, shape_spiral, shape_wave, shape_crescent]
    straight = [shape_polygon, shape_star, shape_arrow, shape_cross]

    def class_0():
        fn = random.choice(curved)
        return fn, _sample_random_attrs()

    def class_1():
        fn = random.choice(straight)
        return fn, _sample_random_attrs(n_sides=random.randint(3, 8))

    return class_0, class_1


def concept_pointy_vs_smooth():
    """Pointy shapes (stars, arrows) vs smooth shapes."""
    def class_0():  # pointy
        fn = random.choice([shape_star, shape_arrow, shape_polygon])
        return fn, _sample_random_attrs(n_sides=random.randint(3, 5), jitter=0.0)

    def class_1():  # smooth
        fn = random.choice([shape_circle, shape_ellipse, shape_blob])
        return fn, _sample_random_attrs()

    return class_0, class_1


def concept_simple_vs_complex():
    """Simple shapes (few sides, single) vs complex (many sides, compound)."""
    def class_0():  # simple
        fn = random.choice([shape_circle, shape_polygon])
        return fn, _sample_random_attrs(n_sides=random.choice([3, 4]), jitter=0.0)

    def class_1():  # complex
        fn = random.choice([shape_star, shape_nested, shape_multi, shape_spiral])
        return fn, _sample_random_attrs(n_points=random.randint(5, 8))

    return class_0, class_1


def concept_filled_vs_outline():
    """Solid filled vs outline only."""
    def class_0():
        fn = random.choice(ALL_SHAPE_FNS[:8])
        return fn, _sample_random_attrs(fill_mode="solid")

    def class_1():
        fn = random.choice(ALL_SHAPE_FNS[:8])
        return fn, _sample_random_attrs(fill_mode="outline")

    return class_0, class_1


def concept_big_vs_small():
    """Large shapes vs small shapes."""
    def class_0():
        fn = random.choice(ALL_SHAPE_FNS[:8])
        return fn, _sample_random_attrs(size=random.randint(28, 38))

    def class_1():
        fn = random.choice(ALL_SHAPE_FNS[:8])
        return fn, _sample_random_attrs(size=random.randint(10, 18))

    return class_0, class_1


def concept_warm_vs_cool():
    """Warm colors (red/orange/yellow) vs cool (blue/green/purple)."""
    def class_0():
        fn = random.choice(ALL_SHAPE_FNS[:8])
        color = _random_color(hue_range=(0.0, 0.15))  # red-yellow
        return fn, _sample_random_attrs(color=color)

    def class_1():
        fn = random.choice(ALL_SHAPE_FNS[:8])
        color = _random_color(hue_range=(0.5, 0.75))  # blue-purple
        return fn, _sample_random_attrs(color=color)

    return class_0, class_1


def concept_tall_vs_wide():
    """Tall aspect ratio vs wide."""
    def class_0():  # tall
        fn = shape_ellipse
        return fn, _sample_random_attrs(aspect=random.uniform(0.3, 0.5))

    def class_1():  # wide
        fn = shape_ellipse
        return fn, _sample_random_attrs(aspect=random.uniform(1.5, 2.5))

    return class_0, class_1


def concept_single_vs_multiple():
    """One shape vs multiple shapes."""
    def class_0():
        fn = random.choice([shape_circle, shape_polygon, shape_star, shape_blob])
        return fn, _sample_random_attrs()

    def class_1():
        return shape_multi, _sample_random_attrs(count=random.randint(3, 6))

    return class_0, class_1


def concept_symmetric_vs_irregular():
    """Regular/symmetric shapes vs irregular ones."""
    def class_0():
        fn = random.choice([shape_circle, shape_polygon])
        return fn, _sample_random_attrs(n_sides=random.choice([3, 4, 5, 6]), jitter=0.0)

    def class_1():
        fn = shape_polygon
        return fn, _sample_random_attrs(n_sides=random.randint(4, 8), jitter=random.uniform(0.25, 0.45))

    return class_0, class_1


def concept_open_vs_closed():
    """Open contours (spiral, wave) vs closed shapes."""
    def class_0():  # open
        fn = random.choice([shape_spiral, shape_wave])
        return fn, _sample_random_attrs()

    def class_1():  # closed
        fn = random.choice([shape_circle, shape_polygon, shape_star, shape_blob])
        return fn, _sample_random_attrs()

    return class_0, class_1


def concept_contained_vs_flat():
    """Nested/contained shapes vs flat single shapes."""
    def class_0():
        return shape_nested, _sample_random_attrs()

    def class_1():
        fn = random.choice([shape_circle, shape_polygon, shape_star])
        return fn, _sample_random_attrs()

    return class_0, class_1


def concept_few_sides_vs_many():
    """Polygons with few sides (3-4) vs many (6-10)."""
    def class_0():
        return shape_polygon, _sample_random_attrs(n_sides=random.choice([3, 4]), jitter=0.05)

    def class_1():
        return shape_polygon, _sample_random_attrs(n_sides=random.randint(7, 12), jitter=0.05)

    return class_0, class_1


def concept_random_shape_pair():
    """Random pair of different shape functions as classes."""
    fns = random.sample(ALL_SHAPE_FNS[:10], 2)

    def class_0():
        return fns[0], _sample_random_attrs()

    def class_1():
        return fns[1], _sample_random_attrs()

    return class_0, class_1


ALL_CONCEPTS = [
    concept_curved_vs_straight,
    concept_pointy_vs_smooth,
    concept_simple_vs_complex,
    concept_filled_vs_outline,
    concept_big_vs_small,
    concept_warm_vs_cool,
    concept_tall_vs_wide,
    concept_single_vs_multiple,
    concept_symmetric_vs_irregular,
    concept_open_vs_closed,
    concept_contained_vs_flat,
    concept_few_sides_vs_many,
    concept_random_shape_pair,
]


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

def generate_episode(n_support=5, n_query=5):
    """Generate a binary classification episode.

    Returns: (support_imgs, support_labels, query_imgs, query_labels)
    """
    concept_fn = random.choice(ALL_CONCEPTS)
    class_0_fn, class_1_fn = concept_fn()

    support_imgs, support_labels = [], []
    query_imgs, query_labels = [], []

    for stage_imgs, stage_labels, count in [
        (support_imgs, support_labels, n_support),
        (query_imgs, query_labels, n_query),
    ]:
        for _ in range(count):
            label = random.randint(0, 1)
            shape_fn, attrs = class_0_fn() if label == 0 else class_1_fn()
            img = render_image(shape_fn, attrs)
            stage_imgs.append(img)
            stage_labels.append(label)

    return (
        torch.stack(support_imgs),
        torch.tensor(support_labels, dtype=torch.long),
        torch.stack(query_imgs),
        torch.tensor(query_labels, dtype=torch.long),
    )


class ProceduralMetaDataset:
    """Generates meta-training episodes on the fly."""

    def __init__(self, n_support=5, n_query=5):
        self.n_support = n_support
        self.n_query = n_query

    def sample_episode(self):
        return generate_episode(self.n_support, self.n_query)


if __name__ == "__main__":
    # Quick visual test
    ds = ProceduralMetaDataset()
    s_img, s_lab, q_img, q_lab = ds.sample_episode()
    print(f"Support: {s_img.shape}, labels: {s_lab.tolist()}")
    print(f"Query:   {q_img.shape}, labels: {q_lab.tolist()}")
    print("OK")
