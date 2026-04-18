"""Feature-based meta-training episodes.

Generates binary classification episodes that force the backbone to learn
8 abstract visual features. Shapes are random procedural primitives
deliberately disjoint from the game's shape vocabulary — the backbone
learns features, not shape recognition.

The 8 features:
    1. curved edges / straight edges
    2. sharp tips / rounded tips
    3. elongated / compact
    4. has-hole / solid
    5. branching / unitary
    6. lopsided / balanced
    7. bumpy / smooth
    8. multicolor / mono

Public API:
    ProceduralMetaDataset - drop-in replacement for the old dataset
    generate_episode(n_support, n_query) - sample a random-feature episode
    EPISODE_FNS - list of all 8 episode generators (for per-feature eval)
"""

from __future__ import annotations

import colorsys
import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

IMG_SIZE = 84
CENTER = IMG_SIZE // 2


# ===========================================================================
# Color & canvas helpers
# ===========================================================================

def _random_color(hue: float | None = None,
                  sat: tuple[float, float] = (0.5, 1.0),
                  val: tuple[float, float] = (0.35, 0.9)) -> tuple[int, int, int]:
    h = hue if hue is not None else random.random()
    s = random.uniform(*sat)
    v = random.uniform(*val)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _contrast_color(base_hue: float) -> tuple[int, int, int]:
    """Pick a color with a visibly different hue."""
    h = (base_hue + random.uniform(0.35, 0.65)) % 1.0
    return _random_color(hue=h)


def _random_bg() -> tuple[int, int, int]:
    v = random.randint(232, 255)
    return (v, v, v)


def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw, tuple[int, int, int]]:
    bg = _random_bg()
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    return img, ImageDraw.Draw(img), bg


def _img_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _rotate_pts(points, angle_deg, cx, cy):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa,
             cy + (x - cx) * sa + (y - cy) * ca)
            for x, y in points]


def _random_offset(max_off: int = 8) -> tuple[int, int]:
    return random.randint(-max_off, max_off), random.randint(-max_off, max_off)


# ===========================================================================
# Shape primitive vertex generators
# ===========================================================================

def _poly_pts(cx, cy, r, n_sides, rotation=0, jitter=0.0):
    pts = []
    for i in range(n_sides):
        a = math.radians(360 * i / n_sides - 90 + rotation)
        rr = r * (1.0 + random.uniform(-jitter, jitter))
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _star_pts(cx, cy, r, n_points, rotation=0, inner_frac=None):
    if inner_frac is None:
        inner_frac = random.uniform(0.28, 0.46)
    r_in = r * inner_frac
    pts = []
    for i in range(n_points * 2):
        a = math.radians(360 * i / (n_points * 2) - 90 + rotation)
        rr = r if i % 2 == 0 else r_in
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _ellipse_pts(cx, cy, rx, ry, rotation=0, n=56):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return _rotate_pts(pts, rotation, cx, cy)


def _blob_pts(cx, cy, r, n_bumps, bump_amp, rotation=0, n=56):
    phase = random.uniform(0, 2 * math.pi)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        noise = 1.0 + bump_amp * math.sin(n_bumps * a + phase)
        rr = r * max(0.30, noise)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return _rotate_pts(pts, rotation, cx, cy)


def _cross_pts(cx, cy, r, thickness=0.30, rotation=0):
    t = r * thickness
    pts = [
        (cx - t, cy - r), (cx + t, cy - r),
        (cx + t, cy - t), (cx + r, cy - t),
        (cx + r, cy + t), (cx + t, cy + t),
        (cx + t, cy + r), (cx - t, cy + r),
        (cx - t, cy + t), (cx - r, cy + t),
        (cx - r, cy - t), (cx - t, cy - t),
    ]
    return _rotate_pts(pts, rotation, cx, cy)


def _crescent_pts(cx, cy, r, offset_frac=0.35):
    """Outer disc minus offset inner disc, sampled as a boundary polygon."""
    n = 72
    outer = [(cx + r * math.cos(2 * math.pi * i / n),
              cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    inner_cx = cx + r * offset_frac
    inner_cy = cy
    inner_r = r * 0.82
    kept = [(px, py) for px, py in outer
            if math.hypot(px - inner_cx, py - inner_cy) >= inner_r * 0.93]
    return kept if len(kept) >= 8 else outer


def _chaikin_round(points, iterations=2):
    """Smooth polygon corners via Chaikin's corner-cutting algorithm.

    Each iteration replaces every vertex with two points at 1/4 and 3/4 along
    its adjacent edges. Two iterations turns a square into a rounded-square;
    three iterations gives almost-circle smoothness.
    """
    for _ in range(iterations):
        n = len(points)
        new_pts = []
        for i in range(n):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            new_pts.append((0.75 * p0[0] + 0.25 * p1[0],
                            0.75 * p0[1] + 0.25 * p1[1]))
            new_pts.append((0.25 * p0[0] + 0.75 * p1[0],
                            0.25 * p0[1] + 0.75 * p1[1]))
        points = new_pts
    return points


# ===========================================================================
# Random compound drawers (return a PIL image on a fresh canvas)
# ===========================================================================

def _draw_fill(draw, pts, color):
    draw.polygon(pts, fill=color)


def _base_attrs():
    """Common random styling attrs used across episodes."""
    return {
        "size": random.randint(20, 34),
        "rotation": random.uniform(0, 360),
        "color": _random_color(),
        "offset": _random_offset(),
    }


# --- Curved (smooth-edge) shapes -------------------------------------------

def _draw_random_curved(img=None) -> Image.Image:
    """Random shape with curved/smooth edges."""
    if img is None:
        img, draw, _bg = _new_canvas()
    else:
        draw = ImageDraw.Draw(img)
    cx, cy = CENTER + random.randint(-8, 8), CENTER + random.randint(-8, 8)
    r = random.randint(20, 34)
    color = _random_color()
    choice = random.choice(["circle", "ellipse", "blob_smooth"])
    if choice == "circle":
        pts = _ellipse_pts(cx, cy, r, r)
    elif choice == "ellipse":
        aspect = random.uniform(0.55, 0.85)
        pts = _ellipse_pts(cx, cy, r, r * aspect, rotation=random.uniform(0, 360))
    else:  # blob with low bump amp → smooth curves
        pts = _blob_pts(cx, cy, r, n_bumps=random.randint(2, 4),
                        bump_amp=random.uniform(0.04, 0.10))
    _draw_fill(draw, pts, color)
    return img


# --- Straight (polygon) shapes ---------------------------------------------

def _draw_random_straight(img=None) -> Image.Image:
    """Random shape with straight edges (polygon, no jitter)."""
    if img is None:
        img, draw, _bg = _new_canvas()
    else:
        draw = ImageDraw.Draw(img)
    cx, cy = CENTER + random.randint(-8, 8), CENTER + random.randint(-8, 8)
    r = random.randint(20, 34)
    color = _random_color()
    n_sides = random.choice([3, 4, 5])
    pts = _poly_pts(cx, cy, r, n_sides,
                    rotation=random.uniform(0, 360), jitter=0.0)
    _draw_fill(draw, pts, color)
    return img


# ===========================================================================
# Feature episode generators
# ===========================================================================

def _episode_template(gen_class_0, gen_class_1, n_support, n_query):
    """Build a balanced binary classification episode."""
    s_imgs, s_labels = [], []
    q_imgs, q_labels = [], []
    for _ in range(n_support):
        s_imgs.append(_img_to_tensor(gen_class_0()))
        s_labels.append(0)
        s_imgs.append(_img_to_tensor(gen_class_1()))
        s_labels.append(1)
    for _ in range(n_query):
        q_imgs.append(_img_to_tensor(gen_class_0()))
        q_labels.append(0)
        q_imgs.append(_img_to_tensor(gen_class_1()))
        q_labels.append(1)
    # Shuffle within each set (so labels aren't ordered)
    s_perm = torch.randperm(len(s_imgs))
    q_perm = torch.randperm(len(q_imgs))
    return (
        torch.stack(s_imgs)[s_perm],
        torch.tensor(s_labels, dtype=torch.long)[s_perm],
        torch.stack(q_imgs)[q_perm],
        torch.tensor(q_labels, dtype=torch.long)[q_perm],
    )


# --- 1. Curved edges vs Straight edges -------------------------------------

def episode_curved_vs_straight(n_support=5, n_query=5):
    return _episode_template(
        _draw_random_straight, _draw_random_curved, n_support, n_query)


# --- 2. Sharp tips vs Rounded tips -----------------------------------------
# Sharp = has acute vertices (<~80°). Rounded = smooth or wide corners (≥90°).

def _draw_random_sharp() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-8, 8), CENTER + random.randint(-8, 8)
    r = random.randint(20, 34)
    color = _random_color()
    choice = random.choice(["triangle", "star", "arrow_head", "spike"])
    if choice == "triangle":
        pts = _poly_pts(cx, cy, r, 3, rotation=random.uniform(0, 360))
    elif choice == "star":
        n_points = random.randint(4, 7)
        inner = random.uniform(0.18, 0.35)  # low inner_frac → very pointy
        pts = _star_pts(cx, cy, r, n_points,
                        rotation=random.uniform(0, 360), inner_frac=inner)
    elif choice == "arrow_head":
        # Isoceles triangle, very narrow — clearly sharp tip
        rotation = random.uniform(0, 360)
        base_w = r * random.uniform(0.35, 0.55)
        pts = [(cx, cy - r), (cx + base_w, cy + r * 0.5),
               (cx - base_w, cy + r * 0.5)]
        pts = _rotate_pts(pts, rotation, cx, cy)
    else:  # spike: 4-pointed shuriken-like star with very narrow points
        pts = _star_pts(cx, cy, r, 4,
                        rotation=random.uniform(0, 360),
                        inner_frac=random.uniform(0.15, 0.25))
    _draw_fill(draw, pts, color)
    return img


def _draw_random_rounded() -> Image.Image:
    """Not-sharp: anything without acute vertices.

    Includes: curved shapes (circle, ellipse, blob), many-sided polygons
    (whose interior angles are ≥135°), genuinely rounded-corner shapes
    built via Chaikin smoothing, and rounded-lobe "fat" stars.
    """
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-8, 8), CENTER + random.randint(-8, 8)
    r = random.randint(20, 34)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice([
        "circle", "ellipse", "blob",
        "high_poly", "high_poly",  # many-sided polygons
        "rounded_square", "rounded_poly", "rounded_poly",  # chaikin-smoothed
        "fat_star",  # star with high inner_frac — rounded lobes
    ])
    if choice == "circle":
        pts = _ellipse_pts(cx, cy, r, r)
    elif choice == "ellipse":
        aspect = random.uniform(0.55, 0.85)
        pts = _ellipse_pts(cx, cy, r, r * aspect, rotation=rotation)
    elif choice == "blob":
        pts = _blob_pts(cx, cy, r, n_bumps=random.randint(2, 4),
                        bump_amp=random.uniform(0.05, 0.12))
    elif choice == "high_poly":
        # 8-12 sided polygons — interior angles 135°-150°, visually "rounded"
        n = random.choice([8, 9, 10, 11, 12])
        pts = _poly_pts(cx, cy, r, n, rotation=rotation)
    elif choice == "rounded_square":
        # Square with Chaikin-smoothed corners — the poster child for
        # "has corners but they're rounded"
        base = _poly_pts(cx, cy, r, 4, rotation=rotation)
        pts = _chaikin_round(base, iterations=random.randint(1, 2))
    elif choice == "rounded_poly":
        # Pentagon/hexagon/heptagon with rounded corners
        n = random.choice([3, 4, 5, 6, 7])
        base = _poly_pts(cx, cy, r, n, rotation=rotation)
        pts = _chaikin_round(base, iterations=random.randint(1, 2))
    else:  # fat_star: star with high inner_frac → rounded scalloped lobes
        n_points = random.randint(5, 9)
        inner = random.uniform(0.68, 0.85)  # very fat → rounded bumps
        base = _star_pts(cx, cy, r, n_points, rotation=rotation, inner_frac=inner)
        pts = _chaikin_round(base, iterations=2)
    _draw_fill(draw, pts, color)
    return img


def episode_sharp_vs_rounded(n_support=5, n_query=5):
    return _episode_template(
        _draw_random_rounded, _draw_random_sharp, n_support, n_query)


# --- 3. Elongated vs Compact -----------------------------------------------

def _draw_random_elongated() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice(["ellipse", "rect", "stretched_poly"])
    if choice == "ellipse":
        rx = random.randint(26, 36)
        ry = rx // random.randint(3, 5)  # aspect 3:1 to 5:1
        pts = _ellipse_pts(cx, cy, rx, ry, rotation=rotation)
    elif choice == "rect":
        rx = random.randint(26, 36)
        ry = rx // random.randint(3, 5)
        pts = [(cx - rx, cy - ry), (cx + rx, cy - ry),
               (cx + rx, cy + ry), (cx - rx, cy + ry)]
        pts = _rotate_pts(pts, rotation, cx, cy)
    else:  # stretched polygon
        n = random.choice([3, 4, 5, 6])
        pts = _poly_pts(cx, cy, random.randint(24, 32), n, rotation=rotation)
        # stretch along x
        stretch = random.uniform(2.0, 3.2)
        pts = [((x - cx) * stretch + cx, y) for x, y in pts]
        pts = _rotate_pts(pts, rotation, cx, cy)
    _draw_fill(draw, pts, color)
    return img


def _draw_random_compact() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(18, 30)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice(["circle", "square", "poly", "blob"])
    if choice == "circle":
        pts = _ellipse_pts(cx, cy, r, r)
    elif choice == "square":
        pts = _poly_pts(cx, cy, r, 4, rotation=rotation)
    elif choice == "poly":
        pts = _poly_pts(cx, cy, r, random.choice([3, 5, 6, 7]), rotation=rotation)
    else:
        pts = _blob_pts(cx, cy, r, n_bumps=random.randint(2, 4), bump_amp=0.08)
    _draw_fill(draw, pts, color)
    return img


def episode_elongated_vs_compact(n_support=5, n_query=5):
    return _episode_template(
        _draw_random_compact, _draw_random_elongated, n_support, n_query)


# --- 4. Has-hole vs Solid --------------------------------------------------

def _draw_shape_for_hole_test(cx, cy, r, color, rotation):
    """Return (pts, shape_type). Used for both has-hole and solid classes."""
    choice = random.choice(["circle", "poly", "blob", "rounded_rect"])
    if choice == "circle":
        return _ellipse_pts(cx, cy, r, r), "circle"
    if choice == "poly":
        return _poly_pts(cx, cy, r, random.choice([4, 5, 6, 7, 8]),
                         rotation=rotation), "poly"
    if choice == "blob":
        return _blob_pts(cx, cy, r, n_bumps=random.randint(2, 4),
                         bump_amp=0.10), "blob"
    # rounded rect ≈ ellipse stretched
    aspect = random.uniform(0.55, 0.85)
    return _ellipse_pts(cx, cy, r, r * aspect, rotation=rotation), "rr"


def _draw_has_hole() -> Image.Image:
    img, draw, bg = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(24, 34)
    color = _random_color()
    rotation = random.uniform(0, 360)
    pts, _ = _draw_shape_for_hole_test(cx, cy, r, color, rotation)
    _draw_fill(draw, pts, color)
    # Punch 1-3 holes in bg color
    n_holes = random.choice([1, 1, 1, 2, 3])
    for _ in range(n_holes):
        hr = int(r * random.uniform(0.18, 0.40))
        # place hole inside the shape (near center for 1-hole, scattered for multi)
        if n_holes == 1:
            hx, hy = cx + random.randint(-2, 2), cy + random.randint(-2, 2)
        else:
            offset = int(r * 0.45)
            hx = cx + random.randint(-offset, offset)
            hy = cy + random.randint(-offset, offset)
        if random.random() < 0.5:
            draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=bg)
        else:
            draw.rectangle([hx - hr, hy - hr, hx + hr, hy + hr], fill=bg)
    return img


def _draw_solid() -> Image.Image:
    img, draw, _bg = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(24, 34)
    color = _random_color()
    rotation = random.uniform(0, 360)
    pts, _ = _draw_shape_for_hole_test(cx, cy, r, color, rotation)
    _draw_fill(draw, pts, color)
    return img


def episode_has_hole_vs_solid(n_support=5, n_query=5):
    return _episode_template(_draw_solid, _draw_has_hole, n_support, n_query)


# --- 5. Branching vs Unitary -----------------------------------------------

def _draw_branching() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(24, 34)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice(["cross", "asterisk", "windmill", "spoke_star"])
    if choice == "cross":
        pts = _cross_pts(cx, cy, r,
                         thickness=random.uniform(0.18, 0.32),
                         rotation=rotation)
        _draw_fill(draw, pts, color)
    elif choice == "asterisk":
        # draw thick line segments from center
        n_arms = random.choice([3, 4, 5, 6])
        width = random.randint(4, 7)
        for i in range(n_arms):
            a = math.radians(360 * i / n_arms + rotation)
            ex = cx + r * math.cos(a)
            ey = cy + r * math.sin(a)
            draw.line([(cx, cy), (ex, ey)], fill=color, width=width)
    elif choice == "windmill":
        # 4 tapered blades
        for i in range(4):
            a = math.radians(90 * i + rotation)
            ca, sa = math.cos(a), math.sin(a)
            blade = [
                (cx, cy),
                (cx + r * ca - r * 0.25 * sa, cy + r * sa + r * 0.25 * ca),
                (cx + r * 0.7 * ca - r * 0.4 * sa, cy + r * 0.7 * sa + r * 0.4 * ca),
            ]
            draw.polygon(blade, fill=color)
    else:  # spoke_star: long narrow points
        pts = _star_pts(cx, cy, r, n_points=random.choice([4, 5, 6]),
                        rotation=rotation, inner_frac=random.uniform(0.18, 0.30))
        _draw_fill(draw, pts, color)
    return img


def _draw_unitary() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(22, 32)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice(["circle", "ellipse", "poly", "blob"])
    if choice == "circle":
        pts = _ellipse_pts(cx, cy, r, r)
    elif choice == "ellipse":
        pts = _ellipse_pts(cx, cy, r, r * random.uniform(0.6, 0.9),
                           rotation=rotation)
    elif choice == "poly":
        pts = _poly_pts(cx, cy, r, random.choice([3, 4, 5, 6]), rotation=rotation)
    else:
        pts = _blob_pts(cx, cy, r, n_bumps=random.randint(2, 4),
                        bump_amp=random.uniform(0.05, 0.12))
    _draw_fill(draw, pts, color)
    return img


def episode_branching_vs_unitary(n_support=5, n_query=5):
    return _episode_template(_draw_unitary, _draw_branching, n_support, n_query)


# --- 6. Lopsided vs Balanced -----------------------------------------------

def _draw_lopsided() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(24, 34)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice(["crescent", "crescent", "offset_blob", "j_shape", "teardrop"])
    if choice == "crescent":
        pts = _crescent_pts(cx, cy, r,
                            offset_frac=random.uniform(0.32, 0.55))
        pts = _rotate_pts(pts, rotation, cx, cy)
        _draw_fill(draw, pts, color)
    elif choice == "offset_blob":
        # Blob whose center is offset from canvas center — mass clearly on one side
        dx = int(r * random.uniform(0.35, 0.55)) * random.choice([-1, 1])
        dy = int(r * random.uniform(0.0, 0.25)) * random.choice([-1, 1])
        pts = _blob_pts(cx + dx, cy + dy, r,
                        n_bumps=random.randint(2, 4),
                        bump_amp=random.uniform(0.05, 0.15))
        _draw_fill(draw, pts, color)
    elif choice == "j_shape":
        # A bigger blob on one side + a smaller one attached to it
        dx_big = -int(r * 0.35)
        pts1 = _ellipse_pts(cx + dx_big, cy, int(r * 0.75), r,
                            rotation=rotation)
        _draw_fill(draw, pts1, color)
        pts2 = _ellipse_pts(cx + int(r * 0.55),
                            cy + int(r * 0.4),
                            int(r * 0.45), int(r * 0.45),
                            rotation=rotation)
        _draw_fill(draw, pts2, color)
    else:  # teardrop: exaggerated asymmetric egg
        n = 56
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            # Much stronger asymmetric radius — narrow on one end
            rr = r * (1.0 - 0.55 * math.cos(a))
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        pts = _rotate_pts(pts, rotation, cx, cy)
        _draw_fill(draw, pts, color)
    return img


def _draw_balanced() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-4, 4), CENTER + random.randint(-4, 4)
    r = random.randint(22, 32)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice(["circle", "square", "poly", "sym_star", "sym_blob"])
    if choice == "circle":
        pts = _ellipse_pts(cx, cy, r, r)
    elif choice == "square":
        pts = _poly_pts(cx, cy, r, 4, rotation=rotation)
    elif choice == "poly":
        pts = _poly_pts(cx, cy, r, random.choice([3, 5, 6]), rotation=rotation)
    elif choice == "sym_star":
        pts = _star_pts(cx, cy, r, random.choice([4, 5, 6]), rotation=rotation)
    else:  # symmetric blob — centered at cx,cy
        pts = _blob_pts(cx, cy, r, n_bumps=random.choice([4, 6, 8]),
                        bump_amp=random.uniform(0.05, 0.10))
    _draw_fill(draw, pts, color)
    return img


def episode_lopsided_vs_balanced(n_support=5, n_query=5):
    return _episode_template(_draw_balanced, _draw_lopsided, n_support, n_query)


# --- 7. Bumpy vs Smooth ----------------------------------------------------

def _draw_bumpy() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(24, 32)
    color = _random_color()
    rotation = random.uniform(0, 360)
    # Heavily bumpy blob — many small bumps around the outline
    n_bumps = random.choice([6, 7, 8, 9, 10, 11])
    bump_amp = random.uniform(0.18, 0.30)
    pts = _blob_pts(cx, cy, r, n_bumps=n_bumps, bump_amp=bump_amp, rotation=rotation)
    _draw_fill(draw, pts, color)
    return img


def _draw_smooth() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(22, 32)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice(["circle", "ellipse", "poly", "gentle_blob"])
    if choice == "circle":
        pts = _ellipse_pts(cx, cy, r, r)
    elif choice == "ellipse":
        pts = _ellipse_pts(cx, cy, r, r * random.uniform(0.6, 0.85), rotation=rotation)
    elif choice == "poly":
        pts = _poly_pts(cx, cy, r, random.choice([3, 4, 5, 6, 7, 8]), rotation=rotation)
    else:
        pts = _blob_pts(cx, cy, r, n_bumps=random.randint(2, 3),
                        bump_amp=random.uniform(0.03, 0.07))
    _draw_fill(draw, pts, color)
    return img


def episode_bumpy_vs_smooth(n_support=5, n_query=5):
    return _episode_template(_draw_smooth, _draw_bumpy, n_support, n_query)


# --- 8. Multicolor vs Mono -------------------------------------------------

def _draw_multicolor() -> Image.Image:
    img, draw, bg = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(24, 34)
    hue_a = random.random()
    color_a = _random_color(hue=hue_a)
    color_b = _contrast_color(hue_a)
    rotation = random.uniform(0, 360)
    style = random.choice(["nested", "half_and_half", "stem_and_body", "striped"])

    if style == "nested":
        # outer shape in color_a, inner shape in color_b
        outer_shape = random.choice(["circle", "poly", "blob"])
        if outer_shape == "circle":
            pts_out = _ellipse_pts(cx, cy, r, r)
        elif outer_shape == "poly":
            pts_out = _poly_pts(cx, cy, r, random.choice([4, 5, 6]), rotation=rotation)
        else:
            pts_out = _blob_pts(cx, cy, r, n_bumps=3, bump_amp=0.08)
        _draw_fill(draw, pts_out, color_a)
        inner_r = int(r * random.uniform(0.40, 0.60))
        inner_shape = random.choice(["circle", "poly"])
        if inner_shape == "circle":
            pts_in = _ellipse_pts(cx, cy, inner_r, inner_r)
        else:
            pts_in = _poly_pts(cx, cy, inner_r, random.choice([3, 4, 5, 6]),
                               rotation=random.uniform(0, 360))
        _draw_fill(draw, pts_in, color_b)

    elif style == "half_and_half":
        # draw full shape in color_a, then overwrite one half with color_b
        pts = _ellipse_pts(cx, cy, r, r)
        _draw_fill(draw, pts, color_a)
        # overlay a rectangle covering half
        if random.random() < 0.5:
            draw.rectangle([cx, cy - r - 2, cx + r + 2, cy + r + 2], fill=color_b)
            # re-clip by drawing the ellipse outline mask? Simplest: redraw other half
            half_pts = [(px, py) for px, py in pts if px >= cx]
            if len(half_pts) >= 3:
                _draw_fill(draw, half_pts + [(cx, cy - r), (cx, cy + r)], color_b)
        else:
            half_pts = [(px, py) for px, py in pts if py >= cy]
            if len(half_pts) >= 3:
                _draw_fill(draw, half_pts + [(cx - r, cy), (cx + r, cy)], color_b)

    elif style == "stem_and_body":
        # body on top, stem below (different colors)
        body_r = int(r * random.uniform(0.65, 0.9))
        body_shape = random.choice(["circle", "poly", "blob"])
        if body_shape == "circle":
            body_pts = _ellipse_pts(cx, cy - int(r * 0.25), body_r, body_r)
        elif body_shape == "poly":
            body_pts = _poly_pts(cx, cy - int(r * 0.25), body_r,
                                 random.choice([3, 4, 5, 6]))
        else:
            body_pts = _blob_pts(cx, cy - int(r * 0.25), body_r, n_bumps=3, bump_amp=0.08)
        _draw_fill(draw, body_pts, color_a)
        stem_w = random.randint(4, 9)
        stem_h = random.randint(int(r * 0.4), int(r * 0.8))
        draw.rectangle([cx - stem_w, cy + int(r * 0.2),
                        cx + stem_w, cy + int(r * 0.2) + stem_h],
                       fill=color_b)

    else:  # striped
        pts = _ellipse_pts(cx, cy, r, r)
        _draw_fill(draw, pts, color_a)
        # draw diagonal stripes in color_b
        stripe_w = random.randint(4, 7)
        gap = stripe_w * 2
        for i in range(-r, r + 1, gap):
            draw.line([(cx + i - r, cy - r), (cx + i + r, cy + r)],
                      fill=color_b, width=stripe_w)
        # clip by re-overlaying ellipse outline in background... simpler to accept
        # stripes spilling slightly — the feature signal is still strong

    return img


def _draw_mono() -> Image.Image:
    img, draw, _ = _new_canvas()
    cx, cy = CENTER + random.randint(-6, 6), CENTER + random.randint(-6, 6)
    r = random.randint(22, 32)
    color = _random_color()
    rotation = random.uniform(0, 360)
    choice = random.choice(["circle", "ellipse", "poly", "blob", "star"])
    if choice == "circle":
        pts = _ellipse_pts(cx, cy, r, r)
    elif choice == "ellipse":
        pts = _ellipse_pts(cx, cy, r, r * random.uniform(0.55, 0.85), rotation=rotation)
    elif choice == "poly":
        pts = _poly_pts(cx, cy, r, random.choice([3, 4, 5, 6, 7, 8]), rotation=rotation)
    elif choice == "blob":
        pts = _blob_pts(cx, cy, r, n_bumps=random.randint(2, 5),
                        bump_amp=random.uniform(0.05, 0.15))
    else:
        pts = _star_pts(cx, cy, r, random.choice([4, 5, 6]), rotation=rotation)
    _draw_fill(draw, pts, color)
    return img


def episode_multicolor_vs_mono(n_support=5, n_query=5):
    return _episode_template(_draw_mono, _draw_multicolor, n_support, n_query)


# ===========================================================================
# Public API
# ===========================================================================

EPISODE_FNS = [
    episode_curved_vs_straight,
    episode_sharp_vs_rounded,
    episode_elongated_vs_compact,
    episode_has_hole_vs_solid,
    episode_branching_vs_unitary,
    episode_lopsided_vs_balanced,
    episode_bumpy_vs_smooth,
    episode_multicolor_vs_mono,
]

FEATURE_NAMES = [
    "curved_vs_straight",
    "sharp_vs_rounded",
    "elongated_vs_compact",
    "has_hole_vs_solid",
    "branching_vs_unitary",
    "lopsided_vs_balanced",
    "bumpy_vs_smooth",
    "multicolor_vs_mono",
]


def generate_episode(n_support=5, n_query=5):
    """Sample a random feature and generate a balanced binary episode."""
    fn = random.choice(EPISODE_FNS)
    return fn(n_support, n_query)


class ProceduralMetaDataset:
    """Drop-in replacement for the old dataset — samples a random feature
    per episode. Used by ``meta_training/train_general.py``."""

    def __init__(self, n_support: int = 5, n_query: int = 5):
        self.n_support = n_support
        self.n_query = n_query

    def sample_episode(self):
        return generate_episode(self.n_support, self.n_query)


if __name__ == "__main__":
    # Quick smoke test
    for name, fn in zip(FEATURE_NAMES, EPISODE_FNS):
        s_img, s_lab, q_img, q_lab = fn(5, 5)
        assert s_img.shape == (10, 3, 84, 84), f"{name}: bad support shape"
        assert q_img.shape == (10, 3, 84, 84), f"{name}: bad query shape"
        assert set(s_lab.tolist()) == {0, 1}, f"{name}: labels missing"
        assert set(q_lab.tolist()) == {0, 1}, f"{name}: labels missing"
        print(f"[OK] {name}: support {s_img.shape}, query {q_img.shape}")
    print("\nAll 8 feature episodes OK.")
