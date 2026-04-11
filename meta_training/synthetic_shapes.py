import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle", "star", "diamond"]
COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 200, 0),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
}
SIZES = {"small": 14, "medium": 24, "large": 34}
THICKNESSES = {"thin": 1, "thick": 3}
PATTERNS = ["solid", "striped"]

IMG_SIZE = 84


def _star_points(cx, cy, r, n=5):
    points = []
    for i in range(2 * n):
        angle = math.pi / 2 + i * math.pi / n
        rad = r if i % 2 == 0 else r * 0.4
        points.append((cx + rad * math.cos(angle), cy - rad * math.sin(angle)))
    return points


def render_shape(shape, color_name, size_name, thickness_name, pattern):
    """Render a single shape on a white 84x84 image and return a (3, 84, 84) tensor."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    rgb = COLORS[color_name]
    r = SIZES[size_name]
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    thick = THICKNESSES[thickness_name]

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=rgb, width=thick)
        if pattern == "solid":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=rgb, width=thick)
        if pattern == "solid":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "triangle":
        pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw.polygon(pts, outline=rgb)
        if pattern == "solid":
            draw.polygon(pts, fill=rgb)
        else:
            for i in range(len(pts)):
                draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=rgb, width=thick)
    elif shape == "star":
        pts = _star_points(cx, cy, r)
        draw.polygon(pts, outline=rgb)
        if pattern == "solid":
            draw.polygon(pts, fill=rgb)
    elif shape == "diamond":
        pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(pts, outline=rgb)
        if pattern == "solid":
            draw.polygon(pts, fill=rgb)

    # Stripes overlay
    if pattern == "striped":
        for y in range(0, IMG_SIZE, 6):
            draw.line([(0, y), (IMG_SIZE, y)], fill=(200, 200, 200), width=1)

    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, 84, 84)


def random_attributes():
    return {
        "shape": random.choice(SHAPES),
        "color": random.choice(list(COLORS.keys())),
        "size": random.choice(list(SIZES.keys())),
        "thickness": random.choice(list(THICKNESSES.keys())),
        "pattern": random.choice(PATTERNS),
    }


# Binary split definitions for each attribute
_SPLITS = {
    "shape": [
        ({"circle", "square"}, {"triangle", "star", "diamond"}),
        ({"circle", "triangle"}, {"square", "star", "diamond"}),
        ({"star", "diamond"}, {"circle", "square", "triangle"}),
    ],
    "color": [
        ({"red", "orange", "yellow"}, {"blue", "green", "purple"}),
        ({"red", "blue"}, {"green", "yellow", "purple", "orange"}),
    ],
    "size": [
        ({"small"}, {"medium", "large"}),
        ({"small", "medium"}, {"large"}),
    ],
    "thickness": [
        ({"thin"}, {"thick"}),
    ],
    "pattern": [
        ({"solid"}, {"striped"}),
    ],
}


def generate_episode(n_support=5, n_query=5):
    """Generate a binary classification episode.

    Returns: (support_images, support_labels, query_images, query_labels)
    Each images tensor is (N, 3, 84, 84), labels tensor is (N,) with values 0 or 1.
    """
    attr = random.choice(list(_SPLITS.keys()))
    group0, group1 = random.choice(_SPLITS[attr])

    support_imgs, support_labs = [], []
    query_imgs, query_labs = [], []

    # Generate support and query for each bin
    for stage_imgs, stage_labs, count in [
        (support_imgs, support_labs, n_support),
        (query_imgs, query_labs, n_query),
    ]:
        for _ in range(count):
            bin_label = random.randint(0, 1)
            group = group0 if bin_label == 0 else group1
            attrs = random_attributes()
            attrs[attr] = random.choice(list(group))
            img = render_shape(
                attrs["shape"], attrs["color"], attrs["size"],
                attrs["thickness"], attrs["pattern"],
            )
            stage_imgs.append(img)
            stage_labs.append(bin_label)

    return (
        torch.stack(support_imgs),
        torch.tensor(support_labs, dtype=torch.long),
        torch.stack(query_imgs),
        torch.tensor(query_labs, dtype=torch.long),
    )


class SyntheticShapesDataset:
    """Iterable-style dataset that generates episodes on the fly."""

    def __init__(self, n_support=5, n_query=5):
        self.n_support = n_support
        self.n_query = n_query

    def sample_episode(self):
        return generate_episode(self.n_support, self.n_query)
