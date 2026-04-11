"""Generate simple game object images programmatically."""

import math
import random

import pygame
import torch

COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 80, 220),
    "green": (50, 180, 50),
    "yellow": (230, 210, 40),
    "purple": (150, 50, 200),
    "orange": (240, 140, 30),
}

SIZES = {"small": 20, "medium": 35, "large": 50}

SHAPES = ["circle", "square", "triangle", "star", "diamond"]


def generate_object_image(shape: str, color: str | tuple, size: str | int) -> pygame.Surface:
    """Generate an 84x84 surface with a colored shape."""
    surf = pygame.Surface((84, 84))
    surf.fill((255, 255, 255))

    rgb = COLORS[color] if isinstance(color, str) else color
    r = SIZES[size] if isinstance(size, str) else size
    cx, cy = 42, 42

    if shape == "circle":
        pygame.draw.circle(surf, rgb, (cx, cy), r)
    elif shape == "square":
        rect = pygame.Rect(cx - r, cy - r, r * 2, r * 2)
        pygame.draw.rect(surf, rgb, rect)
    elif shape == "triangle":
        pts = [
            (cx, cy - r),
            (cx - r, cy + r),
            (cx + r, cy + r),
        ]
        pygame.draw.polygon(surf, rgb, pts)
    elif shape == "star":
        pts = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            rad = r if i % 2 == 0 else r * 0.4
            pts.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
        pygame.draw.polygon(surf, rgb, pts)
    elif shape == "diamond":
        pts = [
            (cx, cy - r),
            (cx + r * 0.6, cy),
            (cx, cy + r),
            (cx - r * 0.6, cy),
        ]
        pygame.draw.polygon(surf, rgb, pts)

    return surf


def generate_toy_box() -> list[tuple[pygame.Surface, str]]:
    """Generate ~20 objects across 5 shape categories."""
    items = []
    for shape in SHAPES:
        colors = random.sample(list(COLORS.keys()), min(4, len(COLORS)))
        for color in colors:
            size = random.choice(list(SIZES.keys()))
            surf = generate_object_image(shape, color, size)
            items.append((surf, shape))
    return items


def generate_sort_objects(n: int = 10) -> list[tuple[pygame.Surface, dict]]:
    """Generate objects with random attributes for sorting."""
    items = []
    for _ in range(n):
        shape = random.choice(SHAPES)
        color_name = random.choice(list(COLORS.keys()))
        size_name = random.choice(list(SIZES.keys()))
        surf = generate_object_image(shape, color_name, size_name)
        attrs = {"shape": shape, "color": color_name, "size": size_name}
        items.append((surf, attrs))
    return items


def surface_to_tensor(surface: pygame.Surface) -> torch.Tensor:
    """Convert pygame surface to 84x84x3 normalized float tensor."""
    arr = pygame.surfarray.array3d(surface)  # (W, H, 3)
    arr = arr.transpose(1, 0, 2)  # (H, W, 3)
    tensor = torch.from_numpy(arr.copy()).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # (3, H, W) for Conv4
    return tensor
