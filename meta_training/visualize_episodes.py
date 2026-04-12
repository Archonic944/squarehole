"""Render one sample episode per feature as a PNG grid, for sanity checking.

Run: .venv/bin/python meta_training/visualize_episodes.py
Outputs: meta_training/debug_episodes/<feature>.png
"""

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meta_training.procedural_meta import (  # noqa: E402
    EPISODE_FNS,
    FEATURE_NAMES,
    IMG_SIZE,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "debug_episodes")


def tensor_to_pil(t):
    """(3, 84, 84) float tensor → PIL Image."""
    import numpy as np
    arr = (t.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_grid(name: str, fn) -> Image.Image:
    s_img, s_lab, q_img, q_lab = fn(n_support=5, n_query=5)

    # 2 rows × 10 cols grid: row 0 = class 0 support, row 1 = class 1 support
    # Then another 2 rows for query, with a gap row
    col_count = 10
    row_count = 4
    gap = 4
    W = col_count * IMG_SIZE + (col_count - 1) * gap
    H = row_count * IMG_SIZE + (row_count - 1) * gap + 40
    grid = Image.new("RGB", (W, H), (255, 255, 255))

    # Group support and query images by label
    def group(imgs, labels):
        by_label = {0: [], 1: []}
        for img, lab in zip(imgs, labels.tolist()):
            by_label[lab].append(img)
        return by_label

    s_groups = group(s_img, s_lab)
    q_groups = group(q_img, q_lab)

    def paste_row(row_idx, imgs, label_text):
        y = row_idx * (IMG_SIZE + gap) + 20
        for col, im_t in enumerate(imgs[:col_count]):
            pil = tensor_to_pil(im_t)
            x = col * (IMG_SIZE + gap)
            grid.paste(pil, (x, y))

    paste_row(0, s_groups[0], "S0")
    paste_row(1, s_groups[1], "S1")
    paste_row(2, q_groups[0], "Q0")
    paste_row(3, q_groups[1], "Q1")

    return grid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, fn in zip(FEATURE_NAMES, EPISODE_FNS):
        grid = build_grid(name, fn)
        path = os.path.join(OUT_DIR, f"{name}.png")
        grid.save(path)
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
