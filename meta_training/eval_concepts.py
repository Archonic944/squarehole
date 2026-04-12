"""Evaluate a trained checkpoint on each procedural concept."""

from __future__ import annotations

import argparse
import os
import random
import sys

import learn2learn as l2l
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.conv4 import Conv4WithHead
from meta_training.procedural_meta import ALL_CONCEPTS, render_image


DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "app",
    "models",
    "checkpoints",
    "general_conv4_128.pt",
)


def evaluate(
    checkpoint: str,
    hidden: int = 128,
    episodes: int = 20,
    support: int = 8,
    query: int = 8,
    inner_steps: int = 7,
    device: str = "cpu",
):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = Conv4WithHead(num_classes=2, hidden=hidden).to(device)
    maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    filtered = {k: v for k, v in state.items() if "head" not in k}
    maml.load_state_dict(filtered, strict=False)

    print(f"Checkpoint: {checkpoint}")
    print(f"Backbone: Conv4-{hidden}")
    print(
        f"Eval setup: episodes={episodes}, support={support}, "
        f"query={query}, inner_steps={inner_steps}, device={device}"
    )
    print()

    for concept_fn in ALL_CONCEPTS:
        name = concept_fn.__name__.replace("concept_", "")
        accs = []
        for _ in range(episodes):
            class_0_fn, class_1_fn = concept_fn()

            s_imgs, s_labs, q_imgs, q_labs = [], [], [], []
            for imgs, labs, n in [(s_imgs, s_labs, support), (q_imgs, q_labs, query)]:
                for _ in range(n):
                    label = random.randint(0, 1)
                    shape_fn, attrs = class_0_fn() if label == 0 else class_1_fn()
                    imgs.append(render_image(shape_fn, attrs))
                    labs.append(label)

            s_imgs = torch.stack(s_imgs).to(device)
            s_labs = torch.tensor(s_labs, dtype=torch.long, device=device)
            q_imgs = torch.stack(q_imgs).to(device)
            q_labs = torch.tensor(q_labs, dtype=torch.long, device=device)

            learner = maml.clone()
            for _ in range(inner_steps):
                preds = learner(s_imgs)
                learner.adapt(F.cross_entropy(preds, s_labs))

            with torch.no_grad():
                preds = learner(q_imgs)
                acc = (preds.argmax(1) == q_labs).float().mean().item()
            accs.append(acc)

        mean = sum(accs) / len(accs) * 100
        bar = "#" * int(mean / 2.5)
        print(f"  {name:<25} {mean:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--support", type=int, default=8)
    parser.add_argument("--query", type=int, default=8)
    parser.add_argument("--inner-steps", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        hidden=args.hidden,
        episodes=args.episodes,
        support=args.support,
        query=args.query,
        inner_steps=args.inner_steps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
