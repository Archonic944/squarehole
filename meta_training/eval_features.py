"""Per-feature accuracy evaluator for the meta-trained backbone.

For each of the 8 visual features, generates N held-out episodes, adapts
the backbone on the support set (MAML inner loop), and measures accuracy
on the query set. Reports per-feature accuracy and a final average.

Usage:
    .venv/bin/python meta_training/eval_features.py [--ckpt PATH] [--episodes 50]
"""

import argparse
import os
import sys

import learn2learn as l2l
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.conv4 import Conv4WithHead  # noqa: E402
from meta_training.procedural_meta import (  # noqa: E402
    EPISODE_FNS,
    FEATURE_NAMES,
)

CKPT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "app", "models", "checkpoints"
)


def load_backbone(ckpt_path: str, hidden: int, device: torch.device) -> l2l.algorithms.MAML:
    model = Conv4WithHead(num_classes=2, hidden=hidden).to(device)
    maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    maml.load_state_dict(state, strict=True)
    return maml


def eval_feature(maml, episode_fn, n_episodes: int, inner_steps: int,
                 n_support: int, n_query: int, device: torch.device) -> float:
    correct = 0
    total = 0
    for _ in range(n_episodes):
        s_img, s_lab, q_img, q_lab = episode_fn(n_support, n_query)
        s_img = s_img.to(device)
        s_lab = s_lab.to(device)
        q_img = q_img.to(device)
        q_lab = q_lab.to(device)

        learner = maml.clone()
        learner.train(True)
        # Inner adaptation on support set
        for _ in range(inner_steps):
            preds = learner(s_img)
            loss = F.cross_entropy(preds, s_lab)
            learner.adapt(loss)

        # Switch to evaluation mode (BN uses running stats from meta-training)
        learner.train(False)
        with torch.no_grad():
            q_preds = learner(q_img)
            pred_labels = q_preds.argmax(dim=1)
            correct += (pred_labels == q_lab).sum().item()
            total += q_lab.numel()

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(CKPT_DIR, "general_conv4_64_robust.pt"),
                        help="Path to the trained MAML checkpoint")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of held-out episodes per feature")
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--support", type=int, default=5)
    parser.add_argument("--query", type=int, default=5)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"ERROR: checkpoint not found")
        sys.exit(1)

    maml = load_backbone(args.ckpt, args.hidden, device)
    print(f"Backbone loaded (hidden={args.hidden})")
    print(f"Evaluating {args.episodes} episodes per feature, "
          f"{args.inner_steps} inner steps\n")

    results = {}
    for name, fn in zip(FEATURE_NAMES, EPISODE_FNS):
        acc = eval_feature(maml, fn, args.episodes, args.inner_steps,
                           args.support, args.query, device)
        results[name] = acc
        bar = "=" * int(acc * 40)
        print(f"  {name:<26} {acc * 100:5.1f}%  {bar}")

    avg = sum(results.values()) / len(results)
    print(f"\n  {'AVERAGE':<26} {avg * 100:5.1f}%")

    # Success threshold from the design spec
    if avg >= 0.85:
        print("\n  PASS — meets ≥85% spec target for feature learnability")
    elif avg >= 0.75:
        print("\n  OK — meets ≥75% feature transfer target but below 85% ideal")
    else:
        print("\n  FAIL — below 75% threshold")


if __name__ == "__main__":
    main()
