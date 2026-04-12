"""Evaluate all snapshot checkpoints and report per-feature accuracy.

Runs eval_features on every file matching a pattern so we can pick
the true best checkpoint (avoiding noisy single-batch metrics).

Usage:
    .venv/bin/python meta_training/eval_all_snapshots.py \
        --hidden 64 --pattern general_conv4_64_robust
"""

import argparse
import glob
import os
import sys

import learn2learn as l2l
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.conv4 import Conv4WithHead  # noqa: E402
from meta_training.procedural_meta import EPISODE_FNS, FEATURE_NAMES  # noqa: E402

CKPT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "app", "models", "checkpoints"
)


def eval_feature(maml, episode_fn, n_episodes, inner_steps,
                 n_support, n_query, device):
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
        for _ in range(inner_steps):
            preds = learner(s_img)
            loss = F.cross_entropy(preds, s_lab)
            learner.adapt(loss)

        learner.train(False)
        with torch.no_grad():
            q_preds = learner(q_img)
            correct += (q_preds.argmax(dim=1) == q_lab).sum().item()
            total += q_lab.numel()
    return correct / total if total > 0 else 0.0


def eval_checkpoint(ckpt_path, hidden, n_episodes, inner_steps,
                    n_support, n_query, device):
    model = Conv4WithHead(num_classes=2, hidden=hidden).to(device)
    maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    maml.load_state_dict(state, strict=True)

    results = {}
    for name, fn in zip(FEATURE_NAMES, EPISODE_FNS):
        acc = eval_feature(maml, fn, n_episodes, inner_steps,
                           n_support, n_query, device)
        results[name] = acc
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--pattern", type=str,
                        default="general_conv4_64_robust",
                        help="Checkpoint filename prefix (finds both iter####.pt snapshots and the main .pt)")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per feature per checkpoint")
    parser.add_argument("--inner-steps", type=int, default=8,
                        help="Inner adaptation steps at eval time")
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
    print(f"Hidden: {args.hidden}")
    print(f"Episodes per feature: {args.episodes}  |  Inner steps: {args.inner_steps}")
    print()

    # Collect checkpoints
    snapshots = sorted(glob.glob(
        os.path.join(CKPT_DIR, f"{args.pattern}_iter*.pt")))
    final = os.path.join(CKPT_DIR, f"{args.pattern}.pt")
    ckpts = snapshots[:]
    if os.path.exists(final):
        ckpts.append(final)

    if not ckpts:
        print(f"No checkpoints found matching {args.pattern}*.pt in {CKPT_DIR}")
        sys.exit(1)

    print(f"Found {len(ckpts)} checkpoints")

    # Evaluate each
    all_results = {}
    for ckpt in ckpts:
        label = os.path.basename(ckpt).replace(args.pattern, "").replace(".pt", "")
        if label.startswith("_"):
            label = label[1:]
        if not label:
            label = "final"
        print(f"\nEvaluating {label}...")
        results = eval_checkpoint(
            ckpt, args.hidden, args.episodes, args.inner_steps,
            args.support, args.query, device)
        all_results[label] = results
        avg = sum(results.values()) / len(results)
        print(f"  avg: {avg * 100:5.1f}%")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Checkpoint':<12}", end="")
    for name in FEATURE_NAMES:
        short = name.replace("_vs_", "/").split("/")[0][:8]
        print(f"{short:>10}", end="")
    print(f"{'AVG':>8}")
    print("-" * 80)

    for label, results in all_results.items():
        print(f"{label:<12}", end="")
        for name in FEATURE_NAMES:
            print(f"{results[name] * 100:>9.1f}%", end="")
        avg = sum(results.values()) / len(results)
        print(f"{avg * 100:>7.1f}%")

    # Find winner
    best_label = max(all_results, key=lambda k: sum(all_results[k].values()))
    best_avg = sum(all_results[best_label].values()) / 8
    print("=" * 80)
    print(f"\n  WINNER: {best_label}  (avg {best_avg * 100:.1f}%)")

    # Also report worst feature of winner
    worst_name = min(all_results[best_label], key=all_results[best_label].get)
    worst_acc = all_results[best_label][worst_name]
    print(f"  Weakest feature of winner: {worst_name} at {worst_acc * 100:.1f}%")


if __name__ == "__main__":
    main()
