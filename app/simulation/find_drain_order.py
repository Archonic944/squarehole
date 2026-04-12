"""Greedy search for an optimal cascade drain order.

For each stage, train a candidate filter for every undrained shape and
pick the one with the highest STAGE SCORE: target recall × other recall.
This minimises the compound loss from false-positive draining of
later-stage shapes.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from app.factory.objects import ObjectGenerator
from app.factory.worker import FactoryWorker
from app.factory.world import resolve_worker_checkpoint

STARTER = ["circle", "triangle", "square", "star_5", "heart"]
SILHOUETTES = ["arrow", "crescent", "cloud", "lightning", "teardrop"]
HOLES = ["donut", "picture_frame", "key", "gear"]
MULTI = ["mushroom", "tree", "flower", "candy_cane", "rainbow"]
ALL_19 = STARTER + SILHOUETTES + HOLES + MULTI

PACK_OF: dict[str, str] = {}
for c in STARTER: PACK_OF[c] = "st"
for c in SILHOUETTES: PACK_OF[c] = "si"
for c in HOLES: PACK_OF[c] = "ho"
for c in MULTI: PACK_OF[c] = "mu"


def diverse_others(target: str, residue: list[str]) -> list[str]:
    pool = [c for c in residue if c != target]
    seen: set[str] = set()
    chosen: list[str] = []
    for c in pool:
        if PACK_OF[c] not in seen:
            chosen.append(c)
            seen.add(PACK_OF[c])
        if len(chosen) >= 4:
            break
    for c in pool:
        if c not in chosen and len(chosen) < 5:
            chosen.append(c)
    return chosen


def score_filter(target: str, residue: list[str], gen, samples=20) -> dict:
    """Train a filter for `target` and measure stage performance.

    Returns dict with:
        target_recall: % of target shapes correctly accepted
        other_recall:  % of non-target shapes correctly rejected
        stage_score:   net coins per object processed if used here
    """
    w = FactoryWorker(target, resolve_worker_checkpoint(), num_classes=2, device="cpu")
    for _ in range(5):
        w.teach(gen.generate(target).tensor, target, force=True)
    others = diverse_others(target, residue)
    for c in others[:5]:
        w.teach(gen.generate(c).tensor, "other", force=True)

    target_total = 0
    target_correct = 0
    other_total = 0
    other_correct = 0
    other_to_target_per: dict[str, int] = {}
    for c in residue:
        for _ in range(samples):
            obj = gen.generate(c)
            pred, _ = w.predict_real(obj.tensor)
            if c == target:
                target_total += 1
                if pred == target:
                    target_correct += 1
            else:
                other_total += 1
                if pred == "other":
                    other_correct += 1
                else:
                    other_to_target_per[c] = other_to_target_per.get(c, 0) + 1

    tr = target_correct / max(1, target_total)
    or_ = other_correct / max(1, other_total)

    # Stage net coins per object processed:
    # - target shape arrival rate ≈ 1/len(residue)
    # - filter pays +15 per (true target accepted), -8 per (non-target accepted, FP)
    # - rejected target = silent loss (will be misclassified later or by terminal)
    n_residue = len(residue)
    p_target = 1.0 / n_residue
    p_other = 1 - p_target
    fp_rate = 1 - or_
    # Net per object: gain if correctly accepted, cost if FP-accepted
    # Rejected target ↦ goes downstream — modelled as 0 here (best case)
    net = p_target * tr * 15 - p_other * fp_rate * 8

    return {
        "target": target,
        "target_recall": tr,
        "other_recall": or_,
        "stage_score": net,
        "top_fp": dict(sorted(other_to_target_per.items(), key=lambda x: -x[1])[:3]),
    }


def main():
    random.seed(42)
    torch.manual_seed(42)

    gen = ObjectGenerator(difficulty=0.0)

    residue = list(ALL_19)
    drain_order: list[str] = []

    print("=" * 78)
    print(" GREEDY DRAIN ORDER SEARCH")
    print("=" * 78)
    print(" At each stage, pick the filter with the highest stage_score")
    print(" (expected coins per object given current residue stream)")
    print()

    while len(residue) > 3:
        results = []
        for shape in residue:
            r = score_filter(shape, residue, gen)
            results.append(r)

        # Sort by stage_score
        results.sort(key=lambda x: -x["stage_score"])

        # Drop filters that lose money (stage_score < 0)
        viable = [r for r in results if r["stage_score"] > 0]
        if not viable:
            print(f"\n  STOP: no viable filters left for residue {residue}")
            break

        best = viable[0]
        drain_order.append(best["target"])
        residue.remove(best["target"])

        # Print top 3 candidates for transparency
        stage = len(drain_order)
        print(f"  Stage {stage:2d} (residue {len(residue)+1:2d} → {len(residue):2d}): "
              f"chose {best['target']:<14}  "
              f"tgt_recall={best['target_recall']*100:5.1f}%  "
              f"other_recall={best['other_recall']*100:5.1f}%  "
              f"score={best['stage_score']:+5.2f}")

    print()
    print(f"Final drain order ({len(drain_order)}): {drain_order}")
    print(f"Final residue ({len(residue)}): {residue}")
    return drain_order, residue


if __name__ == "__main__":
    main()
