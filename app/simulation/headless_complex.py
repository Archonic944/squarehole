"""Headless validation of a deep, sophisticated routing graph.

Goal: build a routing tree that profitably handles ALL 19 shapes from
the 4 contracts (Starter, Tricky Silhouettes, Holes & Cutouts, Multicolor)
under realistic gameplay conditions (full 8 obj/tick throughput, sufficient
processing speed, enough queue capacity for the routing depth).

Architecture (chosen empirically from probes):

  Long instance-discrimination cascade. The Conv4 backbone was meta-trained
  on 1-vs-1 instance discrimination, so binary "is this specifically X?"
  filters are the model's strongest decision. Concept-level routing
  ("multicolor?", "has hole?") fails at ~30-50% accuracy. Pack-level
  routing also fails. So we drain shapes one-by-one with shape-specific
  filters in a carefully-ordered cascade.

  Filter ordering matters: each filter sees the residue stream of all
  shapes that survived earlier stages. Filters with FP partners (e.g.
  heart confuses with cloud/lightning/key) must be drained BEFORE those
  partners are drained themselves. Probe runs revealed which order
  preserves >85% FP-rejection at each stage.
"""

import os
import sys
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from app.factory.objects import ObjectGenerator
from app.factory.worker import FactoryWorker
from app.factory.world import FactoryWorld

# Full vocabulary across the 4 contracts (19 shapes)
STARTER = ["circle", "triangle", "square", "star_5", "heart"]
SILHOUETTES = ["arrow", "crescent", "cloud", "lightning", "teardrop"]
HOLES = ["donut", "picture_frame", "key", "gear"]
MULTI = ["mushroom", "tree", "flower", "candy_cane", "rainbow"]
ALL_19 = STARTER + SILHOUETTES + HOLES + MULTI

QUEUE_CAP = 50  # generous so depth doesn't drop objects under burst load

# ---------------------------------------------------------------------------
# Worker training helpers
# ---------------------------------------------------------------------------


def teach_filter(worker: FactoryWorker, gen, target: str, others_pool: list[str]):
    """Train a balanced 2-class 'is target?' filter (5 target + 5 mixed others)."""
    for _ in range(5):
        worker.teach(gen.generate(target).tensor, target, force=True)
    for c in others_pool[:5]:
        worker.teach(gen.generate(c).tensor, "other", force=True)
    full = {target: target}
    for c in ALL_19:
        if c != target:
            full[c] = "other"
    worker.category_mapping = full


def teach_terminal(worker: FactoryWorker, gen, cats: list[str]):
    """Train an N-class terminal classifier round-robin within memory cap."""
    i = 0
    while not worker.is_memory_full:
        cat = cats[i % len(cats)]
        if not worker.teach(gen.generate(cat).tensor, cat):
            break
        i += 1
    worker.category_mapping = {c: c for c in cats}


def make_world(gen, start_objects=8, speed_level=8) -> FactoryWorld:
    """Set up a world with all 19 categories active and adequate resources."""
    world = FactoryWorld(gen)
    world.economy.coins = 100_000
    for cid in ("silhouettes", "holes", "multicolor"):
        world.accept_contract(cid)
    assert sorted(world.active_categories) == sorted(ALL_19)
    world.objects_per_tick = start_objects
    world.speed_level = speed_level
    world._remaining_categories = []
    return world


def estimate_all(world: FactoryWorld, gen, samples=40):
    """Compute cached_accuracy on each worker against its declared categories."""
    for w in world.workers:
        cats = list(set(w.category_mapping.keys())) if w.category_mapping else list(w.class_names)
        if not cats:
            continue
        test = []
        for c in cats:
            for _ in range(samples):
                test.append(gen.generate(c))
        w.estimate_accuracy(test)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _diverse_others(target: str, residue: list[str]) -> list[str]:
    """Pick 5 diverse 'other' shapes (one per pack first, then fill).

    Diversity is what works empirically — confuser-pool training inverts
    the model's bias and makes filters bin everything as the target.
    """
    pack_of: dict[str, str] = {}
    for c in STARTER: pack_of[c] = "st"
    for c in SILHOUETTES: pack_of[c] = "si"
    for c in HOLES: pack_of[c] = "ho"
    for c in MULTI: pack_of[c] = "mu"

    pool = [c for c in residue if c != target]
    seen: set[str] = set()
    chosen: list[str] = []
    for c in pool:
        if pack_of[c] not in seen:
            chosen.append(c)
            seen.add(pack_of[c])
        if len(chosen) >= 4:
            break
    for c in pool:
        if c not in chosen and len(chosen) < 5:
            chosen.append(c)
    return chosen


def build_cascade(gen) -> tuple[FactoryWorld, list[str]]:
    """Long instance-discrimination cascade with hand-tuned filter training.

    Drain order is chosen so each filter has favourable residue at its stage.
    The 'other' support set for each filter is its top visual confusers
    (from probes) so the model learns those negatives explicitly.

    Shapes that probe poorly as filters (target recall < 60%) are NOT
    drained by the cascade — they fall through to the residue terminal
    classifier at the end.
    """
    world = make_world(gen, start_objects=8, speed_level=8)

    # Drain order — empirically tuned via probe runs.
    #
    # Phase 1 (easy targets, varied residue): candy_cane→cloud
    #   These shapes have strong, distinctive features and perform well
    #   when the residue stream is broad. tree/mushroom/cloud must be
    #   late enough that their FP partners (within multi/silhouettes)
    #   are already drained.
    #
    # Phase 2 (hole pack, late stage): donut→picture_frame
    #   The hole-pack shapes confuse with each other heavily. Drain
    #   them all back-to-back at the end so the residue is dominated by
    #   simple-geometry leftovers — each filter then runs with very few
    #   FP partners around.
    #
    # Final residue (3 shapes — circle, square, crescent) goes to a
    # 3-class terminal classifier. The model genuinely can't separate
    # these at the 10-example cap (probes show ~60%), so we accept that
    # cost as the floor.
    # Drain order chosen empirically by iterating headless tests.
    # Phase 1 — easy first wins (perfect features at any stage):
    #     candy_cane → rainbow → star_5 → lightning
    # Phase 2 — mid-cascade shapes whose FP partners are still around:
    #     heart → triangle → arrow → teardrop
    # Phase 3 — late drains: tree/mushroom/cloud/flower work cleanly
    #     once their visual confusers (in earlier packs) are gone.
    # Phase 4 — hole pack drained back-to-back at the very end so they
    #     disambiguate from each other one shape at a time.
    drain_order = [
        "candy_cane", "rainbow", "star_5", "heart", "lightning", "triangle",
        "arrow", "teardrop", "mushroom", "tree", "cloud", "flower",
        "donut", "key", "gear", "picture_frame",
    ]
    leftover = [c for c in ALL_19 if c not in drain_order]
    # leftover = ["circle", "square", "crescent"]

    # ----- Hire workers -----
    filters: list[tuple[str, FactoryWorker]] = []
    residue = list(ALL_19)
    for shape in drain_order:
        others = _diverse_others(shape, residue)
        w = world.hire_worker(f"Filter: {shape}?", binary=True)
        teach_filter(w, gen, shape, others)
        filters.append((shape, w))
        residue.remove(shape)

    # ----- Build cascade chain -----
    prev_node = None
    for shape, w in filters:
        node_id = f"R_{shape}"
        world.graph.add_node(node_id, queue_capacity=QUEUE_CAP)
        world.assign_worker(w, node_id)
        if prev_node is None:
            world.graph.set_root(node_id)
        else:
            world.graph.connect(prev_node, "other", node_id)
        world.graph.connect(node_id, shape, f"BIN:{shape}")
        prev_node = node_id

    # ----- Final stage: small terminal for the leftover residue -----
    final = world.hire_worker(f"Term: residue-{len(leftover)}", binary=False)
    teach_terminal(final, gen, leftover)

    world.graph.add_node("T_residue", queue_capacity=QUEUE_CAP)
    world.assign_worker(final, "T_residue")
    world.graph.connect(prev_node, "other", "T_residue")
    for c in leftover:
        world.graph.connect("T_residue", c, f"BIN:{c}")

    estimate_all(world, gen)
    return world, drain_order


def print_report(name, world: FactoryWorld, elapsed: float, ticks: int,
                 per_pack_correct: dict | None = None,
                 per_pack_total: dict | None = None):
    stats = world.get_stats()
    print(f"\n{'='*78}")
    print(f"  {name}")
    print(f"{'='*78}")
    print(f"  Ticks: {ticks}  |  Time: {elapsed:.1f}s  |  obj/tick: {stats['objects_per_tick']}  |  speed_lvl: {stats['speed_level']}")
    print(f"  Coins: {stats['coins']:.0f}  |  Earned: {stats['total_earned']:.0f}  |  Lost: {stats['total_penalties']:.0f}  |  Spent: {stats['total_spent']:.0f}")
    print(f"  Coins/tick: {stats['coins_per_tick']:+.2f}")

    if per_pack_correct is not None:
        print()
        print("  Per-contract accuracy (end-to-end through routing graph):")
        for pack_name, pack_cats in (
            ("Starter Pack",         STARTER),
            ("Tricky Silhouettes",   SILHOUETTES),
            ("Holes & Cutouts",      HOLES),
            ("Multicolor",           MULTI),
        ):
            tot = sum(per_pack_total.get(c, 0) for c in pack_cats)
            cor = sum(per_pack_correct.get(c, 0) for c in pack_cats)
            pct = (cor / tot * 100) if tot else 0
            print(f"    {pack_name:<22} {cor}/{tot:<5} ({pct:5.1f}%)")
            for c in pack_cats:
                ct = per_pack_total.get(c, 0)
                cc = per_pack_correct.get(c, 0)
                p = (cc / ct * 100) if ct else 0
                print(f"        {c:<18} {cc}/{ct:<4} ({p:.0f}%)")

    print()
    for w in stats["workers"]:
        real_acc = w["correct"] / w["processed"] * 100 if w["processed"] > 0 else 0
        worker_obj = next((wo for wo in world.workers if wo.name == w["name"]), None)
        n_classes = len(worker_obj.class_names) if worker_obj else "?"
        print(f"    {w['name']:<38} {n_classes}c  est={w['accuracy']*100:4.0f}%  actual={real_acc:4.0f}%  n={w['processed']}")
    print(f"{'='*78}")
    return stats


def run_with_per_pack_tracking(
    world: FactoryWorld,
    ticks: int,
    ramp_every: int | None = None,
    ramp_max: int = 8,
):
    """Run ticks while tracking which categories end up in which bins.

    If *ramp_every* is given, ``objects_per_tick`` is incremented by 1
    every *ramp_every* ticks (capped at *ramp_max*), simulating the
    original throughput ramp. Returns (per_pack_correct, per_pack_total).
    """
    correct: dict[str, int] = {c: 0 for c in ALL_19}
    total:   dict[str, int] = {c: 0 for c in ALL_19}
    for t in range(ticks):
        if (
            ramp_every
            and t > 0
            and t % ramp_every == 0
            and world.objects_per_tick < ramp_max
        ):
            world.objects_per_tick += 1
        results = world.tick()
        for obj, bin_name in results.correct:
            correct[obj.category] += 1
            total[obj.category] += 1
        for obj, predicted_bin, true_cat in results.wrong:
            total[true_cat] += 1
        for obj in results.dropped:
            total[obj.category] += 1
    return correct, total


def main():
    random.seed(42)
    torch.manual_seed(42)

    print("BabyBrain Factory — DEEP ROUTING VALIDATION (cascade)")
    print("=" * 78)
    print(f"Categories ({len(ALL_19)}): {ALL_19}")
    print(f"Contracts: Starter, Tricky Silhouettes, Holes & Cutouts, Multicolor")
    print()

    gen = ObjectGenerator(difficulty=0.0)

    # ----- Test 1: full-throughput burn-in -----
    print("Building cascade routing graph...")
    t0 = time.time()
    world, drain_order = build_cascade(gen)
    print(f"Built in {time.time()-t0:.1f}s. {len(world.workers)} workers, "
          f"{len(world.graph.nodes)} nodes.")
    print(f"Drain order ({len(drain_order)}): {drain_order}")
    print(f"Final residue: {[c for c in ALL_19 if c not in drain_order]}")

    ticks = 500
    print(f"\nRunning {ticks} ticks at {world.objects_per_tick} obj/tick, speed={world.speed_level}...")
    t0 = time.time()
    correct, total = run_with_per_pack_tracking(world, ticks)
    elapsed = time.time() - t0

    print_report("Deep Routing v3 (cascade) — full throughput",
                 world, elapsed, ticks,
                 per_pack_correct=correct, per_pack_total=total)

    coins_per_tick = world.economy.coins_per_tick
    overall_correct = sum(correct.values())
    overall_total = sum(total.values())
    overall_pct = overall_correct / overall_total * 100 if overall_total else 0
    print(f"\n  Overall accuracy: {overall_correct}/{overall_total} ({overall_pct:.1f}%)")
    print(f"  Profitable: {'YES' if coins_per_tick > 0 else 'NO'} ({coins_per_tick:+.2f} $/tick)")

    # ----- Test 2: throughput ramp (realistic gameplay) -----
    print()
    print("=" * 78)
    print("  Test 2 — manual throughput ramp 1 → 8 obj/tick")
    print("=" * 78)
    random.seed(42)
    torch.manual_seed(42)
    world2, _ = build_cascade(gen)
    world2.objects_per_tick = 1
    ramp_ticks = 800
    print(f"Running {ramp_ticks} ticks, +1 obj/tick every 100 ticks (max 8)...")
    t0 = time.time()
    correct2, total2 = run_with_per_pack_tracking(
        world2, ramp_ticks, ramp_every=100, ramp_max=8
    )
    elapsed = time.time() - t0
    print_report("Deep Routing v3 — with ramp", world2, elapsed, ramp_ticks,
                 per_pack_correct=correct2, per_pack_total=total2)
    overall = sum(correct2.values()) / max(1, sum(total2.values())) * 100
    print(f"\n  Overall accuracy under ramp: {overall:.1f}%")
    print(f"  Profitable under ramp: {'YES' if world2.economy.coins_per_tick > 0 else 'NO'} "
          f"({world2.economy.coins_per_tick:+.2f} $/tick)")

    success = coins_per_tick > 0 and world2.economy.coins_per_tick > 0
    print()
    print("=" * 78)
    print(f"  RESULT: {'ALL TESTS PASS' if success else 'FAILED'}")
    print(f"  Workers: {len(world.workers)}  |  Nodes: {len(world.graph.nodes)}")
    print(f"  Cascade depth: {len(drain_order)} stages + 3-class residue terminal")
    print("=" * 78)
    return success


if __name__ == "__main__":
    main()
