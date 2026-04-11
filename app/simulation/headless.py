"""Headless simulation to validate factory game balance.

Tests two things:
1. At scale (5/tick), good factory design >> bad design
2. With throughput ramp (1→8/tick), generalist breaks down and routing becomes necessary

All routing concepts are user-defined — the simulation teaches them
programmatically the way a player would.
"""

import os
import sys
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.factory.objects import ObjectGenerator, SHAPE_FAMILIES
from app.factory.worker import FactoryWorker
from app.factory.routing import RoutingGraph
from app.factory.economy import Economy
from app.factory.world import FactoryWorld


def train_worker_on_concept(worker, generator, concept_mapping, examples_per_class=10):
    """Train a worker on a user-defined concept with balanced sampling."""
    cat_to_concept = {}
    for concept_name, categories in concept_mapping.items():
        for cat in categories:
            cat_to_concept[cat] = concept_name
        per_cat = max(2, examples_per_class // len(categories))
        for cat in categories:
            for _ in range(per_cat):
                obj = generator.generate(cat)
                worker.teach(obj.tensor, concept_name)
    worker.category_mapping = cat_to_concept


def train_terminal_worker(worker, generator, categories, examples_per_class=10):
    """Train a terminal worker on specific categories."""
    for cat in categories:
        for _ in range(examples_per_class):
            obj = generator.generate(cat)
            worker.teach(obj.tensor, cat)
    worker.category_mapping = {cat: cat for cat in categories}


def estimate_all(world, generator, test_per_class=50):
    for worker in world.workers:
        cats = list(set(worker.category_mapping.keys())) if worker.category_mapping else list(worker.class_names)
        if cats:
            test = generator.generate_balanced_batch(test_per_class, cats)
            worker.estimate_accuracy(test)


def print_report(name, world, elapsed, ticks):
    stats = world.get_stats()
    total_objects = sum(1 for _ in range(ticks))  # approximate
    correct_count = int(stats["total_earned"] / world.economy.CORRECT_REWARD)
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Ticks: {ticks}  |  Time: {elapsed:.1f}s  |  Final obj/tick: {stats['objects_per_tick']}")
    print(f"  Coins: {stats['coins']:.0f}  |  Earned: {stats['total_earned']:.0f}  |  Lost: {stats['total_penalties']:.0f}  |  Spent: {stats['total_spent']:.0f}")
    print(f"  Coins/tick: {stats['coins_per_tick']:+.2f}")
    print()
    for w in stats["workers"]:
        real_acc = w["correct"] / w["processed"] * 100 if w["processed"] > 0 else 0
        worker_obj = next((wo for wo in world.workers if wo.name == w["name"]), None)
        speed = worker_obj.natural_speed if worker_obj else "?"
        n_classes = len(worker_obj.class_names) if worker_obj else "?"
        print(f"    {w['name']:<28} {n_classes}-class spd={speed}  est={w['accuracy']*100:4.0f}%  actual={real_acc:4.0f}%  n={w['processed']}")
    print(f"{'='*65}")
    return stats


CATS_8 = ["circle", "triangle", "star_5", "square", "diamond", "oval", "cross", "heart"]
QUEUE_CAP = 10


def make_world(gen, start_objects=1):
    world = FactoryWorld(gen)
    world.economy.coins = 10000
    world.active_categories = list(CATS_8)
    world._remaining_categories = []
    world.objects_per_tick = start_objects
    return world


# ---------------------------------------------------------------------------
# Scenario builders (shared setup, just returns the world)
# ---------------------------------------------------------------------------

def build_generalist(gen, start_objects=1):
    world = make_world(gen, start_objects)
    w = world.hire_worker("Generalist", binary=False)
    train_terminal_worker(w, gen, CATS_8, examples_per_class=8)
    world.graph.add_node("clf", processing_speed=w.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.set_root("clf")
    world.assign_worker(w, "clf")
    for cat in CATS_8:
        world.graph.connect("clf", cat, f"BIN:{cat}")
    estimate_all(world, gen)
    return world


def build_deep_routing(gen, start_objects=1):
    world = make_world(gen, start_objects)
    round_cats = ["circle", "oval", "heart"]
    angular_cats = ["triangle", "star_5", "square", "diamond", "cross"]
    spikey = ["star_5", "diamond"]
    blocky = ["triangle", "square", "cross"]

    root = world.hire_worker("Root: Round?", binary=True)
    sub = world.hire_worker("Sub: Spikey?", binary=True)
    sr = world.hire_worker("Spec: Round", binary=False)
    ss = world.hire_worker("Spec: Spikey", binary=False)
    sb = world.hire_worker("Spec: Blocky", binary=False)

    train_worker_on_concept(root, gen, {"round": round_cats, "angular": angular_cats}, 15)
    train_worker_on_concept(sub, gen, {"spikey": spikey, "blocky": blocky}, 15)
    train_terminal_worker(sr, gen, round_cats, 10)
    train_terminal_worker(ss, gen, spikey, 10)
    train_terminal_worker(sb, gen, blocky, 10)

    for nid, w in [("root", root), ("sub", sub), ("sr", sr), ("ss", ss), ("sb", sb)]:
        world.graph.add_node(nid, processing_speed=w.natural_speed, queue_capacity=QUEUE_CAP)
        world.assign_worker(w, nid)
    world.graph.set_root("root")

    world.graph.connect("root", "round", "sr")
    world.graph.connect("root", "angular", "sub")
    world.graph.connect("sub", "spikey", "ss")
    world.graph.connect("sub", "blocky", "sb")
    for cat in round_cats:
        world.graph.connect("sr", cat, f"BIN:{cat}")
    for cat in spikey:
        world.graph.connect("ss", cat, f"BIN:{cat}")
    for cat in blocky:
        world.graph.connect("sb", cat, f"BIN:{cat}")

    estimate_all(world, gen)
    return world


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_at_scale(gen):
    """Test at high throughput (5/tick fixed) — does routing still beat generalist?"""
    print("\n" + "=" * 65)
    print("  TEST 1: Fixed 5 objects/tick (routing advantage from throughput)")
    print("=" * 65)

    ticks = 500
    t0 = time.time()
    w1 = build_generalist(gen, start_objects=5)
    w1.THROUGHPUT_RAMP_INTERVAL = 999999  # disable ramp
    w1.run_ticks(ticks)
    s1 = print_report("Generalist @ 5/tick", w1, time.time() - t0, ticks)

    t0 = time.time()
    w2 = build_deep_routing(gen, start_objects=5)
    w2.THROUGHPUT_RAMP_INTERVAL = 999999
    w2.run_ticks(ticks)
    s2 = print_report("Deep Routing @ 5/tick", w2, time.time() - t0, ticks)

    ratio = s2["coins_per_tick"] / max(0.01, s1["coins_per_tick"])
    print(f"\n  Routing is {ratio:.1f}x more profitable at 5/tick")
    return ratio


def test_with_ramp(gen):
    """Test with throughput ramp (1→8/tick) — does routing become necessary?"""
    print("\n" + "=" * 65)
    print("  TEST 2: Throughput ramp 1→8/tick (simulates actual gameplay)")
    print("=" * 65)

    ticks = 800  # enough for ramp: starts at 1, +1 every 100 ticks → reaches 8 by tick 700

    t0 = time.time()
    w1 = build_generalist(gen, start_objects=1)
    w1.run_ticks(ticks)
    s1 = print_report("Generalist (with ramp)", w1, time.time() - t0, ticks)

    t0 = time.time()
    w2 = build_deep_routing(gen, start_objects=1)
    w2.run_ticks(ticks)
    s2 = print_report("Deep Routing (with ramp)", w2, time.time() - t0, ticks)

    ratio = s2["coins_per_tick"] / max(0.01, s1["coins_per_tick"])
    print(f"\n  Routing is {ratio:.1f}x more profitable over the full ramp")

    # Show how the advantage grows: compare last 100 ticks
    # (when objects_per_tick is at max)
    print(f"  Final objects/tick: generalist={w1.objects_per_tick}, routing={w2.objects_per_tick}")
    return ratio


def test_early_game(gen):
    """Test that at 1/tick, generalist is viable (not punishing new players)."""
    print("\n" + "=" * 65)
    print("  TEST 3: Early game (1 object/tick) — is generalist viable?")
    print("=" * 65)

    ticks = 100

    t0 = time.time()
    w1 = build_generalist(gen, start_objects=1)
    w1.THROUGHPUT_RAMP_INTERVAL = 999999  # no ramp
    w1.run_ticks(ticks)
    s1 = print_report("Generalist @ 1/tick", w1, time.time() - t0, ticks)

    viable = s1["coins_per_tick"] > 0
    print(f"\n  Generalist viable at 1/tick: {'YES' if viable else 'NO'} ({s1['coins_per_tick']:+.2f} $/tick)")
    return viable


def main():
    print("BabyBrain Factory — Balance Validation (Real ML Inference)")
    print("=" * 65)
    print(f"Categories: {CATS_8}")
    print(f"Speed formula: max(1, 6 - num_classes)")
    print(f"Throughput ramp: +1 object/tick every 100 ticks (1 → 8)")
    print(f"Economy: +15 correct, -3 wrong, -1 drop, -0.3 upkeep/worker")
    print()

    gen = ObjectGenerator(difficulty=0.0)

    # Test 1: Does routing still work at scale?
    r1 = test_at_scale(gen)

    # Test 2: Does the ramp make routing necessary?
    r2 = test_with_ramp(gen)

    # Test 3: Is early game viable?
    r3 = test_early_game(gen)

    # Summary
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  At-scale routing advantage:  {r1:.1f}x {'PASS' if r1 > 2 else 'FAIL'}")
    print(f"  Ramp routing advantage:      {r2:.1f}x {'PASS' if r2 > 1.5 else 'FAIL'}")
    print(f"  Early game viable:           {'PASS' if r3 else 'FAIL'}")

    all_pass = r1 > 2 and r2 > 1.5 and r3
    print(f"\n  {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED — needs tuning'}")
    print()


if __name__ == "__main__":
    main()
