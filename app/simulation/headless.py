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


def train_worker_on_concept(worker, generator, concept_mapping, examples_per_class=None):
    """Train a worker on a user-defined concept, capped at worker.memory_cap.

    Fills memory round-robin across the source categories, distributing
    evenly across concept labels. Examples are spread across ground-truth
    categories (not just concept labels) so each concept sees variety.
    """
    cat_to_concept = {}
    all_items = []  # (category, concept_label)
    for concept_name, categories in concept_mapping.items():
        for cat in categories:
            cat_to_concept[cat] = concept_name
            all_items.append((cat, concept_name))
    # Round-robin teach until memory is full
    i = 0
    while not worker.is_memory_full and all_items:
        cat, label = all_items[i % len(all_items)]
        obj = generator.generate(cat)
        if not worker.teach(obj.tensor, label):
            break
        i += 1
    worker.category_mapping = cat_to_concept


def train_terminal_worker(worker, generator, categories, examples_per_class=None):
    """Train a terminal worker on specific categories, capped at worker.memory_cap.

    Fills memory round-robin across categories so each class gets an
    approximately equal share of the cap.
    """
    i = 0
    while not worker.is_memory_full and categories:
        cat = categories[i % len(categories)]
        obj = generator.generate(cat)
        if not worker.teach(obj.tensor, cat):
            break
        i += 1
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


# Full 19-shape vocabulary from the spec (5 starter + 5 + 4 + 5 across 3 packs)
STARTER = ["circle", "triangle", "square", "star_5", "heart"]
PACK1 = ["arrow", "crescent", "cloud", "lightning", "teardrop"]
PACK2_HOLES = ["donut", "picture_frame", "key", "gear"]
PACK3_MULTI = ["mushroom", "tree", "flower", "candy_cane", "rainbow"]
CATS_19 = STARTER + PACK1 + PACK2_HOLES + PACK3_MULTI
QUEUE_CAP = 10


def make_world(gen, start_objects=1):
    world = FactoryWorld(gen)
    world.economy.coins = 10000
    world.active_categories = list(CATS_19)
    world._remaining_categories = []
    world.objects_per_tick = start_objects
    return world


# ---------------------------------------------------------------------------
# Scenario builders (shared setup, just returns the world)
# ---------------------------------------------------------------------------

def build_generalist(gen, start_objects=1):
    """One worker that handles ALL 19 classes."""
    world = make_world(gen, start_objects)
    w = world.hire_worker("Generalist", binary=False)
    train_terminal_worker(w, gen, CATS_19, examples_per_class=6)
    world.graph.add_node("clf", queue_capacity=QUEUE_CAP)
    world.graph.set_root("clf")
    world.assign_worker(w, "clf")
    for cat in CATS_19:
        world.graph.connect("clf", cat, f"BIN:{cat}")
    estimate_all(world, gen)
    return world


def build_deep_routing(gen, start_objects=1):
    """Routing tree for the 19-shape vocabulary using spec features.

    Tree:
        multicolor? → yes → 5-class terminal (Pack 3)
                     → no  → has-hole? → yes → 4-class terminal (Pack 2)
                                       → no  → curved? → yes → 5-class terminal (curvy)
                                                       → no  → 5-class terminal (straight)
    """
    world = make_world(gen, start_objects)

    # Build the four feature groups
    multi_cats = list(PACK3_MULTI)            # mushroom, tree, flower, candy_cane, rainbow
    hole_cats = list(PACK2_HOLES)             # donut, picture_frame, key, gear
    curvy_cats = ["circle", "heart", "crescent", "cloud", "teardrop"]
    straight_cats = ["triangle", "square", "star_5", "arrow", "lightning"]

    # Binary routing workers
    root = world.hire_worker("Root: Multicolor?", binary=True)
    sub_hole = world.hire_worker("Sub: Has-hole?", binary=True)
    sub_curve = world.hire_worker("Sub: Curved?", binary=True)

    # Terminal classifiers (one per leaf group)
    term_multi = world.hire_worker("Term: Multicolor", binary=False)
    term_hole = world.hire_worker("Term: Has-hole", binary=False)
    term_curvy = world.hire_worker("Term: Curvy", binary=False)
    term_straight = world.hire_worker("Term: Straight", binary=False)

    # Train binary workers on their concept splits
    rest_after_multi = hole_cats + curvy_cats + straight_cats
    train_worker_on_concept(
        root, gen,
        {"multi": multi_cats, "mono": rest_after_multi}, 12)

    rest_after_hole = curvy_cats + straight_cats
    train_worker_on_concept(
        sub_hole, gen,
        {"hole": hole_cats, "solid": rest_after_hole}, 12)

    train_worker_on_concept(
        sub_curve, gen,
        {"curvy": curvy_cats, "straight": straight_cats}, 12)

    # Train terminal workers
    train_terminal_worker(term_multi, gen, multi_cats, examples_per_class=8)
    train_terminal_worker(term_hole, gen, hole_cats, examples_per_class=8)
    train_terminal_worker(term_curvy, gen, curvy_cats, examples_per_class=8)
    train_terminal_worker(term_straight, gen, straight_cats, examples_per_class=8)

    # Wire the graph
    nodes = [
        ("root", root),
        ("sub_hole", sub_hole),
        ("sub_curve", sub_curve),
        ("t_multi", term_multi),
        ("t_hole", term_hole),
        ("t_curvy", term_curvy),
        ("t_straight", term_straight),
    ]
    for nid, w in nodes:
        world.graph.add_node(nid, queue_capacity=QUEUE_CAP)
        world.assign_worker(w, nid)
    world.graph.set_root("root")

    # Routing edges
    world.graph.connect("root", "multi", "t_multi")
    world.graph.connect("root", "mono", "sub_hole")
    world.graph.connect("sub_hole", "hole", "t_hole")
    world.graph.connect("sub_hole", "solid", "sub_curve")
    world.graph.connect("sub_curve", "curvy", "t_curvy")
    world.graph.connect("sub_curve", "straight", "t_straight")

    # Terminal → bin edges (one bin per ground-truth category)
    for cat in multi_cats:
        world.graph.connect("t_multi", cat, f"BIN:{cat}")
    for cat in hole_cats:
        world.graph.connect("t_hole", cat, f"BIN:{cat}")
    for cat in curvy_cats:
        world.graph.connect("t_curvy", cat, f"BIN:{cat}")
    for cat in straight_cats:
        world.graph.connect("t_straight", cat, f"BIN:{cat}")

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
    # Pin seeds so model comparisons are reproducible
    import torch
    random.seed(42)
    torch.manual_seed(42)

    print("BabyBrain Factory — Balance Validation (Real ML Inference)")
    print("=" * 65)
    print(f"Categories ({len(CATS_19)}): {CATS_19}")
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
