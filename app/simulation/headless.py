"""Headless simulation to validate factory game balance.

Proves that factory design drastically affects throughput by testing
with 8 categories where a single generalist can't keep up but a
well-designed routing network thrives.

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
    """Train a worker on a user-defined concept.

    concept_mapping: {"concept_name": ["category1", "category2", ...], ...}

    Uses balanced sampling: each source category gets at least
    examples_per_class // len(categories) examples, ensuring the model
    sees every category that maps to each concept.
    """
    cat_to_concept = {}
    for concept_name, categories in concept_mapping.items():
        for cat in categories:
            cat_to_concept[cat] = concept_name
        # Balanced: each source category gets a fair share
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
    """Estimate accuracy for all workers with a large test set for stability."""
    for worker in world.workers:
        cats = list(set(worker.category_mapping.keys())) if worker.category_mapping else list(worker.class_names)
        if cats:
            test = generator.generate_balanced_batch(test_per_class, cats)
            worker.estimate_accuracy(test)


def print_report(name, world, elapsed, ticks):
    stats = world.get_stats()
    total_objects = ticks * world.objects_per_tick
    correct_count = int(stats["total_earned"] / world.economy.CORRECT_REWARD)
    wrong_and_dropped = total_objects - correct_count
    overall_acc = correct_count / total_objects * 100 if total_objects > 0 else 0

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    print(f"  Ticks: {ticks}  |  Objects: {total_objects}  |  Correct bins: {correct_count} ({overall_acc:.0f}%)")
    print(f"  Coins: {stats['coins']:.0f}  |  Earned: {stats['total_earned']:.0f}  |  Lost: {stats['total_penalties']:.0f}  |  Spent: {stats['total_spent']:.0f}")
    print(f"  Coins/tick: {stats['coins_per_tick']:+.2f}")
    print()

    # Show worker speeds
    for w in stats["workers"]:
        real_acc = w["correct"] / w["processed"] * 100 if w["processed"] > 0 else 0
        worker_obj = next((wo for wo in world.workers if wo.name == w["name"]), None)
        speed = worker_obj.natural_speed if worker_obj else "?"
        n_classes = len(worker_obj.class_names) if worker_obj else "?"
        print(f"    {w['name']:<28} {n_classes}-class spd={speed}  est={w['accuracy']*100:4.0f}%  actual={real_acc:4.0f}%  n={w['processed']}")
    print(f"{'='*65}")
    return stats


# -- Scenarios (all use 8 categories) ----------------------------------------

CATS_8 = ["circle", "triangle", "star_5", "square", "diamond", "oval", "cross", "heart"]

OBJECTS_PER_TICK = 5
QUEUE_CAP = 10


def make_world(gen):
    world = FactoryWorld(gen)
    world.economy.coins = 10000
    world.active_categories = list(CATS_8)
    world._remaining_categories = []
    world.objects_per_tick = OBJECTS_PER_TICK
    return world


def scenario_baseline(gen, ticks=500):
    """No workers — everything drops."""
    world = make_world(gen)
    world.graph.add_node("empty", queue_capacity=QUEUE_CAP)
    world.graph.set_root("empty")
    t0 = time.time()
    world.run_ticks(ticks)
    return print_report("BASELINE: No Workers", world, time.time() - t0, ticks)


def scenario_generalist(gen, ticks=500):
    """Single worker doing 8-way classification. Bottlenecked on throughput."""
    world = make_world(gen)

    w = world.hire_worker("Generalist", binary=False)
    train_terminal_worker(w, gen, CATS_8, examples_per_class=8)

    world.graph.add_node("clf", processing_speed=w.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.set_root("clf")
    world.assign_worker(w, "clf")
    for cat in CATS_8:
        world.graph.connect("clf", cat, f"BIN:{cat}")

    estimate_all(world, gen)
    t0 = time.time()
    world.run_ticks(ticks)
    return print_report("BAD: Single Generalist (throughput bottleneck)", world, time.time() - t0, ticks)


def scenario_simple_routing(gen, ticks=500):
    """Binary router + 2 specialists. Better throughput distribution."""
    world = make_world(gen)

    round_cats = ["circle", "oval", "heart"]
    angular_cats = ["triangle", "star_5", "square", "diamond", "cross"]

    router = world.hire_worker("Router: Round?", binary=True)
    spec_round = world.hire_worker("Round Specialist", binary=False)
    spec_angular = world.hire_worker("Angular Specialist", binary=False)

    train_worker_on_concept(router, gen, {
        "round": round_cats, "angular": angular_cats,
    }, examples_per_class=15)
    train_terminal_worker(spec_round, gen, round_cats, examples_per_class=10)
    train_terminal_worker(spec_angular, gen, angular_cats, examples_per_class=10)

    world.graph.add_node("router", processing_speed=router.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.add_node("round_spec", processing_speed=spec_round.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.add_node("angular_spec", processing_speed=spec_angular.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.set_root("router")
    world.assign_worker(router, "router")
    world.assign_worker(spec_round, "round_spec")
    world.assign_worker(spec_angular, "angular_spec")

    world.graph.connect("router", "round", "round_spec")
    world.graph.connect("router", "angular", "angular_spec")
    for cat in round_cats:
        world.graph.connect("round_spec", cat, f"BIN:{cat}")
    for cat in angular_cats:
        world.graph.connect("angular_spec", cat, f"BIN:{cat}")

    estimate_all(world, gen)
    t0 = time.time()
    world.run_ticks(ticks)
    return print_report("MEDIUM: Router + 2 Specialists", world, time.time() - t0, ticks)


def scenario_deep_routing(gen, ticks=500):
    """Deep routing with 4 terminal specialists. Maximum throughput."""
    world = make_world(gen)

    round_cats = ["circle", "oval", "heart"]
    angular_cats = ["triangle", "star_5", "square", "diamond", "cross"]
    spikey = ["star_5", "diamond"]
    blocky = ["triangle", "square", "cross"]

    root = world.hire_worker("Root: Round?", binary=True)
    sub_angular = world.hire_worker("Sub: Spikey?", binary=True)
    spec_round = world.hire_worker("Spec: Round", binary=False)
    spec_spikey = world.hire_worker("Spec: Spikey", binary=False)
    spec_blocky = world.hire_worker("Spec: Blocky", binary=False)

    train_worker_on_concept(root, gen, {"round": round_cats, "angular": angular_cats}, 15)
    train_worker_on_concept(sub_angular, gen, {"spikey": spikey, "blocky": blocky}, 15)
    train_terminal_worker(spec_round, gen, round_cats, 10)
    train_terminal_worker(spec_spikey, gen, spikey, 10)
    train_terminal_worker(spec_blocky, gen, blocky, 10)

    world.graph.add_node("root", processing_speed=root.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.add_node("sub_angular", processing_speed=sub_angular.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.add_node("spec_round", processing_speed=spec_round.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.add_node("spec_spikey", processing_speed=spec_spikey.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.add_node("spec_blocky", processing_speed=spec_blocky.natural_speed, queue_capacity=QUEUE_CAP)
    world.graph.set_root("root")

    world.assign_worker(root, "root")
    world.assign_worker(sub_angular, "sub_angular")
    world.assign_worker(spec_round, "spec_round")
    world.assign_worker(spec_spikey, "spec_spikey")
    world.assign_worker(spec_blocky, "spec_blocky")

    world.graph.connect("root", "round", "spec_round")
    world.graph.connect("root", "angular", "sub_angular")
    world.graph.connect("sub_angular", "spikey", "spec_spikey")
    world.graph.connect("sub_angular", "blocky", "spec_blocky")

    for cat in round_cats:
        world.graph.connect("spec_round", cat, f"BIN:{cat}")
    for cat in spikey:
        world.graph.connect("spec_spikey", cat, f"BIN:{cat}")
    for cat in blocky:
        world.graph.connect("spec_blocky", cat, f"BIN:{cat}")

    estimate_all(world, gen)
    t0 = time.time()
    world.run_ticks(ticks)
    return print_report("GOOD: Deep Routing (3 parallel specialists)", world, time.time() - t0, ticks)


def main():
    print("BabyBrain Factory — Headless Balance Test")
    print("=" * 65)
    print(f"8 categories: {CATS_8}")
    print(f"{OBJECTS_PER_TICK} objects/tick, speed = max(1, 6-num_classes), 500 ticks = {OBJECTS_PER_TICK*500} objects")
    print(f"  Binary (2-class) → speed 4 | 3-class → 3 | 8-class → 1")
    print("Training with REAL MAML inference...\n")

    gen = ObjectGenerator(difficulty=0.0)

    results = {}
    results["baseline"] = scenario_baseline(gen)
    results["bad"] = scenario_generalist(gen)
    results["medium"] = scenario_simple_routing(gen)
    results["good"] = scenario_deep_routing(gen)

    # Summary
    print(f"\n{'='*65}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Scenario':<40} {'$/tick':>8} {'Final $':>10}")
    print(f"  {'-'*58}")
    for label, stats in results.items():
        cpt = stats["coins_per_tick"]
        coins = stats["coins"]
        if cpt > 0:
            bar = "+" * min(40, int(cpt * 1.5))
        else:
            bar = "-" * min(20, int(abs(cpt) * 2))
        print(f"  {label:<40} {cpt:>+8.2f} {coins:>10.0f}  {bar}")

    bad_cpt = results["bad"]["coins_per_tick"]
    good_cpt = results["good"]["coins_per_tick"]
    med_cpt = results["medium"]["coins_per_tick"]

    print()
    if good_cpt > bad_cpt * 1.5 and good_cpt > med_cpt:
        print("  PASS: Good design is significantly more profitable!")
    elif good_cpt > bad_cpt:
        print("  PARTIAL: Good > bad, but the difference could be larger.")
    else:
        print("  FAIL: Routing isn't outperforming the generalist.")
        print("  This means the ML accuracy for routing concepts needs improvement,")
        print("  or the economic model needs adjustment.")

    print()


if __name__ == "__main__":
    main()
