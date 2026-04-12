"""Probe script: measure which binary splits + terminal groupings work well.

The goal is to discover, empirically, which routing decisions the backbone
can actually make accurately under the 10-example memory cap. We try a
catalog of candidate splits and terminal groups and print accuracy.
"""

import os
import sys
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from app.factory.objects import ObjectGenerator
from app.factory.worker import FactoryWorker
from app.factory.world import resolve_worker_checkpoint


def make_worker(name: str, num_classes: int) -> FactoryWorker:
    return FactoryWorker(
        name=name,
        checkpoint_path=resolve_worker_checkpoint(),
        num_classes=num_classes,
        device="cpu",
    )


def fill_round_robin(worker, gen, mapping):
    """Teach worker examples round-robin across (category, label) pairs.

    `mapping` is dict[label] -> list[category].
    """
    items = []
    for label, cats in mapping.items():
        for cat in cats:
            items.append((cat, label))
    i = 0
    while not worker.is_memory_full and items:
        cat, label = items[i % len(items)]
        obj = gen.generate(cat)
        if not worker.teach(obj.tensor, label):
            break
        i += 1
    worker.category_mapping = {cat: label for label, cats in mapping.items() for cat in cats}


def measure_binary(name, mapping, gen, samples_per_cat=40):
    w = make_worker(name, num_classes=len(mapping))
    fill_round_robin(w, gen, mapping)
    test = []
    for label, cats in mapping.items():
        for cat in cats:
            for _ in range(samples_per_cat):
                test.append(gen.generate(cat))
    acc = w.estimate_accuracy(test)
    return acc


def measure_terminal(name, cats, gen, samples_per_cat=40):
    w = make_worker(name, num_classes=len(cats))
    items = list(cats)
    i = 0
    while not w.is_memory_full:
        cat = items[i % len(items)]
        obj = gen.generate(cat)
        if not w.teach(obj.tensor, cat):
            break
        i += 1
    w.category_mapping = {c: c for c in cats}
    test = []
    for cat in cats:
        for _ in range(samples_per_cat):
            test.append(gen.generate(cat))
    acc = w.estimate_accuracy(test)
    return acc


def main():
    random.seed(42)
    torch.manual_seed(42)

    gen = ObjectGenerator(difficulty=0.0)

    print("=" * 70)
    print("BINARY SPLIT ACCURACY PROBES")
    print("=" * 70)

    # Candidate binary splits — want each at >= 90% to be useful
    splits = [
        ("multi vs mono",
         {"multi": ["mushroom", "tree", "flower", "candy_cane", "rainbow"],
          "mono": ["circle", "triangle", "square", "star_5", "heart"]}),
        ("multi vs all-mono(14)",
         {"multi": ["mushroom", "tree", "flower", "candy_cane", "rainbow"],
          "mono": ["circle", "triangle", "square", "star_5", "heart",
                   "arrow", "crescent", "cloud", "lightning", "teardrop",
                   "donut", "picture_frame", "key", "gear"]}),
        ("hole vs solid",
         {"hole": ["donut", "picture_frame", "key", "gear"],
          "solid": ["circle", "triangle", "square", "star_5", "heart",
                    "arrow", "crescent", "cloud", "lightning", "teardrop"]}),
        ("curved vs straight",
         {"curved": ["circle", "heart", "crescent", "cloud", "teardrop"],
          "straight": ["triangle", "square", "star_5", "arrow", "lightning"]}),
        # Smaller-grain splits — fewer shapes per side
        ("circle+heart vs square+triangle",
         {"a": ["circle", "heart"], "b": ["square", "triangle"]}),
        ("star vs square",
         {"a": ["star_5"], "b": ["square"]}),
        ("crescent vs cloud",
         {"a": ["crescent"], "b": ["cloud"]}),
        ("teardrop vs heart",
         {"a": ["teardrop"], "b": ["heart"]}),
        ("arrow vs lightning",
         {"a": ["arrow"], "b": ["lightning"]}),
        ("donut vs picture_frame",
         {"a": ["donut"], "b": ["picture_frame"]}),
        ("key vs gear",
         {"a": ["key"], "b": ["gear"]}),
        ("mushroom vs tree",
         {"a": ["mushroom"], "b": ["tree"]}),
        ("flower vs rainbow",
         {"a": ["flower"], "b": ["rainbow"]}),
        ("candy_cane vs flower",
         {"a": ["candy_cane"], "b": ["flower"]}),
        # Group splits within a pack
        ("(circle,heart) vs (square,triangle,star_5)",
         {"a": ["circle", "heart"], "b": ["square", "triangle", "star_5"]}),
        ("(arrow,lightning) vs (crescent,cloud,teardrop)",
         {"a": ["arrow", "lightning"], "b": ["crescent", "cloud", "teardrop"]}),
        ("(donut,picture_frame) vs (key,gear)",
         {"a": ["donut", "picture_frame"], "b": ["key", "gear"]}),
        ("(mushroom,tree) vs (flower,candy_cane,rainbow)",
         {"a": ["mushroom", "tree"], "b": ["flower", "candy_cane", "rainbow"]}),
    ]
    for name, mapping in splits:
        acc = measure_binary(name, mapping, gen)
        flag = "OK" if acc >= 0.90 else ("..." if acc >= 0.75 else "BAD")
        print(f"  [{flag}] {acc*100:5.1f}%  {name}")

    print()
    print("=" * 70)
    print("TERMINAL CLASSIFIER ACCURACY (N classes, memory cap 10)")
    print("=" * 70)

    print()
    print("=" * 70)
    print("N-WAY ROUTER PROBES (treat routing as classification)")
    print("=" * 70)
    # Idea: instead of binary 'is multi?' use a 4-class router whose
    # support set has each class = a few representative members of one pack.
    nway_routers = [
        ("4-pack router (2 reps each)",
         {"p_starter": ["circle", "heart"],
          "p_sil":     ["arrow", "cloud"],
          "p_hole":    ["donut", "key"],
          "p_multi":   ["mushroom", "rainbow"]}),
        ("4-pack router (1 rep each + filler)",
         {"p_starter": ["star_5"],
          "p_sil":     ["lightning"],
          "p_hole":    ["gear"],
          "p_multi":   ["candy_cane"]}),
        ("4-pack router 3-rep",
         {"p_starter": ["circle", "square", "heart"],
          "p_sil":     ["arrow", "cloud", "teardrop"],
          "p_hole":    ["donut", "key", "gear"],
          "p_multi":   ["mushroom", "rainbow", "flower"]}),
        # Variant: only route between 2 packs at a time, fewer classes
        ("2-pack router starter vs sil (3 reps)",
         {"starter": ["circle", "heart", "square"],
          "silhouettes": ["arrow", "cloud", "lightning"]}),
        ("2-pack router hole vs multi (3 reps)",
         {"hole": ["donut", "key", "gear"],
          "multi": ["mushroom", "rainbow", "flower"]}),
        ("2-pack router starter+sil vs hole+multi (mixed reps)",
         {"a": ["circle", "arrow", "cloud", "square", "heart"],
          "b": ["donut", "key", "gear", "mushroom", "rainbow"]}),
    ]
    # For the n-way router, we test recall on all categories in the pack
    # (not just the support categories) — this measures whether the
    # router generalizes to unseen pack members.
    pack_members = {
        "p_starter": ["circle", "triangle", "square", "star_5", "heart"],
        "p_sil":     ["arrow", "crescent", "cloud", "lightning", "teardrop"],
        "p_hole":    ["donut", "picture_frame", "key", "gear"],
        "p_multi":   ["mushroom", "tree", "flower", "candy_cane", "rainbow"],
        "starter":   ["circle", "triangle", "square", "star_5", "heart"],
        "silhouettes":["arrow", "crescent", "cloud", "lightning", "teardrop"],
        "hole":      ["donut", "picture_frame", "key", "gear"],
        "multi":     ["mushroom", "tree", "flower", "candy_cane", "rainbow"],
        "a":         ["circle", "triangle", "square", "star_5", "heart",
                      "arrow", "crescent", "cloud", "lightning", "teardrop"],
        "b":         ["donut", "picture_frame", "key", "gear",
                      "mushroom", "tree", "flower", "candy_cane", "rainbow"],
    }
    for name, support_map in nway_routers:
        # Build worker
        w = make_worker(name, num_classes=len(support_map))
        items = []
        for label, cats in support_map.items():
            for cat in cats:
                items.append((cat, label))
        i = 0
        while not w.is_memory_full and items:
            cat, label = items[i % len(items)]
            obj = gen.generate(cat)
            if not w.teach(obj.tensor, label):
                break
            i += 1
        # Build full mapping covering ALL pack members for this router's labels
        full_map = {}
        for label in support_map.keys():
            for cat in pack_members[label]:
                full_map[cat] = label
        w.category_mapping = full_map
        # Test on every category covered, sampling 30 each
        test = []
        for cat in full_map.keys():
            for _ in range(30):
                test.append(gen.generate(cat))
        acc = w.estimate_accuracy(test)
        flag = "OK" if acc >= 0.90 else ("..." if acc >= 0.75 else "BAD")
        print(f"  [{flag}] {acc*100:5.1f}%  {name}")

    print()
    print("=" * 70)
    print("GENERALIZATION PROBES — does router handle unseen rejects?")
    print("=" * 70)
    # Train on 2 specific classes, test on the trained classes PLUS
    # other shapes routed by ground-truth pack assignment.
    # We measure: when an untrained shape arrives, where does it go?
    def test_generalization(name, support_map, ground_truth_map):
        w = make_worker(name, num_classes=len(support_map))
        items = []
        for label, cats in support_map.items():
            for cat in cats:
                items.append((cat, label))
        i = 0
        while not w.is_memory_full and items:
            cat, label = items[i % len(items)]
            obj = gen.generate(cat)
            if not w.teach(obj.tensor, label):
                break
            i += 1
        w.category_mapping = ground_truth_map
        test = []
        for cat in ground_truth_map.keys():
            for _ in range(30):
                test.append(gen.generate(cat))
        acc = w.estimate_accuracy(test)
        flag = "OK" if acc >= 0.90 else ("..." if acc >= 0.75 else "BAD")
        print(f"  [{flag}] {acc*100:5.1f}%  {name}")
        return acc

    # Hypothesis: train on 2 multi vs 2 mono — does it generalize to others?
    test_generalization(
        "trained mushroom/rainbow vs square/star — test ALL",
        {"multi": ["mushroom", "rainbow"], "mono": ["square", "star_5"]},
        {**{c: "multi" for c in ["mushroom", "tree", "flower", "candy_cane", "rainbow"]},
         **{c: "mono" for c in ["circle", "triangle", "square", "star_5", "heart"]}})

    test_generalization(
        "trained 1 from each pack as 4-way",
        {"p_starter": ["heart"], "p_sil": ["lightning"], "p_hole": ["gear"], "p_multi": ["candy_cane"]},
        {**{c: "p_starter" for c in ["circle", "triangle", "square", "star_5", "heart"]},
         **{c: "p_sil" for c in ["arrow", "crescent", "cloud", "lightning", "teardrop"]},
         **{c: "p_hole" for c in ["donut", "picture_frame", "key", "gear"]},
         **{c: "p_multi" for c in ["mushroom", "tree", "flower", "candy_cane", "rainbow"]}})

    # Inside one pack: train on subset, test on full pack
    test_generalization(
        "multi pack: trained 2/5, test all 5",
        {"a": ["mushroom"], "b": ["rainbow"]},
        {"mushroom": "a", "tree": "a", "flower": "b", "candy_cane": "b", "rainbow": "b"})

    # 6-class terminal with 5 pack + 'other' — does 'other' work as escape?
    print("  -- 6-class terminal w/ 'other' escape --")
    def test_with_other(pack_name, in_cats, out_cats):
        w = make_worker(f"{pack_name}+other", num_classes=len(in_cats) + 1)
        # support: 5 in-pack (1 each) + 5 out-pack (round robin)
        ex_cap = w.memory_cap
        # In-pack: 1 example per class
        for cat in in_cats:
            obj = gen.generate(cat)
            w.teach(obj.tensor, cat)
        # Other: fill remaining cap with mix from out
        while not w.is_memory_full:
            cat = out_cats[w.get_support_set_size() % len(out_cats)]
            obj = gen.generate(cat)
            w.teach(obj.tensor, "other")
        # Build mapping: in-cats map to themselves, all out cats map to "other"
        full_map = {c: c for c in in_cats}
        for c in out_cats:
            full_map[c] = "other"
        w.category_mapping = full_map
        test = []
        for c in full_map:
            for _ in range(30):
                test.append(gen.generate(c))
        acc = w.estimate_accuracy(test)
        # Also: how often does an in-pack shape get the wrong specific bin
        # vs how often it gets routed to "other"?
        in_correct = in_to_other = 0
        out_to_other = out_to_in = 0
        for c in in_cats:
            for _ in range(20):
                pred, _ = w.predict_real(gen.generate(c).tensor)
                if pred == c:
                    in_correct += 1
                elif pred == "other":
                    in_to_other += 1
        for c in out_cats:
            for _ in range(10):
                pred, _ = w.predict_real(gen.generate(c).tensor)
                if pred == "other":
                    out_to_other += 1
                else:
                    out_to_in += 1
        print(f"    [{pack_name}] overall {acc*100:.0f}%  in→correct {in_correct}, in→other {in_to_other}, "
              f"out→other {out_to_other}, out→in-bin {out_to_in}")

    test_with_other("multi", ["mushroom", "tree", "flower", "candy_cane", "rainbow"],
                    ["circle", "square", "donut", "arrow", "crescent", "key"])
    test_with_other("hole", ["donut", "picture_frame", "key", "gear"],
                    ["circle", "square", "mushroom", "arrow", "crescent", "rainbow"])

    print()
    print("=" * 70)
    print("TERMINAL CLASSIFIER ACCURACY (N classes, memory cap 10)")
    print("=" * 70)

    terminals = [
        # Whole packs (the ones we hadn't tested)
        ("starter pack (5)", ["circle", "triangle", "square", "star_5", "heart"]),
        ("silhouettes pack (5)", ["arrow", "crescent", "cloud", "lightning", "teardrop"]),
        # Pairs (5 examples each)
        ("circle, heart", ["circle", "heart"]),
        ("square, triangle", ["square", "triangle"]),
        ("arrow, lightning", ["arrow", "lightning"]),
        ("crescent, cloud", ["crescent", "cloud"]),
        ("crescent, teardrop", ["crescent", "teardrop"]),
        ("cloud, teardrop", ["cloud", "teardrop"]),
        ("donut, picture_frame", ["donut", "picture_frame"]),
        ("key, gear", ["key", "gear"]),
        ("mushroom, tree", ["mushroom", "tree"]),
        ("flower, candy_cane", ["flower", "candy_cane"]),
        ("rainbow, candy_cane", ["rainbow", "candy_cane"]),
        # Triples (3 each)
        ("circle, heart, square", ["circle", "heart", "square"]),
        ("square, triangle, star_5", ["square", "triangle", "star_5"]),
        ("crescent, cloud, teardrop", ["crescent", "cloud", "teardrop"]),
        ("arrow, lightning, star_5", ["arrow", "lightning", "star_5"]),
        ("flower, candy_cane, rainbow", ["flower", "candy_cane", "rainbow"]),
        # Quads
        ("donut, picture_frame, key, gear", ["donut", "picture_frame", "key", "gear"]),
        # Five
        ("multicolor pack (5)", ["mushroom", "tree", "flower", "candy_cane", "rainbow"]),
    ]
    for name, cats in terminals:
        acc = measure_terminal(name, cats, gen)
        flag = "OK" if acc >= 0.90 else ("..." if acc >= 0.75 else "BAD")
        print(f"  [{flag}] {acc*100:5.1f}%  {len(cats)}-class: {name}")


if __name__ == "__main__":
    main()
