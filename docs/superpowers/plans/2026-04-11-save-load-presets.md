# Save/Load System & Factory Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a save/load system for the factory game state and create three ready-to-play presets (simple split, two-level tree, stress test) with pre-taught MAML workers.

**Architecture:** `save_load.py` serializes the full world state (graph topology, workers with support sets, economy) into a dict and uses `torch.save`/`torch.load` for persistence. `presets.py` defines factory functions that programmatically build a `FactoryWorld`, hire workers, teach them with procedurally generated objects, wire up the DAG routing graph with edges terminating at `BIN:` targets, and return a ready-to-play world. The main entry point gets a `--preset` CLI flag.

**Tech Stack:** Python 3.11, PyTorch (torch.save/load), existing FactoryWorld/RoutingGraph/FactoryWorker APIs.

---

## File Structure

- **Create:** `app/factory/save_load.py` — `save_game(world, path)` and `load_game(path, device) -> FactoryWorld` functions
- **Create:** `app/factory/presets.py` — `simple_split(device)`, `two_level_tree(device)`, `stress_test(device)` preset factories
- **Modify:** `app/main.py` — add `--preset` CLI argument to load a preset on startup

---

### Task 1: Save/Load Module

**Files:**
- Create: `app/factory/save_load.py`

- [ ] **Step 1: Create `save_game` function**

Write the serialization function that extracts all game state into a plain dict and saves it with `torch.save`. The key challenge is mapping worker references in graph nodes to indices into the `world.workers` list.

```python
"""Save/load system for factory game state."""

from __future__ import annotations

import os
from collections import defaultdict

import torch

from .objects import ObjectGenerator
from .economy import Economy
from .routing import RoutingGraph, RoutingNode, RoutingEdge
from .worker import FactoryWorker
from .world import FactoryWorld, GENERAL_CHECKPOINT, WHATS_CHECKPOINT


def save_game(world: FactoryWorld, path: str) -> None:
    """Serialize the full game state to *path* using torch.save."""
    # Build worker index for cross-referencing from graph nodes
    worker_to_idx: dict[int, int] = {id(w): i for i, w in enumerate(world.workers)}

    # Serialize workers
    workers_data = []
    for w in world.workers:
        support = {}
        for cls_name, tensors in w._support_set.items():
            support[cls_name] = [t.cpu() for t in tensors]
        workers_data.append({
            "name": w.name,
            "role": w.role,
            "class_names": list(w.class_names),
            "num_classes": w.num_classes,
            "cached_accuracy": w.cached_accuracy,
            "category_mapping": dict(w.category_mapping),
            "support_set": support,
            "stats": {
                "total_processed": w.stats.total_processed,
                "total_correct": w.stats.total_correct,
                "coins_earned": w.stats.coins_earned,
            },
        })

    # Serialize graph nodes
    nodes_data = {}
    for nid, node in world.graph.nodes.items():
        worker_idx = None
        if node.worker is not None:
            worker_idx = worker_to_idx.get(id(node.worker))
        nodes_data[nid] = {
            "node_id": nid,
            "worker_index": worker_idx,
            "queue_capacity": node.queue_capacity,
            "edges": [
                {"output_label": e.output_label, "target": e.target}
                for e in node.edges
            ],
        }

    data = {
        "version": 1,
        "world": {
            "tick_count": world.tick_count,
            "objects_per_tick": world.objects_per_tick,
            "speed_level": world.speed_level,
            "active_categories": list(world.active_categories),
            "remaining_categories": list(world._remaining_categories),
        },
        "economy": {
            "coins": world.economy.coins,
            "total_earned": world.economy.total_earned,
            "total_spent": world.economy.total_spent,
            "total_penalties": world.economy.total_penalties,
        },
        "graph": {
            "root_id": world.graph.root_id,
            "nodes": nodes_data,
        },
        "workers": workers_data,
        "generator": {
            "difficulty": world.object_generator.difficulty,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(data, path)
```

- [ ] **Step 2: Create `load_game` function**

Write the deserialization function. It reconstructs the `ObjectGenerator`, `FactoryWorld`, `Economy`, workers (with full support sets via `teach()`), and the routing graph with edges. Workers are reconstructed by creating fresh instances, then calling `teach()` for each support set example so the model head gets built and MAML adaptation is ready.

Add this to the same file:

```python
def load_game(path: str, device: str = "cpu") -> FactoryWorld:
    """Load a saved game from *path* and return a ready-to-play FactoryWorld."""
    data = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct generator
    gen = ObjectGenerator(difficulty=data["generator"]["difficulty"])

    # Reconstruct world
    world = FactoryWorld(gen)
    wd = data["world"]
    world.tick_count = wd["tick_count"]
    world.objects_per_tick = wd["objects_per_tick"]
    world.speed_level = wd["speed_level"]
    world.active_categories = list(wd["active_categories"])
    world._remaining_categories = list(wd["remaining_categories"])

    # Reconstruct economy
    ed = data["economy"]
    world.economy.coins = ed["coins"]
    world.economy.total_earned = ed["total_earned"]
    world.economy.total_spent = ed["total_spent"]
    world.economy.total_penalties = ed["total_penalties"]

    # Reconstruct workers
    checkpoint = GENERAL_CHECKPOINT if os.path.exists(GENERAL_CHECKPOINT) else WHATS_CHECKPOINT
    workers: list[FactoryWorker] = []
    for wd_worker in data["workers"]:
        num_classes = max(wd_worker["num_classes"], 2)
        w = FactoryWorker(
            name=wd_worker["name"],
            checkpoint_path=checkpoint,
            num_classes=num_classes,
            device=device,
        )
        # Teach all support set examples to rebuild head and prepare adaptation
        for cls_name in wd_worker["class_names"]:
            tensors = wd_worker["support_set"].get(cls_name, [])
            for t in tensors:
                w.teach(t.to(device), cls_name)
        w.role = wd_worker["role"]
        w.cached_accuracy = wd_worker["cached_accuracy"]
        w.category_mapping = dict(wd_worker["category_mapping"])
        w.stats.total_processed = wd_worker["stats"]["total_processed"]
        w.stats.total_correct = wd_worker["stats"]["total_correct"]
        w.stats.coins_earned = wd_worker["stats"]["coins_earned"]
        workers.append(w)

    world.workers = workers

    # Reconstruct graph
    gd = data["graph"]
    world.graph = RoutingGraph()
    for nid, nd in gd["nodes"].items():
        worker = None
        if nd["worker_index"] is not None:
            worker = workers[nd["worker_index"]]
        node = world.graph.add_node(
            nid, worker=worker, queue_capacity=nd["queue_capacity"]
        )
        for ed_edge in nd["edges"]:
            node.edges.append(
                RoutingEdge(
                    output_label=ed_edge["output_label"],
                    target=ed_edge["target"],
                )
            )

    if gd["root_id"] is not None:
        world.graph.root_id = gd["root_id"]

    return world
```

- [ ] **Step 3: Verify save_load module imports cleanly**

Run: `cd /Users/gabriel/Development/PythonProjects/babybrain && .venv/bin/python -c "from app.factory.save_load import save_game, load_game; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add app/factory/save_load.py
git commit -m "feat: add save/load system for factory game state"
```

---

### Task 2: Presets Module

**Files:**
- Create: `app/factory/presets.py`

The presets module provides three factory functions. Each:
1. Creates an `ObjectGenerator` and `FactoryWorld`
2. Creates workers with `FactoryWorker` directly (bypassing `hire_worker` economy checks)
3. Teaches workers with procedurally generated objects (10 examples per class)
4. Builds a DAG routing graph with edges that all terminate at `BIN:` targets
5. Returns a ready-to-play `FactoryWorld`

The key helper is `_teach_worker` which generates balanced training examples and calls `worker.teach()` for each one, then estimates accuracy.

- [ ] **Step 1: Create presets module with helper and simple_split preset**

```python
"""Pre-built factory presets for quick testing."""

from __future__ import annotations

import os

from .objects import ObjectGenerator, SHAPE_FAMILIES
from .economy import Economy
from .routing import RoutingGraph
from .worker import FactoryWorker
from .world import FactoryWorld, GENERAL_CHECKPOINT, WHATS_CHECKPOINT

_EXAMPLES_PER_CLASS = 10


def _checkpoint() -> str:
    return GENERAL_CHECKPOINT if os.path.exists(GENERAL_CHECKPOINT) else WHATS_CHECKPOINT


def _make_worker(
    name: str,
    class_names: list[str],
    category_mapping: dict[str, str],
    gen: ObjectGenerator,
    device: str,
) -> FactoryWorker:
    """Create a worker, teach it with generated examples, and estimate accuracy."""
    w = FactoryWorker(
        name=name,
        checkpoint_path=_checkpoint(),
        num_classes=len(class_names),
        device=device,
    )
    # Teach with generated examples
    all_cats_for_worker = list(category_mapping.keys())
    for cat in all_cats_for_worker:
        label = category_mapping[cat]
        for _ in range(_EXAMPLES_PER_CLASS):
            obj = gen.generate(category=cat)
            w.teach(obj.tensor, label)
    w.category_mapping = dict(category_mapping)
    # Estimate accuracy on fresh test objects
    test_objs = []
    for cat in all_cats_for_worker:
        for _ in range(5):
            test_objs.append(gen.generate(category=cat))
    w.estimate_accuracy(test_objs)
    return w


def simple_split(device: str = "cpu") -> FactoryWorld:
    """A minimal 1-node routing DAG.

    Root worker splits objects into "rounded" vs "angular".
    Each output edge goes directly to a BIN.

    Categories: circle, oval, triangle, diamond (the 4 starting categories
    minus star_5/square, replaced with oval/diamond for a clean 2-way split).

    DAG structure:
        [root: round_vs_angular]
           |-- "rounded"  --> BIN:rounded
           |-- "angular"  --> BIN:angular
    """
    gen = ObjectGenerator(difficulty=0.0)
    world = FactoryWorld(gen)
    world.objects_per_tick = 2

    rounded_cats = ["circle", "oval"]
    angular_cats = ["triangle", "diamond"]
    all_cats = rounded_cats + angular_cats
    world.active_categories = list(all_cats)
    world._remaining_categories = [
        c for c in gen.ALL_CATEGORIES if c not in all_cats
    ]

    # Build category mapping: each shape maps to "rounded" or "angular"
    cat_map = {}
    for c in rounded_cats:
        cat_map[c] = "rounded"
    for c in angular_cats:
        cat_map[c] = "angular"

    root_worker = _make_worker(
        "Sorter", ["rounded", "angular"], cat_map, gen, device
    )
    world.workers.append(root_worker)

    # Build DAG
    world.graph.add_node("root", worker=root_worker)
    world.graph.set_root("root")
    world.graph.connect("root", "rounded", "BIN:rounded")
    world.graph.connect("root", "angular", "BIN:angular")

    # Give enough coins to play with
    world.economy.coins = 500.0

    return world


def two_level_tree(device: str = "cpu") -> FactoryWorld:
    """A two-level routing DAG with 8 categories.

    Root splits "pointy" vs "smooth". Second level splits each group
    into individual category bins.

    Categories: circle, oval, triangle, diamond, star_5, square, heart, cross

    DAG structure:
        [root: pointy_vs_smooth]
           |-- "pointy"  --> [node_pointy: which_pointy]
           |                    |-- "star"     --> BIN:star_5
           |                    |-- "triangle" --> BIN:triangle
           |                    |-- "diamond"  --> BIN:diamond
           |                    |-- "cross"    --> BIN:cross
           |-- "smooth"  --> [node_smooth: which_smooth]
                                |-- "circle"  --> BIN:circle
                                |-- "oval"    --> BIN:oval
                                |-- "heart"   --> BIN:heart
                                |-- "square"  --> BIN:square
    """
    gen = ObjectGenerator(difficulty=0.0)
    world = FactoryWorld(gen)
    world.objects_per_tick = 3

    pointy_cats = ["star_5", "triangle", "diamond", "cross"]
    smooth_cats = ["circle", "oval", "heart", "square"]
    all_cats = pointy_cats + smooth_cats
    world.active_categories = list(all_cats)
    world._remaining_categories = [
        c for c in gen.ALL_CATEGORIES if c not in all_cats
    ]

    # Root worker: pointy vs smooth
    root_map = {}
    for c in pointy_cats:
        root_map[c] = "pointy"
    for c in smooth_cats:
        root_map[c] = "smooth"
    root_worker = _make_worker(
        "Router", ["pointy", "smooth"], root_map, gen, device
    )
    world.workers.append(root_worker)

    # Pointy sub-router: identify individual pointy shapes
    pointy_labels = ["star", "triangle", "diamond", "cross"]
    pointy_map = dict(zip(pointy_cats, pointy_labels))
    pointy_worker = _make_worker(
        "Pointy Sorter", pointy_labels, pointy_map, gen, device
    )
    world.workers.append(pointy_worker)

    # Smooth sub-router: identify individual smooth shapes
    smooth_labels = ["circle", "oval", "heart", "square"]
    smooth_map = dict(zip(smooth_cats, smooth_labels))
    smooth_worker = _make_worker(
        "Smooth Sorter", smooth_labels, smooth_map, gen, device
    )
    world.workers.append(smooth_worker)

    # Build DAG
    world.graph.add_node("root", worker=root_worker)
    world.graph.set_root("root")
    world.graph.add_node("pointy", worker=pointy_worker)
    world.graph.add_node("smooth", worker=smooth_worker)

    # Root edges -> sub-routers
    world.graph.connect("root", "pointy", "pointy")
    world.graph.connect("root", "smooth", "smooth")

    # Pointy sub-router -> bins
    world.graph.connect("pointy", "star", "BIN:star_5")
    world.graph.connect("pointy", "triangle", "BIN:triangle")
    world.graph.connect("pointy", "diamond", "BIN:diamond")
    world.graph.connect("pointy", "cross", "BIN:cross")

    # Smooth sub-router -> bins
    world.graph.connect("smooth", "circle", "BIN:circle")
    world.graph.connect("smooth", "oval", "BIN:oval")
    world.graph.connect("smooth", "heart", "BIN:heart")
    world.graph.connect("smooth", "square", "BIN:square")

    world.economy.coins = 1000.0

    return world


def stress_test(device: str = "cpu") -> FactoryWorld:
    """A large DAG with 20+ nodes to stress-test the visual layout.

    Uses all 18 shape categories. Three-level deep routing tree:
    - Level 0: Root splits by shape family (pointy/angular/rounded/boxy/organic)
    - Level 1: Each family node splits into sub-groups of 2-3 categories
    - Level 2: Sub-group nodes route to individual BIN: targets

    DAG structure (simplified):
        [root: family_router]
           |-- "pointy"  --> [pointy_hub]
           |                    |-- "stars"    --> [stars_node]
           |                    |                    |-- "star_4" --> BIN:star_4
           |                    |                    |-- "star_5" --> BIN:star_5
           |                    |                    |-- "star_6" --> BIN:star_6
           |                    |-- "non_stars" --> [pointy_other]
           |                                         |-- "arrow"     --> BIN:arrow
           |                                         |-- "lightning" --> BIN:lightning
           |-- "angular" --> [angular_hub]
           |                    |-- "tri"  --> [tri_node]
           |                    |                |-- "triangle"       --> BIN:triangle
           |                    |                |-- "right_triangle" --> BIN:right_triangle
           |                    |-- "quad" --> [quad_node]
           |                                     |-- "diamond"       --> BIN:diamond
           |                                     |-- "parallelogram" --> BIN:parallelogram
           |                                     |-- "trapezoid"     --> BIN:trapezoid
           |-- "rounded" --> [rounded_hub]
           |                    |-- "circle"    --> BIN:circle
           |                    |-- "oval"      --> BIN:oval
           |                    |-- "semicircle"--> BIN:semicircle
           |-- "boxy"    --> [boxy_hub]
           |                    |-- "square"    --> BIN:square
           |                    |-- "rectangle" --> BIN:rectangle
           |                    |-- "cross"     --> BIN:cross
           |-- "organic" --> [organic_hub]
                                |-- "heart"    --> BIN:heart
                                |-- "crescent" --> BIN:crescent
    """
    gen = ObjectGenerator(difficulty=0.0)
    world = FactoryWorld(gen)
    world.objects_per_tick = 5
    world.active_categories = list(gen.ALL_CATEGORIES)
    world._remaining_categories = []

    families = SHAPE_FAMILIES  # {"pointy": [...], "angular": [...], ...}

    # --- Root worker: classify by family ---
    family_names = list(families.keys())  # pointy, angular, rounded, boxy, organic
    root_map: dict[str, str] = {}
    for fam, cats in families.items():
        for c in cats:
            root_map[c] = fam
    root_worker = _make_worker("Family Router", family_names, root_map, gen, device)
    world.workers.append(root_worker)

    # Build the root node
    world.graph.add_node("root", worker=root_worker)
    world.graph.set_root("root")

    # --- Per-family hubs and sub-nodes ---

    # Define sub-groups for families that are large enough to split further
    sub_groups: dict[str, dict[str, list[str]]] = {
        "pointy": {
            "stars": ["star_4", "star_5", "star_6"],
            "non_stars": ["arrow", "lightning"],
        },
        "angular": {
            "triangles": ["triangle", "right_triangle"],
            "quads": ["diamond", "parallelogram", "trapezoid"],
        },
        # rounded, boxy, organic go directly to bins (small enough)
    }

    for fam in family_names:
        fam_node_id = f"{fam}_hub"
        cats_in_family = families[fam]

        if fam in sub_groups:
            # This family has sub-groups: hub -> sub-group nodes -> bins
            groups = sub_groups[fam]
            group_names = list(groups.keys())

            # Hub worker: classify into sub-groups
            hub_map: dict[str, str] = {}
            for grp_name, grp_cats in groups.items():
                for c in grp_cats:
                    hub_map[c] = grp_name
            hub_worker = _make_worker(
                f"{fam.title()} Hub", group_names, hub_map, gen, device
            )
            world.workers.append(hub_worker)
            world.graph.add_node(fam_node_id, worker=hub_worker)
            world.graph.connect("root", fam, fam_node_id)

            # Sub-group nodes -> individual bins
            for grp_name, grp_cats in groups.items():
                grp_node_id = f"{fam}_{grp_name}"

                if len(grp_cats) == 1:
                    # Only one category: hub routes directly to bin
                    world.graph.connect(fam_node_id, grp_name, f"BIN:{grp_cats[0]}")
                else:
                    # Sub-group worker identifies individual categories
                    grp_map = {c: c for c in grp_cats}
                    grp_worker = _make_worker(
                        f"{grp_name.title().replace('_', ' ')} Sorter",
                        list(grp_cats), grp_map, gen, device,
                    )
                    world.workers.append(grp_worker)
                    world.graph.add_node(grp_node_id, worker=grp_worker)
                    world.graph.connect(fam_node_id, grp_name, grp_node_id)

                    # Each category -> bin
                    for c in grp_cats:
                        world.graph.connect(grp_node_id, c, f"BIN:{c}")
        else:
            # Small family: hub routes directly to bins
            if len(cats_in_family) == 1:
                # Single-category family: root routes directly to bin
                world.graph.connect("root", fam, f"BIN:{cats_in_family[0]}")
            else:
                hub_map = {c: c for c in cats_in_family}
                hub_worker = _make_worker(
                    f"{fam.title()} Sorter", list(cats_in_family),
                    hub_map, gen, device,
                )
                world.workers.append(hub_worker)
                world.graph.add_node(fam_node_id, worker=hub_worker)
                world.graph.connect("root", fam, fam_node_id)

                for c in cats_in_family:
                    world.graph.connect(fam_node_id, c, f"BIN:{c}")

    world.economy.coins = 5000.0

    return world
```

- [ ] **Step 2: Verify presets module imports and each preset builds without error**

Run: `cd /Users/gabriel/Development/PythonProjects/babybrain && .venv/bin/python -c "
from app.factory.presets import simple_split, two_level_tree, stress_test
w = simple_split(); print(f'simple_split: {len(w.graph.nodes)} nodes, {len(w.workers)} workers')
w = two_level_tree(); print(f'two_level_tree: {len(w.graph.nodes)} nodes, {len(w.workers)} workers')
w = stress_test(); print(f'stress_test: {len(w.graph.nodes)} nodes, {len(w.workers)} workers')
print('OK')
"`

Expected: Node/worker counts printed, `OK` at the end. `simple_split` should have 1 node, `two_level_tree` should have 3, `stress_test` should have 10+.

- [ ] **Step 3: Commit**

```bash
git add app/factory/presets.py
git commit -m "feat: add factory presets (simple_split, two_level_tree, stress_test)"
```

---

### Task 3: Integration — CLI Preset Flag & Save/Load Round-Trip

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Add --preset CLI argument to main.py**

Modify `app/main.py` to accept `--preset {simple_split,two_level_tree,stress_test}` and `--save`/`--load` arguments. When a preset is specified, use its factory function instead of creating a blank world. When `--load` is specified, load from the given path.

Replace the contents of `app/main.py`:

```python
"""BabyBrain Factory — entry point."""

import argparse
import sys
import pygame

from app.factory.objects import ObjectGenerator
from app.factory.world import FactoryWorld
from app.ui.factory_floor import FactoryFloorUI

WIDTH, HEIGHT = 1024, 768
FPS = 30


def _detect_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="BabyBrain Factory")
    parser.add_argument(
        "--preset",
        choices=["simple_split", "two_level_tree", "stress_test"],
        help="Load a pre-built factory preset instead of starting blank",
    )
    parser.add_argument(
        "--load",
        metavar="PATH",
        help="Load a saved game from the given file path",
    )
    args = parser.parse_args()

    device = _detect_device()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BabyBrain Factory")
    clock = pygame.time.Clock()

    if args.load:
        from app.factory.save_load import load_game
        world = load_game(args.load, device=device)
        gen = world.object_generator
    elif args.preset:
        from app.factory.presets import simple_split, two_level_tree, stress_test
        presets = {
            "simple_split": simple_split,
            "two_level_tree": two_level_tree,
            "stress_test": stress_test,
        }
        world = presets[args.preset](device=device)
        gen = world.object_generator
    else:
        gen = ObjectGenerator(difficulty=0.0)
        world = FactoryWorld(gen)
        world.objects_per_tick = 1

    ui = FactoryFloorUI(world, gen)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        events = pygame.event.get()

        for e in events:
            if e.type == pygame.QUIT:
                running = False

        ui.update(events, dt)
        result = ui.draw(screen)
        if result == "MENU":
            running = False

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test save/load round-trip**

Run: `cd /Users/gabriel/Development/PythonProjects/babybrain && .venv/bin/python -c "
from app.factory.presets import two_level_tree
from app.factory.save_load import save_game, load_game

# Create a preset world
w1 = two_level_tree()
print(f'Original: {len(w1.graph.nodes)} nodes, {len(w1.workers)} workers, coins={w1.economy.coins}')
print(f'Root edges: {[(e.output_label, e.target) for e in w1.graph.nodes[w1.graph.root_id].edges]}')

# Save it
save_game(w1, '/tmp/babybrain_test_save.pt')
print('Saved OK')

# Load it back
w2 = load_game('/tmp/babybrain_test_save.pt')
print(f'Loaded: {len(w2.graph.nodes)} nodes, {len(w2.workers)} workers, coins={w2.economy.coins}')
print(f'Root edges: {[(e.output_label, e.target) for e in w2.graph.nodes[w2.graph.root_id].edges]}')

# Verify worker support sets survived
for w in w2.workers:
    print(f'  {w.name}: classes={w.class_names}, support={w.get_support_set_size()}, accuracy={w.cached_accuracy:.2f}')

# Verify real inference works on loaded workers
from app.factory.objects import ObjectGenerator
gen = ObjectGenerator(difficulty=0.0)
obj = gen.generate('circle')
pred, conf = w2.workers[0].predict_real(obj.tensor)
print(f'Inference test: circle -> {pred} (conf={conf:.2f})')

print('Round-trip OK')
"
`

Expected: Matching node/edge counts before and after save/load, non-zero support set sizes, inference produces a prediction with reasonable confidence.

- [ ] **Step 3: Test --preset flag launches the game**

Run: `cd /Users/gabriel/Development/PythonProjects/babybrain && timeout 5 .venv/bin/python -m app.main --preset simple_split 2>&1 || true`

Expected: Game window appears briefly (killed by timeout). No import errors or crashes.

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: add --preset and --load CLI flags to main entry point"
```
