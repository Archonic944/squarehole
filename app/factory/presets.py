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
