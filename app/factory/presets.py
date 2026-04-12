"""Pre-built factory presets for quick testing."""

from __future__ import annotations

from .objects import ObjectGenerator, SHAPE_FAMILIES
from .contracts import ALL_CONTRACTS
from .economy import Economy
from .routing import RoutingGraph
from .worker import FactoryWorker, MEMORY_CAP
from .world import FactoryWorld, resolve_worker_checkpoint


def _checkpoint() -> str:
    return resolve_worker_checkpoint()


def _make_worker(
    name: str,
    class_names: list[str],
    category_mapping: dict[str, str],
    gen: ObjectGenerator,
    device: str,
    training_cats: dict[str, list[str]] | None = None,
) -> FactoryWorker:
    """Create a worker, teach it with generated examples, and estimate accuracy.

    *category_mapping* is the full map of every incoming category to the
    output class label (used for routing stats and downstream correctness).

    *training_cats* lets the caller specify a narrow set of representative
    categories to teach each class from. This matters for concept hubs:
    the MAML backbone is meta-trained on abstract visual features, so
    prototypes built from a single clean exemplar per class generalise
    far better than prototypes averaged across a whole family. When
    omitted, training defaults to the category_mapping (terminal sorter
    behaviour — each class is taught from its own category).
    """
    w = FactoryWorker(
        name=name,
        checkpoint_path=_checkpoint(),
        num_classes=len(class_names),
        device=device,
    )

    if training_cats is None:
        training_cats = {lab: [] for lab in class_names}
        for cat, label in category_mapping.items():
            training_cats.setdefault(label, []).append(cat)

    per_class = max(1, MEMORY_CAP // max(1, len(class_names)))

    for label in class_names:
        cats_for_label = training_cats.get(label) or []
        if not cats_for_label:
            continue
        for i in range(per_class):
            cat = cats_for_label[i % len(cats_for_label)]
            obj = gen.generate(category=cat)
            w.teach(obj.tensor, label)

    w.category_mapping = dict(category_mapping)

    test_objs = []
    for cat in category_mapping.keys():
        for _ in range(20):
            test_objs.append(gen.generate(category=cat))
    w.estimate_accuracy(test_objs)
    return w


def simple_split(device: str = "cpu") -> FactoryWorld:
    """A minimal 1-node routing DAG using three Starter shapes.

    Root worker classifies objects directly into circle/triangle/square
    bins. Three classes × four examples fits the ``MEMORY_CAP=12`` budget
    exactly and the distinct shapes give the backbone an easy task.

    DAG structure:
        [root: Sorter]
           |-- "circle"   --> BIN:circle
           |-- "triangle" --> BIN:triangle
           |-- "square"   --> BIN:square
    """
    gen = ObjectGenerator(difficulty=0.0)
    world = FactoryWorld(gen)
    world.objects_per_tick = 2

    cats = ["circle", "triangle", "square"]
    world.active_categories = list(cats)
    world._remaining_categories = []

    cat_map = {c: c for c in cats}
    root_worker = _make_worker("Sorter", list(cats), cat_map, gen, device)
    world.workers.append(root_worker)

    world.graph.add_node("root", worker=root_worker)
    world.graph.set_root("root")
    for c in cats:
        world.graph.connect("root", c, f"BIN:{c}")

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
    """A profitable DAG across all four contract packs (19 shapes).

    Tree design notes:

    - Every internal hub is a **binary concept split** anchored on a
      contract shape the backbone's meta-training already saw.
    - ``mono_hub`` uses ``cloud`` vs ``triangle``: this activates the
      backbone's trained ``curved_edges vs straight_edges`` feature.
      Under that feature ``teardrop`` reads as straight (sharp tip), so
      it is routed through the straight branch.
    - Depth is capped at 3 hops via a 3-way root so per-object accuracy
      doesn't compound too steeply.
    - ``objects_per_tick`` starts at 1: each node has ``processing_speed=1``
      so the whole pipeline can absorb 1 obj/tick without the root queue
      overflowing. The throughput ramp will push this up later; by then
      the player is expected to buy speed upgrades.

        root: holed / multicolor / mono
                 (donut+gear / candy_cane+mushroom / triangle+circle)
          holed      -> holed_sorter        (4 cats)
          multicolor -> multicolor_sorter   (5 cats)
          mono       -> mono_hub: curved / straight  (cloud / triangle)
                          curved   -> 4-cat sorter (circle/heart/crescent/cloud)
                          straight -> 6-cat sorter (triangle/square/star_5/
                                                    arrow/lightning/teardrop)
    """
    gen = ObjectGenerator(difficulty=0.0)
    world = FactoryWorld(gen)
    world.objects_per_tick = 1

    starter = ["circle", "triangle", "square", "star_5", "heart"]
    sils = ["arrow", "crescent", "cloud", "lightning", "teardrop"]
    holes = ["donut", "picture_frame", "key", "gear"]
    mc_cats = ["mushroom", "tree", "flower", "candy_cane", "rainbow"]

    curved_cats = ["circle", "heart", "crescent", "cloud"]
    straight_cats = [
        "triangle", "square", "star_5", "arrow", "lightning", "teardrop",
    ]

    all_contract = starter + sils + holes + mc_cats

    for contract in ALL_CONTRACTS:
        world.accept_contract(contract.id)
    world.active_categories = list(all_contract)
    world._remaining_categories = []

    # --- Root: 3-way has_hole / multicolor / mono --------------------
    root_map: dict[str, str] = {}
    for c in holes:
        root_map[c] = "holed"
    for c in mc_cats:
        root_map[c] = "multicolor"
    for c in curved_cats + straight_cats:
        root_map[c] = "mono"
    root_worker = _make_worker(
        "Root",
        ["holed", "multicolor", "mono"],
        root_map,
        gen,
        device,
        training_cats={
            # Two-anchor teach sets — diverse within-class exemplars pull
            # the MAML prototype toward the shared feature rather than
            # over-fitting to one category's specifics. Empirically (~70%
            # 3-way accuracy) these beat single-anchor teaching.
            "holed": ["donut", "gear"],
            "multicolor": ["candy_cane", "mushroom"],
            "mono": ["triangle", "circle"],
        },
    )
    world.workers.append(root_worker)
    world.graph.add_node("root", worker=root_worker)
    world.graph.set_root("root")

    # --- Holed branch: direct 4-class sorter --------------------------
    holed_sorter = _make_worker(
        "Holed Sorter",
        list(holes),
        {c: c for c in holes},
        gen,
        device,
    )
    world.workers.append(holed_sorter)
    world.graph.add_node("holed_sorter", worker=holed_sorter)
    world.graph.connect("root", "holed", "holed_sorter")
    for c in holes:
        world.graph.connect("holed_sorter", c, f"BIN:{c}")

    # --- Multicolor branch: direct 5-class sorter ---------------------
    multicolor_sorter = _make_worker(
        "Multicolor Sorter",
        list(mc_cats),
        {c: c for c in mc_cats},
        gen,
        device,
    )
    world.workers.append(multicolor_sorter)
    world.graph.add_node("multicolor_sorter", worker=multicolor_sorter)
    world.graph.connect("root", "multicolor", "multicolor_sorter")
    for c in mc_cats:
        world.graph.connect("multicolor_sorter", c, f"BIN:{c}")

    # --- Mono branch: curved / straight (cloud / triangle anchors) ----
    mono_map: dict[str, str] = {}
    for c in curved_cats:
        mono_map[c] = "curved"
    for c in straight_cats:
        mono_map[c] = "straight"
    mono_hub = _make_worker(
        "Mono Hub",
        ["curved", "straight"],
        mono_map,
        gen,
        device,
        training_cats={"curved": ["cloud"], "straight": ["triangle"]},
    )
    world.workers.append(mono_hub)
    world.graph.add_node("mono_hub", worker=mono_hub)
    world.graph.connect("root", "mono", "mono_hub")

    curved_sorter = _make_worker(
        "Curved Sorter",
        list(curved_cats),
        {c: c for c in curved_cats},
        gen,
        device,
    )
    world.workers.append(curved_sorter)
    world.graph.add_node("curved_sorter", worker=curved_sorter)
    world.graph.connect("mono_hub", "curved", "curved_sorter")
    for c in curved_cats:
        world.graph.connect("curved_sorter", c, f"BIN:{c}")

    straight_sorter = _make_worker(
        "Straight Sorter",
        list(straight_cats),
        {c: c for c in straight_cats},
        gen,
        device,
    )
    world.workers.append(straight_sorter)
    world.graph.add_node("straight_sorter", worker=straight_sorter)
    world.graph.connect("mono_hub", "straight", "straight_sorter")
    for c in straight_cats:
        world.graph.connect("straight_sorter", c, f"BIN:{c}")

    world.economy.coins = 5000.0
    return world
