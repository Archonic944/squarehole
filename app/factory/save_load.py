"""Save and load factory game state."""

from __future__ import annotations

import os

import torch

from .objects import ObjectGenerator
from .routing import RoutingGraph, RoutingEdge
from .worker import FactoryWorker
from .world import FactoryWorld, GENERAL_CHECKPOINT, WHATS_CHECKPOINT

SAVE_VERSION = 1


def save_game(world: FactoryWorld, path: str) -> None:
    """Serialize the full game state and write it to *path* using torch.save.

    Parameters
    ----------
    world :
        The :class:`FactoryWorld` to serialize.
    path :
        Destination file path.  Parent directories are created as needed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # --- Workers ---
    workers_data = []
    for w in world.workers:
        support = {
            cls: [t.cpu() for t in tensors]
            for cls, tensors in w._support_set.items()
        }
        workers_data.append(
            {
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
            }
        )

    # --- Graph nodes ---
    # Build a worker → index map so nodes can reference workers by index.
    worker_index: dict[int, int] = {id(w): i for i, w in enumerate(world.workers)}

    nodes_data = []
    for node_id, node in world.graph.nodes.items():
        worker_idx = None
        if node.worker is not None:
            worker_idx = worker_index.get(id(node.worker))

        edges_data = [
            {"output_label": e.output_label, "target": e.target}
            for e in node.edges
        ]

        nodes_data.append(
            {
                "node_id": node_id,
                "queue_capacity": node.queue_capacity,
                "worker_index": worker_idx,
                "edges": edges_data,
            }
        )

    # --- Economy ---
    economy_data = {
        "coins": world.economy.coins,
        "total_earned": world.economy.total_earned,
        "total_spent": world.economy.total_spent,
        "total_penalties": world.economy.total_penalties,
    }

    state = {
        "version": SAVE_VERSION,
        # World
        "tick_count": world.tick_count,
        "objects_per_tick": world.objects_per_tick,
        "speed_level": world.speed_level,
        "active_categories": list(world.active_categories),
        "_remaining_categories": list(world._remaining_categories),
        # Economy
        "economy": economy_data,
        # Graph
        "root_id": world.graph.root_id,
        "nodes": nodes_data,
        # Workers
        "workers": workers_data,
        # Generator
        "generator_difficulty": world.object_generator.difficulty,
    }

    torch.save(state, path)


def load_game(path: str, device: str = "cpu") -> FactoryWorld:
    """Load a save file produced by :func:`save_game` and return a ready-to-play
    :class:`FactoryWorld`.

    Parameters
    ----------
    path :
        Path to the save file.
    device :
        Torch device string for worker models.

    Returns
    -------
    FactoryWorld
    """
    state = torch.load(path, map_location=device, weights_only=False)

    # --- Generator ---
    difficulty = state.get("generator_difficulty", 0.0)
    generator = ObjectGenerator(difficulty=difficulty)

    # --- World skeleton ---
    world = FactoryWorld(object_generator=generator)
    world.tick_count = state["tick_count"]
    world.objects_per_tick = state["objects_per_tick"]
    world.speed_level = state["speed_level"]
    world.active_categories = list(state["active_categories"])
    world._remaining_categories = list(state["_remaining_categories"])

    # --- Economy ---
    econ_data = state["economy"]
    world.economy.coins = econ_data["coins"]
    world.economy.total_earned = econ_data["total_earned"]
    world.economy.total_spent = econ_data["total_spent"]
    world.economy.total_penalties = econ_data["total_penalties"]

    # --- Workers ---
    checkpoint = (
        GENERAL_CHECKPOINT
        if os.path.exists(GENERAL_CHECKPOINT)
        else WHATS_CHECKPOINT
    )

    workers: list[FactoryWorker] = []
    for wdata in state["workers"]:
        # num_classes at construction time; the head will be rebuilt by teach()
        num_classes = max(wdata["num_classes"], 2)
        worker = FactoryWorker(
            name=wdata["name"],
            checkpoint_path=checkpoint,
            num_classes=num_classes,
            device=device,
        )

        # Replay support set through teach() so heads and class_names are rebuilt
        support_set: dict[str, list[torch.Tensor]] = wdata["support_set"]
        for class_name in wdata["class_names"]:
            for tensor in support_set.get(class_name, []):
                worker.teach(tensor.to(device), class_name)

        # Restore fields that teach() would overwrite
        worker.role = wdata["role"]
        worker.cached_accuracy = wdata["cached_accuracy"]
        worker.category_mapping = dict(wdata["category_mapping"])

        # Restore stats
        stats_data = wdata["stats"]
        worker.stats.total_processed = stats_data["total_processed"]
        worker.stats.total_correct = stats_data["total_correct"]
        worker.stats.coins_earned = stats_data["coins_earned"]

        workers.append(worker)

    world.workers = workers

    # --- Graph ---
    graph = RoutingGraph()
    for ndata in state["nodes"]:
        node_id = ndata["node_id"]
        worker_idx = ndata["worker_index"]
        worker = workers[worker_idx] if worker_idx is not None else None
        node = graph.add_node(
            node_id,
            worker=worker,
            queue_capacity=ndata["queue_capacity"],
        )
        for edata in ndata["edges"]:
            node.edges.append(
                RoutingEdge(
                    output_label=edata["output_label"],
                    target=edata["target"],
                )
            )

    # add_node sets root_id to the first inserted node; override with saved value
    graph.root_id = state.get("root_id")

    world.graph = graph

    return world
