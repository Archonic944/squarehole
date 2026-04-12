"""Save/load system for factory game state."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass

import torch

from .objects import ObjectGenerator
from .economy import Economy
from .routing import RoutingGraph, RoutingNode, RoutingEdge
from .worker import FactoryWorker
from .world import FactoryWorld, resolve_worker_checkpoint


@dataclass
class SaveInfo:
    name: str
    path: str
    mtime: float
    size: int


def saves_dir() -> str:
    d = os.path.expanduser("~/.babybrain/saves")
    os.makedirs(d, exist_ok=True)
    return d


def sanitize_save_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_\- ]", "", name)
    return name[:40]


def save_path_for(name: str) -> str:
    return os.path.join(saves_dir(), f"{name}.pt")


def list_saves() -> list[SaveInfo]:
    d = saves_dir()
    out: list[SaveInfo] = []
    for fn in os.listdir(d):
        if not fn.endswith(".pt"):
            continue
        p = os.path.join(d, fn)
        try:
            st = os.stat(p)
        except OSError:
            continue
        out.append(SaveInfo(name=fn[:-3], path=p, mtime=st.st_mtime, size=st.st_size))
    out.sort(key=lambda s: s.mtime, reverse=True)
    return out


def delete_save(name: str) -> bool:
    p = save_path_for(name)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


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
            "accepted_contract_ids": sorted(world.accepted_contract_ids),
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
    world.accepted_contract_ids = set(wd.get("accepted_contract_ids", ["starter"]))

    # Reconstruct economy
    ed = data["economy"]
    world.economy.coins = ed["coins"]
    world.economy.total_earned = ed["total_earned"]
    world.economy.total_spent = ed["total_spent"]
    world.economy.total_penalties = ed["total_penalties"]

    # Reconstruct workers
    checkpoint = resolve_worker_checkpoint()
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
                w.teach(t.to(device), cls_name, force=True)
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
