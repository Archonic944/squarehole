"""Routing graph — directs FactoryObjects through a network of workers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .worker import FactoryWorker

# Avoid circular import; FactoryObject is only needed for type hints at
# runtime, so we import it inside methods where necessary and use a
# string annotation everywhere else.


@dataclass
class RoutingEdge:
    """An edge from a routing node, keyed by the prediction label."""

    output_label: str  # e.g. "pointy" or "star_5"
    target: str  # node_id or "BIN:category_name"


@dataclass
class TickResults:
    """Accumulated results from a single graph tick."""

    correct: list[tuple] = field(default_factory=list)
    # (FactoryObject, bin_name)

    wrong: list[tuple] = field(default_factory=list)
    # (FactoryObject, predicted_bin, true_category)

    dropped: list = field(default_factory=list)
    # FactoryObject instances that fell off (no matching edge or no worker)

    unprocessed: int = 0
    # Count of objects still sitting in queues (capacity limit)

    flows: list[tuple] = field(default_factory=list)
    # (FactoryObject, from_node_id, target_str, prediction)
    # target_str is a node_id or "BIN:name"


class RoutingNode:
    """A single node in the routing graph.

    Each node holds a worker (or ``None`` for pass-through/drop) and a queue
    of objects waiting to be processed.
    """

    def __init__(
        self,
        node_id: str,
        worker: FactoryWorker | None = None,
        processing_speed: int = 1,
        queue_capacity: int = 20,
    ):
        self.node_id = node_id
        self.worker = worker
        self.edges: list[RoutingEdge] = []
        self.queue: deque = deque()
        self.processing_speed = processing_speed
        self.queue_capacity = queue_capacity

    def route(self, prediction: str) -> str | None:
        """Return the target for *prediction*, or ``None`` if no edge matches."""
        for edge in self.edges:
            if edge.output_label == prediction:
                return edge.target
        return None

    def __repr__(self) -> str:
        worker_name = self.worker.name if self.worker else "None"
        return (
            f"RoutingNode(id={self.node_id!r}, worker={worker_name}, "
            f"edges={len(self.edges)}, queue={len(self.queue)})"
        )


class RoutingGraph:
    """A directed graph of :class:`RoutingNode` instances.

    Objects enter at the *root* node, get classified by each node's worker,
    and are routed along edges until they reach an output bin (``BIN:name``)
    or fall off the graph.
    """

    def __init__(self):
        self.nodes: dict[str, RoutingNode] = {}
        self.root_id: str | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        worker: FactoryWorker | None = None,
        processing_speed: int = 1,
        queue_capacity: int = 20,
    ) -> RoutingNode:
        node = RoutingNode(node_id, worker=worker, processing_speed=processing_speed, queue_capacity=queue_capacity)
        self.nodes[node_id] = node
        if self.root_id is None:
            self.root_id = node_id
        return node

    def connect(self, from_id: str, output_label: str, target: str):
        """Add an edge from *from_id* to *target* on prediction *output_label*.

        *target* can be another node_id or ``"BIN:category_name"``.
        """
        node = self.nodes[from_id]
        node.edges.append(RoutingEdge(output_label=output_label, target=target))

    def remove_node(self, node_id: str):
        """Remove a node and all edges pointing to it."""
        self.nodes.pop(node_id, None)
        if self.root_id == node_id:
            self.root_id = None
        # Prune dangling edges in remaining nodes
        for node in self.nodes.values():
            node.edges = [e for e in node.edges if e.target != node_id]

    def set_root(self, node_id: str):
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id!r} does not exist in the graph")
        self.root_id = node_id

    # ------------------------------------------------------------------
    # Tick processing
    # ------------------------------------------------------------------

    def _bfs_order(self) -> list[str]:
        """Return node ids in BFS order starting from root."""
        if self.root_id is None or self.root_id not in self.nodes:
            return []

        visited: set[str] = set()
        order: list[str] = []
        frontier: deque[str] = deque([self.root_id])

        while frontier:
            nid = frontier.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            order.append(nid)

            node = self.nodes[nid]
            for edge in node.edges:
                if not edge.target.startswith("BIN:") and edge.target in self.nodes:
                    frontier.append(edge.target)

        return order

    def process_tick(
        self,
        new_objects: list,
        use_real_inference: bool = False,
    ) -> TickResults:
        """Process one simulation tick.

        1. Enqueue *new_objects* into the root node's queue.
        2. Process nodes in BFS order so objects can flow through
           the entire graph within a single tick.
        3. For each node that has a worker and queued objects, classify
           up to ``processing_speed`` objects, then route them.

        Returns a :class:`TickResults` summarising what happened.
        """
        results = TickResults()

        # Feed new objects into the root (drop overflow)
        if self.root_id and self.root_id in self.nodes:
            root = self.nodes[self.root_id]
            for obj in new_objects:
                if len(root.queue) < root.queue_capacity:
                    root.queue.append(obj)
                else:
                    results.dropped.append(obj)
        else:
            # No root — everything is dropped
            results.dropped.extend(new_objects)
            return results

        processing_order = self._bfs_order()

        for nid in processing_order:
            node = self.nodes[nid]

            if not node.queue:
                continue

            # No worker — drop everything in this node's queue
            if node.worker is None:
                while node.queue:
                    results.dropped.append(node.queue.popleft())
                continue

            worker = node.worker
            to_process = min(len(node.queue), node.processing_speed)

            for _ in range(to_process):
                obj = node.queue.popleft()

                # Classify
                if use_real_inference:
                    prediction, _ = worker.predict_real(obj.tensor)
                else:
                    prediction, _ = worker.predict_simulated(
                        obj.category, worker.class_names
                    )

                # Update worker stats
                worker.stats.total_processed += 1
                if worker.category_mapping:
                    expected = worker.category_mapping.get(obj.category)
                    if prediction == expected:
                        worker.stats.total_correct += 1
                elif prediction == obj.category:
                    worker.stats.total_correct += 1

                # Route
                target = node.route(prediction)
                if target is None:
                    results.dropped.append(obj)
                elif target.startswith("BIN:"):
                    bin_name = target[4:]
                    results.flows.append((obj, nid, target, prediction))
                    if bin_name == obj.category:
                        results.correct.append((obj, bin_name))
                    else:
                        results.wrong.append((obj, bin_name, obj.category))
                else:
                    # Route to another node's queue (drop if full)
                    if target in self.nodes:
                        child = self.nodes[target]
                        if len(child.queue) < child.queue_capacity:
                            child.queue.append(obj)
                            results.flows.append((obj, nid, target, prediction))
                        else:
                            results.dropped.append(obj)
                    else:
                        results.dropped.append(obj)

        # Count objects still sitting in queues
        results.unprocessed = sum(len(n.queue) for n in self.nodes.values())

        return results
