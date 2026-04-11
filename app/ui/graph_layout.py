"""Force-directed DAG layout engine.

Runs a physics simulation to position graph nodes without overlap,
maintaining left-to-right flow via depth-constrained x-positioning.
Pure computation — no pygame dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.factory.routing import RoutingGraph

# Layout parameters
IDEAL_EDGE_LEN = 200
ALPHA_DECAY = 0.05
ALPHA_MIN = 0.005
VELOCITY_DECAY = 0.4
MAX_ITERS = 150
REPULSION_STRENGTH = -300
DEPTH_X_STRENGTH = 0.3
CENTER_Y_STRENGTH = 0.02
COLLISION_PAD = 20

# Node dimensions (must match factory_floor.py rendering)
NODE_W = 140
NODE_H = 50
BIN_W = 60
BIN_H = 20


@dataclass
class SimNode:
    node_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    depth: int = 0
    is_bin: bool = False

    @property
    def w(self) -> int:
        return BIN_W if self.is_bin else NODE_W

    @property
    def h(self) -> int:
        return BIN_H if self.is_bin else NODE_H


class ForceLayout:
    """Runs a force-directed simulation on a RoutingGraph."""

    def __init__(self, graph: RoutingGraph):
        self.sim_nodes: dict[str, SimNode] = {}
        self.edges: list[tuple[str, str]] = []
        self._build_from_graph(graph)

    def _build_from_graph(self, graph: RoutingGraph) -> None:
        """Extract SimNodes and edges from a RoutingGraph via BFS."""
        if not graph.nodes:
            return

        # BFS to assign depths
        depths: dict[str, int] = {}
        if graph.root_id and graph.root_id in graph.nodes:
            q: deque[tuple[str, int]] = deque([(graph.root_id, 0)])
            visited: set[str] = set()
            while q:
                nid, d = q.popleft()
                if nid in visited:
                    continue
                visited.add(nid)
                depths[nid] = d
                node = graph.nodes[nid]
                for edge in node.edges:
                    t = edge.target
                    if not t.startswith("BIN:") and t in graph.nodes and t not in visited:
                        q.append((t, d + 1))

        # Assign orphan nodes
        max_depth = max(depths.values()) if depths else -1
        for nid in graph.nodes:
            if nid not in depths:
                max_depth += 1
                depths[nid] = max_depth

        # Create SimNodes for real nodes
        for nid in graph.nodes:
            self.sim_nodes[nid] = SimNode(
                node_id=nid, x=0.0, y=0.0, depth=depths.get(nid, 0)
            )

        # Create SimNodes for bins + edges for all connections
        for nid, node in graph.nodes.items():
            parent_depth = depths.get(nid, 0)
            for edge in node.edges:
                target = edge.target
                if target.startswith("BIN:"):
                    bin_key = f"{nid}:{target}"
                    if bin_key not in self.sim_nodes:
                        self.sim_nodes[bin_key] = SimNode(
                            node_id=bin_key, x=0.0, y=0.0,
                            depth=parent_depth + 1, is_bin=True,
                        )
                    self.edges.append((nid, bin_key))
                elif target in graph.nodes:
                    self.edges.append((nid, target))

        # Initialize positions: depth-based grid
        self._init_positions()

    def _init_positions(self) -> None:
        """Place nodes in a depth-based grid as starting positions."""
        by_depth: dict[int, list[SimNode]] = {}
        for sn in self.sim_nodes.values():
            by_depth.setdefault(sn.depth, []).append(sn)

        for depth, nodes in by_depth.items():
            x = depth * IDEAL_EDGE_LEN + IDEAL_EDGE_LEN / 2
            total_h = sum(n.h for n in nodes) + (len(nodes) - 1) * 30
            y_start = -total_h / 2
            for i, sn in enumerate(nodes):
                sn.x = x
                sn.y = y_start + i * (sn.h + 30)

    # ------------------------------------------------------------------
    # Forces
    # ------------------------------------------------------------------

    def _apply_depth_x_force(self, alpha: float) -> None:
        """Push nodes toward x = depth * IDEAL_EDGE_LEN."""
        for sn in self.sim_nodes.values():
            target_x = sn.depth * IDEAL_EDGE_LEN + IDEAL_EDGE_LEN / 2
            sn.vx += (target_x - sn.x) * DEPTH_X_STRENGTH * alpha

    def _apply_center_y_force(self, alpha: float) -> None:
        """Weak pull toward y=0 to prevent vertical drift."""
        for sn in self.sim_nodes.values():
            sn.vy += (0 - sn.y) * CENTER_Y_STRENGTH * alpha

    def _apply_repulsion(self, alpha: float) -> None:
        """Repel all node pairs (O(n^2), fine for small graphs)."""
        nodes = list(self.sim_nodes.values())
        n = len(nodes)
        for i in range(n):
            a = nodes[i]
            for j in range(i + 1, n):
                b = nodes[j]
                dx = b.x - a.x
                dy = b.y - a.y
                dist_sq = dx * dx + dy * dy
                if dist_sq < 1.0:
                    dx = (hash(a.node_id) % 10 - 5) * 0.1
                    dy = (hash(b.node_id) % 10 - 5) * 0.1
                    dist_sq = dx * dx + dy * dy + 0.1
                dist = math.sqrt(dist_sq)
                if dist > 500:
                    continue
                force = REPULSION_STRENGTH * alpha / dist
                fx = force * dx / dist
                fy = force * dy / dist
                a.vx += fx
                a.vy += fy
                b.vx -= fx
                b.vy -= fy

    def _apply_edge_springs(self, alpha: float) -> None:
        """Spring force pulling connected nodes toward ideal distance."""
        degree: dict[str, int] = {}
        for src, tgt in self.edges:
            degree[src] = degree.get(src, 0) + 1
            degree[tgt] = degree.get(tgt, 0) + 1

        for src_id, tgt_id in self.edges:
            a = self.sim_nodes.get(src_id)
            b = self.sim_nodes.get(tgt_id)
            if not a or not b:
                continue
            dx = b.x - a.x
            dy = b.y - a.y
            dist = math.sqrt(dx * dx + dy * dy) or 0.1
            strength = 1.0 / min(degree.get(src_id, 1), degree.get(tgt_id, 1))
            displacement = (dist - IDEAL_EDGE_LEN) * strength * alpha
            fx = displacement * dx / dist
            fy = displacement * dy / dist
            a.vx += fx * 0.5
            a.vy += fy * 0.5
            b.vx -= fx * 0.5
            b.vy -= fy * 0.5

    def _apply_collision(self) -> None:
        """Push overlapping node rectangles apart."""
        nodes = list(self.sim_nodes.values())
        n = len(nodes)
        for i in range(n):
            a = nodes[i]
            for j in range(i + 1, n):
                b = nodes[j]
                hw = (a.w + b.w) / 2 + COLLISION_PAD
                hh = (a.h + b.h) / 2 + COLLISION_PAD
                dx = b.x - a.x
                dy = b.y - a.y
                overlap_x = hw - abs(dx)
                overlap_y = hh - abs(dy)
                if overlap_x > 0 and overlap_y > 0:
                    if overlap_x < overlap_y:
                        push = overlap_x / 2
                        if dx > 0:
                            a.x -= push
                            b.x += push
                        else:
                            a.x += push
                            b.x -= push
                    else:
                        push = overlap_y / 2
                        if dy > 0:
                            a.y -= push
                            b.y += push
                        else:
                            a.y += push
                            b.y -= push

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Run simulation to convergence. Returns iteration count."""
        if not self.sim_nodes:
            return 0
        alpha = 1.0
        for i in range(MAX_ITERS):
            alpha += (0 - alpha) * ALPHA_DECAY
            if alpha < ALPHA_MIN:
                return i

            self._apply_depth_x_force(alpha)
            self._apply_center_y_force(alpha)
            self._apply_repulsion(alpha)
            self._apply_edge_springs(alpha)
            self._apply_collision()

            max_disp = 0.0
            for sn in self.sim_nodes.values():
                sn.vx *= (1 - VELOCITY_DECAY)
                sn.vy *= (1 - VELOCITY_DECAY)
                dx = sn.vx
                dy = sn.vy
                sn.x += dx
                sn.y += dy
                max_disp = max(max_disp, abs(dx) + abs(dy))

            if max_disp < 1.0:
                return i
        return MAX_ITERS

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Return (min_x, min_y, max_x, max_y) bounding box of all nodes."""
        if not self.sim_nodes:
            return (0, 0, 100, 100)
        min_x = min(sn.x - sn.w / 2 for sn in self.sim_nodes.values())
        min_y = min(sn.y - sn.h / 2 for sn in self.sim_nodes.values())
        max_x = max(sn.x + sn.w / 2 for sn in self.sim_nodes.values())
        max_y = max(sn.y + sn.h / 2 for sn in self.sim_nodes.values())
        return (min_x, min_y, max_x, max_y)
