# Force-Directed DAG Layout with Minimap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the tree-based graph layout with a force-directed simulation that prevents node overlap, and add a minimap for panning large graphs.

**Architecture:** A new `app/ui/graph_layout.py` module owns the force simulation (SimNode, ForceLayout). `factory_floor.py` calls it from `_layout_graph()`, stores results in `_node_rects`/`_bin_positions` (now in virtual canvas coords), and draws everything with a viewport offset. A minimap in the top-right corner shows the full graph thumbnail and handles click-to-pan.

**Tech Stack:** Python 3.11, pygame 2.x. No external graph libraries — simulation implemented from scratch.

**Spec:** `docs/superpowers/specs/2026-04-11-force-directed-layout-design.md`

---

## File Structure

| File | Role |
|------|------|
| `app/ui/graph_layout.py` | **NEW** — SimNode dataclass, ForceLayout class with `run()` method. Pure computation, no pygame dependency. |
| `app/ui/factory_floor.py` | **MODIFY** — Replace `_layout_graph()`, add viewport state + transforms, modify `_draw_graph()` for viewport offset, add minimap rendering/interaction, update all hit testing. |

---

### Task 1: Force Simulation Engine

**Files:**
- Create: `app/ui/graph_layout.py`

This is the core physics engine. It takes a RoutingGraph, builds SimNodes (including bins), and runs the force simulation to convergence. No pygame imports — pure math.

- [ ] **Step 1: Create `app/ui/graph_layout.py` with data structures and BFS depth assignment**

```python
"""Force-directed DAG layout engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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

# Node dimensions (must match factory_floor.py)
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
        self.edges: list[tuple[str, str]] = []  # (source_id, target_id)
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
```

- [ ] **Step 2: Add force application methods**

Append to `ForceLayout` class in `app/ui/graph_layout.py`:

```python
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
                    # Jitter to break ties
                    dx = (hash(a.node_id) % 10 - 5) * 0.1
                    dy = (hash(b.node_id) % 10 - 5) * 0.1
                    dist_sq = dx * dx + dy * dy + 0.1
                dist = math.sqrt(dist_sq)
                if dist > 500:
                    continue
                # Force magnitude: strength / dist
                force = REPULSION_STRENGTH * alpha / dist
                fx = force * dx / dist
                fy = force * dy / dist
                a.vx += fx
                a.vy += fy
                b.vx -= fx
                b.vy -= fy

    def _apply_edge_springs(self, alpha: float) -> None:
        """Spring force pulling connected nodes toward ideal distance."""
        # Precompute degree for strength scaling
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
            # Strength inversely proportional to degree
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
                # Half-widths and half-heights with padding
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
```

- [ ] **Step 3: Add the `run()` method**

Append to `ForceLayout` class:

```python
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
```

- [ ] **Step 4: Verify it runs without errors**

```bash
cd /Users/gabriel/Development/PythonProjects/babybrain
.venv/bin/python -c "
from app.factory.routing import RoutingGraph
from app.ui.graph_layout import ForceLayout

g = RoutingGraph()
g.add_node('root')
g.add_node('child')
g.set_root('root')
g.connect('root', 'left', 'child')
g.connect('root', 'right', 'BIN:circle')
g.connect('child', 'star', 'BIN:star_5')

layout = ForceLayout(g)
iters = layout.run()
print(f'Converged in {iters} iterations')
for nid, sn in layout.sim_nodes.items():
    print(f'  {nid}: ({sn.x:.0f}, {sn.y:.0f}) depth={sn.depth} bin={sn.is_bin}')
bounds = layout.get_bounds()
print(f'Bounds: {bounds}')
"
```

Expected: Converges in <150 iterations. Nodes at different positions. Bins at depth = parent+1. No overlapping positions.

- [ ] **Step 5: Commit**

```bash
git add app/ui/graph_layout.py
git commit -m "feat: add force-directed DAG layout engine"
```

---

### Task 2: Replace `_layout_graph()` and Add Viewport State

**Files:**
- Modify: `app/ui/factory_floor.py`

Replace the tree layout with the force simulation. Add viewport state (virtual canvas coords). Keep `_node_rects` and `_bin_positions` as the output interface — they now hold virtual canvas coordinates.

- [ ] **Step 1: Add viewport state to `__init__`**

In `app/ui/factory_floor.py`, add after `self._layout_dirty = True` (line 327):

```python
        # Viewport for virtual canvas (panning)
        self._viewport_x = 0.0
        self._viewport_y = 0.0
        self._canvas_x = 0.0
        self._canvas_y = 0.0
        self._canvas_w = float(GRAPH_W)
        self._canvas_h = float(GRAPH_H)
        self._viewport_initialized = False
```

- [ ] **Step 2: Add import for ForceLayout**

At top of `factory_floor.py`, add:

```python
from app.ui.graph_layout import ForceLayout, NODE_W as SIM_NODE_W, NODE_H as SIM_NODE_H
```

- [ ] **Step 3: Replace `_layout_graph()` entirely**

Delete the entire old `_layout_graph()` method (lines 373-539) and replace with:

```python
    def _layout_graph(self):
        """Run force-directed simulation and store results."""
        self._node_rects.clear()
        self._bin_positions.clear()

        graph = self.world.graph
        if not graph.nodes:
            self._layout_dirty = False
            return

        layout = ForceLayout(graph)
        layout.run()

        # Copy node positions to _node_rects (virtual canvas coords)
        for nid, sn in layout.sim_nodes.items():
            if not sn.is_bin:
                self._node_rects[nid] = pygame.Rect(
                    int(sn.x - sn.w / 2), int(sn.y - sn.h / 2),
                    sn.w, sn.h,
                )

        # Copy bin positions to _bin_positions
        for nid, sn in layout.sim_nodes.items():
            if sn.is_bin:
                self._bin_positions[nid] = (int(sn.x - sn.w / 2), int(sn.y - sn.h / 2))

        # Compute virtual canvas bounds
        self._compute_canvas_bounds()

        # Center viewport on root on first layout
        if not self._viewport_initialized:
            self._center_viewport_on_root()
            self._viewport_initialized = True

        self._layout_dirty = False

    def _compute_canvas_bounds(self):
        """Compute virtual canvas bounding box from all nodes + bins."""
        margin = 60
        all_rects = list(self._node_rects.values())
        all_points = list(self._bin_positions.values())

        if not all_rects and not all_points:
            self._canvas_x = 0.0
            self._canvas_y = 0.0
            self._canvas_w = float(GRAPH_W)
            self._canvas_h = float(GRAPH_H)
            return

        min_x = min((r.left for r in all_rects), default=0)
        min_y = min((r.top for r in all_rects), default=0)
        max_x = max((r.right for r in all_rects), default=0)
        max_y = max((r.bottom for r in all_rects), default=0)

        for bx, by in all_points:
            min_x = min(min_x, bx)
            min_y = min(min_y, by)
            max_x = max(max_x, bx + 80)  # approximate bin label width
            max_y = max(max_y, by + 20)

        self._canvas_x = min_x - margin
        self._canvas_y = min_y - margin
        self._canvas_w = max(float(GRAPH_W), (max_x - min_x) + 2 * margin)
        self._canvas_h = max(float(GRAPH_H), (max_y - min_y) + 2 * margin)

    def _center_viewport_on_root(self):
        """Center viewport on the root node."""
        graph = self.world.graph
        if graph.root_id and graph.root_id in self._node_rects:
            rect = self._node_rects[graph.root_id]
            self._viewport_x = rect.centerx - GRAPH_W / 2
            self._viewport_y = rect.centery - GRAPH_H / 2
        else:
            self._viewport_x = self._canvas_x
            self._viewport_y = self._canvas_y
        self._clamp_viewport()

    def _clamp_viewport(self):
        """Clamp viewport so it doesn't go outside canvas bounds."""
        self._viewport_x = max(self._canvas_x,
                               min(self._canvas_x + self._canvas_w - GRAPH_W,
                                   self._viewport_x))
        self._viewport_y = max(self._canvas_y,
                               min(self._canvas_y + self._canvas_h - GRAPH_H,
                                   self._viewport_y))
```

- [ ] **Step 4: Remove old layout constants that are no longer needed**

Remove or comment out these lines (they were only used by the old tree layout):

```python
LEVEL_GAP_X = 260  # horizontal distance between tree levels
NODE_GAP_Y = 12    # vertical gap between nodes at same level
```

Keep `GRAPH_PAD = 30` — it's still used elsewhere.

- [ ] **Step 5: Verify the game launches and graph nodes appear**

```bash
.venv/bin/python -m app.main
```

Add a couple of nodes and verify they appear in the graph area (they may be offset since drawing hasn't been updated for viewport yet — that's fine, just verify no crashes).

- [ ] **Step 6: Commit**

```bash
git add app/ui/graph_layout.py app/ui/factory_floor.py
git commit -m "feat: replace tree layout with force-directed simulation"
```

---

### Task 3: Viewport-Offset Drawing

**Files:**
- Modify: `app/ui/factory_floor.py` — `_draw_graph()` method

All drawing in `_draw_graph()` needs to apply the viewport offset. Nodes, edges, bins, flow shapes, and the connecting-mode indicator all use virtual canvas coords and must be translated to screen coords.

- [ ] **Step 1: Add viewport transform helper**

Add method to `FactoryFloorUI`:

```python
    def _vx(self, virtual_x: float) -> int:
        """Transform virtual canvas x to screen x."""
        return int(virtual_x - self._viewport_x) + GRAPH_X

    def _vy(self, virtual_y: float) -> int:
        """Transform virtual canvas y to screen y."""
        return int(virtual_y - self._viewport_y) + GRAPH_Y
```

- [ ] **Step 2: Update `_draw_graph()` to use clipping and viewport offset**

Rewrite `_draw_graph()`. The key changes are:
1. Set clip rect to graph area before drawing
2. Every coordinate uses `_vx()`/`_vy()` transforms
3. Unset clip after drawing (before minimap)

```python
    def _draw_graph(self, surface):
        # Background
        pygame.draw.rect(surface, BG, (GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H))

        graph = self.world.graph

        if not graph.nodes:
            msg = self.font.render("No nodes yet. Add a router or specialist.", True, (150, 150, 150))
            surface.blit(msg, (GRAPH_W // 2 - msg.get_width() // 2, GRAPH_Y + GRAPH_H // 2))
            return

        # Clip to graph area so nothing draws outside
        surface.set_clip(pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H))

        vx, vy = self._vx, self._vy

        # Draw edges first (behind nodes)
        for nid, node in graph.nodes.items():
            src_rect = self._node_rects.get(nid)
            if not src_rect:
                continue
            for edge in node.edges:
                if edge.target.startswith("BIN:"):
                    key = f"{nid}:{edge.target}"
                    if key in self._bin_positions:
                        bx_v, by_v = self._bin_positions[key]
                        bin_edges = [e for e in node.edges if e.target.startswith("BIN:")]
                        edge_i = bin_edges.index(edge)
                        n_bins = len(bin_edges)
                        y_spread = (edge_i - (n_bins - 1) / 2) * 10
                        start = (vx(src_rect.right), vy(src_rect.centery + int(y_spread)))
                        end = (vx(bx_v), vy(by_v + 8))
                        pygame.draw.line(surface, EDGE_COLOR, start, end, 2)
                        # Bin label
                        cat_name = edge.target[4:]
                        bt = self.font_sm.render(f"[{cat_name}]", True, (60, 120, 60))
                        surface.blit(bt, (vx(bx_v), vy(by_v)))
                        # Edge label centered on line
                        label_text = edge.output_label
                        lt = self.font_sm.render(label_text, True, (70, 70, 130))
                        lw, lh = lt.get_size()
                        lx = (start[0] + end[0]) // 2 - lw // 2
                        ly = (start[1] + end[1]) // 2 - lh // 2
                        pygame.draw.rect(surface, BG, (lx - 3, ly - 1, lw + 6, lh + 2))
                        surface.blit(lt, (lx, ly))
                elif edge.target in self._node_rects:
                    dst_rect = self._node_rects[edge.target]
                    edge_idx = next((i for i, e in enumerate(
                        [e for e in node.edges if not e.target.startswith("BIN:")]) if e is edge), 0)
                    n_node_edges = len([e for e in node.edges if not e.target.startswith("BIN:")])
                    y_off = (edge_idx - n_node_edges / 2) * 12

                    start = (vx(src_rect.right), vy(src_rect.centery + int(y_off)))
                    end = (vx(dst_rect.left), vy(dst_rect.centery + int(y_off)))
                    pygame.draw.line(surface, EDGE_COLOR, start, end, 2)
                    # Arrow head
                    ax, ay = end
                    pygame.draw.polygon(surface, EDGE_COLOR, [
                        (ax, ay), (ax - 8, ay - 5), (ax - 8, ay + 5)
                    ])
                    # Edge label
                    lt = self.font_sm.render(edge.output_label, True, (70, 70, 130))
                    lw, lh = lt.get_size()
                    mid_x = (start[0] + end[0]) // 2 - lw // 2
                    mid_y = (start[1] + end[1]) // 2 - lh // 2
                    pygame.draw.rect(surface, BG, (mid_x - 3, mid_y - 1, lw + 6, lh + 2))
                    surface.blit(lt, (mid_x, mid_y))

        # Draw nodes
        for nid, rect in self._node_rects.items():
            node = graph.nodes.get(nid)
            if not node:
                continue
            screen_rect = pygame.Rect(vx(rect.x), vy(rect.y), rect.w, rect.h)

            is_selected = (nid == self.selected_node)
            if is_selected:
                fill = NODE_SELECTED
            elif node.worker:
                fill = NODE_FILL
            else:
                fill = NODE_EMPTY

            pygame.draw.rect(surface, fill, screen_rect, border_radius=6)
            pygame.draw.rect(surface, (0, 0, 0), screen_rect, 2, border_radius=6)

            if nid == graph.root_id:
                pygame.draw.circle(surface, ACCENT, (screen_rect.left + 10, screen_rect.top + 10), 5)

            if node.worker:
                name_t = self.font_sm.render(node.worker.name[:18], True, TEXT_DARK)
                surface.blit(name_t, (screen_rect.x + 5, screen_rect.y + 5))
                acc = node.worker.cached_accuracy * 100
                spd = node.processing_speed
                info = f"{acc:.0f}% spd={spd}"
                info_t = self.font_sm.render(info, True, (80, 80, 80))
                surface.blit(info_t, (screen_rect.x + 5, screen_rect.y + 22))
                qlen = len(node.queue)
                if qlen > 0:
                    q_t = self.font_sm.render(f"q:{qlen}", True, RED if qlen > 5 else (100, 100, 100))
                    surface.blit(q_t, (screen_rect.x + 5, screen_rect.y + 37))
            else:
                name_t = self.font_sm.render(nid, True, (120, 120, 120))
                surface.blit(name_t, (screen_rect.x + 5, screen_rect.y + 18))

        # Flow shapes (already in virtual coords)
        for fs in self._flow_shapes:
            fs.draw_offset(surface, self._viewport_x - GRAPH_X, self._viewport_y - GRAPH_Y)

        # Floating money text
        for ft in self._floating_texts:
            ft.draw_offset(surface, self.font, self._viewport_x - GRAPH_X, self._viewport_y - GRAPH_Y)

        # Connecting mode indicator
        if self.state == CONNECTING and self._connect_from:
            msg = self.font.render(f"Click target node (from {self._connect_from})", True, ACCENT)
            surface.blit(msg, (GRAPH_PAD, GRAPH_Y + GRAPH_H - 25))

        # Remove clip
        surface.set_clip(None)

        # Radial context menu (drawn after clip removed so it can extend outside graph)
        if self.state == IDLE:
            self._draw_radial_menu(surface)
```

- [ ] **Step 3: Add `draw_offset` methods to FlowShape and FloatingText**

Add to `FlowShape` class:

```python
    def draw_offset(self, surface, off_x, off_y):
        if not self.alive or self.t < 0:
            return
        frac = min(self.t / self.duration, 1.0)
        cx = self.sx + (self.ex - self.sx) * frac - off_x
        cy = self.sy + (self.ey - self.sy) * frac - off_y
        s = THUMB_SIZE + 6
        border_rect = pygame.Rect(int(cx - s // 2), int(cy - s // 2), s, s)
        pygame.draw.rect(surface, self.border, border_rect, border_radius=3)
        inner = pygame.Rect(border_rect.x + 2, border_rect.y + 2, THUMB_SIZE + 2, THUMB_SIZE + 2)
        surface.blit(self.thumb, inner)
```

Add to `FloatingText` class:

```python
    def draw_offset(self, surface, font, off_x, off_y):
        if not self.alive:
            return
        alpha = max(0, int(255 * (1 - self.t / self.duration)))
        txt = font.render(self.text, True, self.color)
        txt.set_alpha(alpha)
        surface.blit(txt, (int(self.x - off_x), int(self.y - off_y)))
```

- [ ] **Step 4: Update radial menu to use screen coords**

The radial menu position computation uses `rect.centerx/y` which are now virtual coords. Update `_draw_radial_menu` and `_handle_idle_click` radial button check to translate:

In `_draw_radial_menu`, change the `cx, cy` computation:

```python
        cx = self._vx(rect.centerx)
        cy = self._vy(rect.centery)
```

In `_handle_idle_click`, same change for the radial button position calculation:

```python
                cx = self._vx(rect.centerx)
                cy = self._vy(rect.centery)
```

- [ ] **Step 5: Update hit testing to translate mouse coords to virtual**

In `_handle_idle_click`, after the radial menu check, translate mouse position to virtual coords for node rect collision:

```python
        # 2. Check if clicked a node rect — translate to virtual coords
        vmx = mx + self._viewport_x - GRAPH_X
        vmy = my + self._viewport_y - GRAPH_Y
        for nid, rect in self._node_rects.items():
            if rect.collidepoint(vmx, vmy):
                self.selected_node = nid
                return
```

In `_handle_connecting_click`, same translation:

```python
        mx, my = self._click_pos
        vmx = mx + self._viewport_x - GRAPH_X
        vmy = my + self._viewport_y - GRAPH_Y
        for nid, rect in self._node_rects.items():
            if rect.collidepoint(vmx, vmy) and nid != self._connect_from:
                ...
```

- [ ] **Step 6: Test the game with viewport-offset drawing**

```bash
.venv/bin/python -m app.main
```

Add nodes, connect them, verify:
- Nodes render correctly (not shifted weirdly)
- Clicking nodes selects them
- Radial menu appears at the correct position
- Connecting nodes works
- Flow animations appear correctly on edges

- [ ] **Step 7: Commit**

```bash
git add app/ui/factory_floor.py
git commit -m "feat: add viewport-offset drawing for force-directed layout"
```

---

### Task 4: Minimap

**Files:**
- Modify: `app/ui/factory_floor.py`

The minimap is a real scaled-down thumbnail of the graph in the top-right corner. It shows all nodes, edges, bins, and a viewport rectangle. Click/drag to pan.

- [ ] **Step 1: Add minimap constants and transform methods**

Add constants near the top of the file:

```python
MINIMAP_MAX_W = 160
MINIMAP_MAX_H = 120
MINIMAP_PAD = 8
```

Add methods to `FactoryFloorUI`:

```python
    def _minimap_params(self) -> tuple[float, float, float, float, float] | None:
        """Return (scale, mm_x, mm_y, mm_w, mm_h) or None if minimap not needed."""
        # Hide minimap when entire graph fits in viewport
        if self._canvas_w <= GRAPH_W and self._canvas_h <= GRAPH_H:
            return None
        scale = min(MINIMAP_MAX_W / self._canvas_w, MINIMAP_MAX_H / self._canvas_h)
        mm_w = self._canvas_w * scale
        mm_h = self._canvas_h * scale
        mm_x = GRAPH_X + GRAPH_W - mm_w - MINIMAP_PAD
        mm_y = GRAPH_Y + MINIMAP_PAD
        return (scale, mm_x, mm_y, mm_w, mm_h)

    def _virtual_to_minimap(self, vx: float, vy: float,
                            scale: float, mm_x: float, mm_y: float) -> tuple[int, int]:
        """Convert virtual canvas coords to minimap pixel coords."""
        mx = mm_x + (vx - self._canvas_x) * scale
        my = mm_y + (vy - self._canvas_y) * scale
        return (int(mx), int(my))

    def _minimap_to_virtual(self, mx: float, my: float,
                            scale: float, mm_x: float, mm_y: float) -> tuple[float, float]:
        """Convert minimap pixel coords to virtual canvas coords."""
        vx = self._canvas_x + (mx - mm_x) / scale
        vy = self._canvas_y + (my - mm_y) / scale
        return (vx, vy)
```

- [ ] **Step 2: Add `_draw_minimap()` method**

```python
    def _draw_minimap(self, surface):
        """Draw the minimap thumbnail in the top-right corner."""
        params = self._minimap_params()
        if not params:
            return
        scale, mm_x, mm_y, mm_w, mm_h = params
        v2m = lambda vx, vy: self._virtual_to_minimap(vx, vy, scale, mm_x, mm_y)

        # Semi-transparent background
        bg_surf = pygame.Surface((int(mm_w), int(mm_h)), pygame.SRCALPHA)
        bg_surf.fill((240, 235, 225, 200))
        surface.blit(bg_surf, (int(mm_x), int(mm_y)))

        # Border
        pygame.draw.rect(surface, (100, 100, 100),
                         (int(mm_x), int(mm_y), int(mm_w), int(mm_h)), 1)

        graph = self.world.graph

        # Draw edges
        for nid, node in graph.nodes.items():
            src_rect = self._node_rects.get(nid)
            if not src_rect:
                continue
            sx, sy = v2m(src_rect.centerx, src_rect.centery)
            for edge in node.edges:
                if edge.target.startswith("BIN:"):
                    key = f"{nid}:{edge.target}"
                    if key in self._bin_positions:
                        bx, by = self._bin_positions[key]
                        ex, ey = v2m(bx + 30, by + 10)
                        pygame.draw.line(surface, (150, 150, 150), (sx, sy), (ex, ey), 1)
                elif edge.target in self._node_rects:
                    dst_rect = self._node_rects[edge.target]
                    ex, ey = v2m(dst_rect.centerx, dst_rect.centery)
                    pygame.draw.line(surface, (150, 150, 150), (sx, sy), (ex, ey), 1)

        # Draw nodes as small rects
        for nid, rect in self._node_rects.items():
            nx, ny = v2m(rect.x, rect.y)
            nw = max(3, int(rect.w * scale))
            nh = max(2, int(rect.h * scale))
            color = NODE_SELECTED if nid == self.selected_node else NODE_FILL
            pygame.draw.rect(surface, color, (nx, ny, nw, nh))
            pygame.draw.rect(surface, (0, 0, 0), (nx, ny, nw, nh), 1)

        # Draw bins as green dots
        for key, (bx, by) in self._bin_positions.items():
            mx, my = v2m(bx, by)
            pygame.draw.circle(surface, (60, 120, 60), (mx, my), 2)

        # Viewport rectangle
        vp_x1, vp_y1 = v2m(self._viewport_x, self._viewport_y)
        vp_x2, vp_y2 = v2m(self._viewport_x + GRAPH_W, self._viewport_y + GRAPH_H)
        pygame.draw.rect(surface, ACCENT,
                         (vp_x1, vp_y1, vp_x2 - vp_x1, vp_y2 - vp_y1), 2)
```

- [ ] **Step 3: Call `_draw_minimap()` from `_draw_graph()`**

At the end of `_draw_graph()`, after `surface.set_clip(None)` and after the radial menu:

```python
        # Minimap (drawn in screen coords, after clip removed)
        self._draw_minimap(surface)
```

- [ ] **Step 4: Add minimap click handling**

Add a method and integrate into `_handle_idle_click`. The minimap check should be the FIRST thing in `_handle_idle_click`, before radial menu and node checks:

```python
    def _handle_minimap_click(self, mx, my) -> bool:
        """Handle click on minimap. Returns True if click was consumed."""
        params = self._minimap_params()
        if not params:
            return False
        scale, mm_x, mm_y, mm_w, mm_h = params
        if not (mm_x <= mx <= mm_x + mm_w and mm_y <= my <= mm_y + mm_h):
            return False
        # Convert click to virtual coords and center viewport there
        cvx, cvy = self._minimap_to_virtual(mx, my, scale, mm_x, mm_y)
        self._viewport_x = cvx - GRAPH_W / 2
        self._viewport_y = cvy - GRAPH_H / 2
        self._clamp_viewport()
        return True
```

At the start of `_handle_idle_click`, before the radial menu check:

```python
        # 0. Check minimap click first
        if self._handle_minimap_click(mx, my):
            return
```

- [ ] **Step 5: Add minimap drag support in `update()`**

In the `update()` method, handle mouse drag on the minimap by checking if mouse button is held:

```python
        # Minimap drag — check on every frame when mouse is held
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            params = self._minimap_params()
            if params:
                scale, mm_x, mm_y, mm_w, mm_h = params
                if mm_x <= mx <= mm_x + mm_w and mm_y <= my <= mm_y + mm_h:
                    cvx, cvy = self._minimap_to_virtual(mx, my, scale, mm_x, mm_y)
                    self._viewport_x = cvx - GRAPH_W / 2
                    self._viewport_y = cvy - GRAPH_H / 2
                    self._clamp_viewport()
```

- [ ] **Step 6: Test the minimap**

```bash
.venv/bin/python -m app.main
```

Test:
- Add 5+ nodes with connections so the graph is larger than viewport
- Verify minimap appears in top-right corner
- Verify minimap shows node thumbnails, edges, and viewport rectangle
- Click on minimap to pan the viewport
- Drag on minimap to continuously pan
- Verify minimap hides when graph is small enough to fit in viewport
- Verify radial menu, node selection, and connecting still work alongside minimap

- [ ] **Step 7: Commit**

```bash
git add app/ui/factory_floor.py
git commit -m "feat: add minimap thumbnail with click-to-pan navigation"
```

---

### Task 5: Flow Animation Viewport Integration

**Files:**
- Modify: `app/ui/factory_floor.py`

The flow shape spawn positions (`_spawn_flow_shapes`) use `_node_rects` and `_bin_positions`, which are now in virtual canvas coordinates. The spawn code is already correct (it stores virtual coords in FlowShape). We just need to ensure the `draw_offset` methods from Task 3 are being called correctly and the entry animation (objects entering from the left edge) uses viewport-aware coordinates.

- [ ] **Step 1: Fix entry animation to use viewport-relative left edge**

In `_spawn_flow_shapes`, the entry point for objects entering the system should be at the left edge of the viewport, not GRAPH_X:

```python
            entry_x = self._viewport_x
            entry = (entry_x, root_rect.centery + int(y_jitter))
```

- [ ] **Step 2: Fix dropped object animation**

Dropped objects fall from a node — their positions are already in virtual coords from `src_rect`, so this should work automatically. Verify it does.

- [ ] **Step 3: Test flow animations**

```bash
.venv/bin/python -m app.main
```

- Set up a routing tree with at least one bin
- Train a worker and run the factory
- Verify flow shapes animate along edges correctly
- Pan the viewport and verify flow shapes move with the graph
- Verify floating money text (+$15, -$8) appears at correct positions

- [ ] **Step 4: Commit**

```bash
git add app/ui/factory_floor.py
git commit -m "fix: flow animation entry point uses viewport-relative coords"
```

---

### Task 6: Polish and Parameter Tuning

**Files:**
- Modify: `app/ui/graph_layout.py` — tune parameters
- Modify: `app/ui/factory_floor.py` — any final fixes

- [ ] **Step 1: Test with various graph topologies**

```bash
.venv/bin/python -m app.main
```

Test these scenarios and note any layout issues:
1. Single root node, no connections
2. Root → 2 children (binary router)
3. Root → child → 3 bins (chain with bins)
4. Root → 3 children, each with 2 bins (wide tree)
5. 6+ nodes in a deep chain
6. Disconnected/orphan nodes

- [ ] **Step 2: Tune force parameters if needed**

Adjust values in `app/ui/graph_layout.py` based on testing. Common issues and fixes:
- Nodes too close: increase `REPULSION_STRENGTH` (more negative)
- Nodes too spread: increase `CENTER_Y_STRENGTH`
- Edge labels don't fit: increase `IDEAL_EDGE_LEN`
- Slow convergence: increase `ALPHA_DECAY`
- Jittery layout: increase `VELOCITY_DECAY`

- [ ] **Step 3: Clean up any dead code from old tree layout**

Remove any remaining references to the old tree layout that are no longer used (old constants, old helper functions).

- [ ] **Step 4: Final test**

Run the full game loop: add nodes, connect to nodes and bins, train workers, run the factory, verify everything works together with the force-directed layout and minimap.

- [ ] **Step 5: Commit**

```bash
git add app/ui/graph_layout.py app/ui/factory_floor.py
git commit -m "fix: tune force layout parameters and clean up dead code"
```
