# Force-Directed DAG Layout with Minimap

## Problem

The tree-based layout in `_layout_graph()` uses rigid positional math that breaks down with complex graphs: nodes overlap, bins overlap child nodes, edge labels get clipped. Fundamentally, packing rectangular nodes into a fixed-size area without overlap is not solvable with static arithmetic for arbitrary topologies.

## Solution

Replace the tree layout with a force-directed simulation that runs on topology change. Add a virtual canvas (larger than the viewport) with a minimap thumbnail for navigation.

## Architecture

### Force Simulation Engine

A new module `app/ui/graph_layout.py` containing the simulation logic, decoupled from rendering.

**Data structures:**

```python
@dataclass
class SimNode:
    node_id: str
    x: float          # position in virtual canvas
    y: float
    vx: float = 0.0   # velocity
    vy: float = 0.0
    depth: int = 0     # BFS depth from root
    # bins are not SimNodes — they're positioned post-simulation

class ForceLayout:
    nodes: list[SimNode]
    edges: list[tuple[str, str]]  # (source_id, target_id) — non-bin edges only
    # ... methods for running simulation
```

**Forces (applied in order each tick):**

| Force | Purpose | Key parameter |
|-------|---------|---------------|
| Depth X-positioning | Maintains left-to-right DAG flow | `target_x = depth * IDEAL_EDGE_LEN`, strength 0.3 |
| Y-centering | Weak pull to vertical center | strength 0.02 |
| Repulsion | Pushes all node pairs apart | strength -300, cutoff 500px |
| Edge springs | Pulls connected nodes toward ideal distance | distance 200px, strength 1/min(degree_src, degree_tgt) |
| Rectangular collision | Prevents node rect overlap | padding 20px around each node |

**Simulation loop:**

```
initialize node positions (phyllotaxis or depth-based grid)
alpha = 1.0
for i in range(MAX_ITERS):        # start with 150, tune
    alpha += (0 - alpha) * ALPHA_DECAY   # ~0.05
    if alpha < ALPHA_MIN: break          # 0.005
    
    apply depth positioning force (modify vx)
    apply y-centering force (modify vy)
    apply repulsion force (modify vx, vy for all pairs)
    apply edge spring force (modify vx, vy for linked pairs)
    apply rectangular collision (push overlapping rects apart)
    
    max_displacement = 0
    for node in nodes:
        node.vx *= (1 - VELOCITY_DECAY)   # 0.4
        node.vy *= (1 - VELOCITY_DECAY)
        dx = node.vx * alpha
        dy = node.vy * alpha
        node.x += dx
        node.y += dy
        max_displacement = max(max_displacement, abs(dx) + abs(dy))
    
    if max_displacement < 1.0: break  # converged
```

**Collision detection (rectangular):**

For each pair of nodes, check if their rects (with padding) overlap. If they do, compute the overlap on each axis and push apart along the axis of minimum overlap. This prevents the "tunneling" that circle-based collision gets with rectangles.

```
padded_w = NODE_W + COLLISION_PAD
padded_h = NODE_H + COLLISION_PAD
dx = node_b.x - node_a.x
dy = node_b.y - node_a.y
overlap_x = padded_w - abs(dx)
overlap_y = padded_h - abs(dy)
if overlap_x > 0 and overlap_y > 0:
    if overlap_x < overlap_y:
        push on x-axis by overlap_x/2 each
    else:
        push on y-axis by overlap_y/2 each
```

**Bin positioning (post-simulation):**

Bins are NOT force-simulated. After convergence:
- For each node with bin edges: place bins at `(node.right + 60, node.centery +/- spread)`
- Vertically spread bins around node center using `BIN_LINE_H = 28`
- No child-node-awareness needed because the simulation already separated nodes

**Initial positions:**

On first layout or when nodes are added: place nodes in a depth-based grid as starting positions. `x = depth * IDEAL_EDGE_LEN`, `y` spaced evenly per depth level. This gives the simulation a good starting point and reduces iterations needed.

### Virtual Canvas & Viewport

**Virtual canvas:**
- After simulation converges, compute bounding box of all node rects + bin positions
- Add margin (e.g., 60px each side)
- Canvas minimum size = graph area (769x673). Grows beyond that as needed.
- Stored as `canvas_x, canvas_y, canvas_w, canvas_h`

**Viewport state:**
- `viewport_x, viewport_y` — top-left corner of the visible window in virtual canvas coordinates
- Initialized to show the root node area
- Clamped so viewport doesn't go outside canvas bounds

**Drawing transform:**
- All node/edge/bin drawing: `screen_x = virtual_x - viewport_x`, `screen_y = virtual_y - viewport_y`
- Clip to graph area rect before drawing (pygame.set_clip)

**Hit testing transform:**
- Mouse click at `(screen_x, screen_y)` → virtual coords `(screen_x + viewport_x, screen_y + viewport_y)`
- Then check against `_node_rects` in virtual coords

### Minimap

**The single hardest subproblem.** The minimap is a real scaled-down thumbnail of the graph, not abstract dots.

**Location:** Top-right corner of graph area, inset by 8px. Semi-transparent background.

**Size:** Dynamic aspect ratio matching the virtual canvas, capped at 160px wide and 120px tall. Whichever dimension hits the cap first determines the scale; the other dimension scales proportionally.

```python
scale = min(MINIMAP_MAX_W / canvas_w, MINIMAP_MAX_H / canvas_h)
minimap_w = canvas_w * scale
minimap_h = canvas_h * scale
minimap_x = GRAPH_X + GRAPH_W - minimap_w - 8
minimap_y = GRAPH_Y + 8
```

**Single source of truth for coordinate transforms:**

Two transform functions encapsulate ALL coordinate mapping:

```python
def virtual_to_minimap(self, vx, vy) -> tuple[int, int]:
    """Convert virtual canvas coords to minimap pixel coords."""
    mx = minimap_x + (vx - canvas_x) * scale
    my = minimap_y + (vy - canvas_y) * scale
    return (int(mx), int(my))

def minimap_to_virtual(self, mx, my) -> tuple[float, float]:
    """Convert minimap pixel coords to virtual canvas coords."""
    vx = canvas_x + (mx - minimap_x) / scale
    vy = canvas_y + (my - minimap_y) / scale
    return (vx, vy)
```

Every piece of code that maps between minimap and virtual canvas MUST use these two functions. No inline math.

**Rendering the minimap:**

1. Draw semi-transparent background rect
2. For each node rect in `_node_rects`: transform to minimap coords via `virtual_to_minimap`, draw a small filled rect (scaled down but at least 3x2px)
3. For each edge: draw a thin line between source/target minimap positions
4. For bins: draw tiny green dots at minimap-transformed bin positions
5. Draw viewport rectangle: transform `(viewport_x, viewport_y)` and `(viewport_x + GRAPH_W, viewport_y + GRAPH_H)` to minimap coords, draw outline rect (bright color, 1-2px)

**Minimap interaction:**

- On mouse click within minimap bounds: convert click to virtual coords via `minimap_to_virtual`, center viewport on that point (clamped to canvas bounds)
- On mouse drag within minimap: continuously update viewport center
- Minimap hidden when entire graph fits in viewport (canvas size <= graph area)

### Integration with Existing Code

**Files changed:**

| File | Change |
|------|--------|
| `app/ui/graph_layout.py` | NEW — force simulation engine |
| `app/ui/factory_floor.py` | Replace `_layout_graph()`, modify `_draw_graph()` for viewport offset, add minimap drawing/interaction, update hit testing |

**`_layout_graph()` replacement:**

```python
def _layout_graph(self):
    graph = self.world.graph
    layout = ForceLayout(graph)
    layout.run()  # runs simulation to convergence
    
    # Copy results to _node_rects and _bin_positions
    for sim_node in layout.nodes:
        self._node_rects[sim_node.node_id] = pygame.Rect(...)
    
    # Position bins post-simulation
    self._position_bins(graph, layout)
    
    # Compute virtual canvas bounds
    self._compute_canvas_bounds()
    
    # Auto-center viewport on root
    if not self._viewport_initialized:
        self._center_viewport_on_root()
        self._viewport_initialized = True
    
    self._layout_dirty = False
```

**`_draw_graph()` changes:**

- Set clip rect to graph area before drawing
- Offset all coordinates by `(-viewport_x, -viewport_y)` when drawing nodes, edges, bins, flow shapes
- Draw minimap last (on top, no viewport offset — it's in screen coords)
- Unset clip rect after drawing

**Click handling changes:**

- `_handle_idle_click`: check minimap bounds first, then translate remaining clicks to virtual coords
- `_handle_connecting_click`: translate to virtual coords
- Radial menu: compute positions in screen space relative to the viewport-adjusted node rect

**Flow animation:**

- `FlowShape` positions stored in virtual canvas coords
- Drawing applies viewport offset

### Performance

- O(n^2) repulsion is fine for n <= 20-30 nodes typical in this game
- ~50-150 iterations, early exit on convergence
- Simulation runs ONCE on topology change, not per frame
- Minimap rendering is cheap (tiny rects and lines)
- Drawing with viewport offset is just addition — negligible cost

### Parameters (starting values, to be tuned)

```python
IDEAL_EDGE_LEN = 200    # target edge length (room for labels)
ALPHA_DECAY = 0.05      # cooling rate
ALPHA_MIN = 0.005       # convergence threshold
VELOCITY_DECAY = 0.4    # friction
MAX_ITERS = 150         # hard cap
REPULSION_STRENGTH = -300
DEPTH_X_STRENGTH = 0.3
CENTER_Y_STRENGTH = 0.02
COLLISION_PAD = 20      # padding around node rects
MINIMAP_MAX_W = 160
MINIMAP_MAX_H = 120
```
