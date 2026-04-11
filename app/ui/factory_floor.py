"""Factory floor UI — graph editor, training, and live simulation view."""

import math
import random
import pygame
from collections import deque

from app.ui.drawing_canvas import DrawingCanvas, CANVAS_SIZE


def surface_to_tensor(surface):
    """Convert a pygame surface to a (3, H, W) normalized float tensor."""
    import torch, numpy as np
    arr = pygame.surfarray.array3d(surface)  # (W, H, 3)
    arr = arr.transpose(1, 0, 2)  # (H, W, 3)
    tensor = torch.from_numpy(arr.copy()).float() / 255.0
    return tensor.permute(2, 0, 1)  # (3, H, W)

# ---------------------------------------------------------------------------
# Layout constants (1024x768)
# ---------------------------------------------------------------------------
W, H = 1024, 768

TOP_H = 45
BOTTOM_H = 50
SIDE_W = 255

GRAPH_X = 0
GRAPH_Y = TOP_H
GRAPH_W = W - SIDE_W
GRAPH_H = H - TOP_H - BOTTOM_H  # 673

SIDE_X = W - SIDE_W  # 769
SIDE_Y = TOP_H

NODE_W = 140
NODE_H = 50
LEVEL_GAP_X = 180  # horizontal distance between tree levels
NODE_GAP_Y = 12    # vertical gap between nodes at same level
GRAPH_PAD = 30      # padding inside graph area

# Side panel button layout
BTN_W = 225
BTN_H = 32
BTN_X = SIDE_X + 15
BTN_GAP = 6

# Colors
BG = (240, 235, 225)
PANEL_BG = (228, 224, 216)
TOP_BG = (60, 70, 90)
BOTTOM_BG = (60, 70, 90)
NODE_FILL = (215, 225, 240)
NODE_SELECTED = (255, 238, 195)
NODE_EMPTY = (200, 200, 200)
BIN_FILL = (200, 230, 200)
EDGE_COLOR = (100, 100, 100)
TEXT_LIGHT = (230, 230, 230)
TEXT_DARK = (30, 30, 30)
ACCENT = (70, 130, 200)
GREEN = (60, 170, 60)
RED = (200, 60, 60)
SEPARATOR = (190, 185, 175)

FLOW_SPEED = 150       # pixels per second
THUMB_SIZE = 26        # shape thumbnail size
BORDER_CORRECT = (40, 200, 40)
BORDER_WRONG = (220, 50, 50)
BORDER_TRANSIT = (180, 180, 100)


def _tensor_to_thumb(tensor):
    """Convert a (3,84,84) float tensor to a small pygame Surface."""
    import numpy as np
    arr = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
    return pygame.transform.smoothscale(surf, (THUMB_SIZE, THUMB_SIZE))


class FlowShape:
    """An animated shape thumbnail that moves along an edge."""

    __slots__ = ("sx", "sy", "ex", "ey", "thumb", "border", "t", "duration", "alive")

    def __init__(self, start, end, thumb_surface, border_color=BORDER_TRANSIT):
        self.sx, self.sy = float(start[0]), float(start[1])
        self.ex, self.ey = float(end[0]), float(end[1])
        self.thumb = thumb_surface
        self.border = border_color
        self.t = 0.0
        dist = math.hypot(self.ex - self.sx, self.ey - self.sy)
        self.duration = max(0.2, dist / FLOW_SPEED)
        self.alive = True

    def update(self, dt):
        if self.t < 0:
            self.t += dt  # still in stagger delay
            return
        self.t += dt / self.duration
        if self.t >= 1.0:
            self.alive = False

    @property
    def pos(self):
        p = max(0.0, min(self.t, 1.0))
        return (self.sx + (self.ex - self.sx) * p,
                self.sy + (self.ey - self.sy) * p)

    def draw(self, surface):
        if not self.alive or self.t < 0:
            return
        x, y = self.pos
        half = THUMB_SIZE // 2
        ix, iy = int(x) - half, int(y) - half
        surface.blit(self.thumb, (ix, iy))
        pygame.draw.rect(surface, self.border,
                         (ix - 1, iy - 1, THUMB_SIZE + 2, THUMB_SIZE + 2), 2)


# States
IDLE = 0
CONNECTING = 1
TRAINING = 2
DIALOG_TEXT = 3
DIALOG_SELECT = 4


class FactoryFloorUI:
    def __init__(self, world, generator):
        self.world = world
        self.gen = generator
        self.state = IDLE
        self.selected_node = None  # node_id or None
        self.paused = True  # start paused so player can build first
        self._status_msg = ""
        self._status_timer = 0.0
        self.tick_timer = 0.0
        self.tick_interval = 1.5  # seconds between ticks — one shape at a time

        # Fonts (initialized on first draw since pygame must be init'd)
        self._fonts_ready = False
        self.font = None
        self.font_sm = None
        self.font_lg = None

        # Graph layout cache
        self._node_rects: dict[str, pygame.Rect] = {}
        self._bin_positions: dict[str, tuple[int, int]] = {}
        self._layout_dirty = True

        # Flow animation
        self._flow_shapes: list[FlowShape] = []

        # Connecting state
        self._connect_from = None

        # Training state
        self._canvas = DrawingCanvas()
        self._training_worker = None
        self._training_label_idx = 0
        self._training_new_class_input = ""
        self._training_typing_new_class = False

        # Dialog state
        self._dialog_title = ""
        self._dialog_input = ""
        self._dialog_callback = None
        self._dialog_options: list[str] = []
        self._dialog_result = None

        # Pending actions (for chained dialogs)
        self._pending_action = None
        self._pending_data = {}

        # Click tracking
        self._click_pos = None

    def _init_fonts(self):
        if not self._fonts_ready:
            self.font = pygame.font.SysFont("Arial", 16)
            self.font_sm = pygame.font.SysFont("Arial", 13)
            self.font_lg = pygame.font.SysFont("Arial", 22, bold=True)
            self._fonts_ready = True

    # ------------------------------------------------------------------
    # Graph auto-layout
    # ------------------------------------------------------------------

    def _layout_graph(self):
        """Compute node positions using a tree layout that accounts for
        each node's vertical footprint (bins, children, subtree size).
        """
        self._node_rects.clear()
        self._bin_positions.clear()

        graph = self.world.graph
        if not graph.root_id or graph.root_id not in graph.nodes:
            for i, nid in enumerate(graph.nodes):
                x = GRAPH_PAD
                y = GRAPH_Y + GRAPH_PAD + i * (NODE_H + NODE_GAP_Y)
                self._node_rects[nid] = pygame.Rect(x, y, NODE_W, NODE_H)
            self._layout_dirty = False
            return

        # --- Step 1: BFS to assign tree levels ---
        levels: dict[str, int] = {}
        children: dict[str, list[str]] = {}  # node -> child node ids
        q = deque([(graph.root_id, 0)])
        visited = set()
        max_level = 0

        while q:
            nid, level = q.popleft()
            if nid in visited:
                continue
            visited.add(nid)
            levels[nid] = level
            max_level = max(max_level, level)
            children[nid] = []

            node = graph.nodes[nid]
            for edge in node.edges:
                t = edge.target
                if not t.startswith("BIN:") and t in graph.nodes and t not in visited:
                    children[nid].append(t)
                    q.append((t, level + 1))

        for nid in graph.nodes:
            if nid not in levels:
                max_level += 1
                levels[nid] = max_level
                children.setdefault(nid, [])

        # --- Step 2: Compute vertical footprint for each node ---
        # Footprint = max(node_height, bins_height, sum_of_children_footprints)
        BIN_LINE_H = 28  # vertical space per bin label
        MIN_NODE_FOOT = NODE_H + NODE_GAP_Y

        footprint: dict[str, float] = {}

        def calc_footprint(nid):
            if nid in footprint:
                return footprint[nid]
            node = graph.nodes.get(nid)
            n_bins = len([e for e in node.edges if e.target.startswith("BIN:")]) if node else 0

            # Space needed for this node's bins
            bins_h = max(0, n_bins) * BIN_LINE_H

            # Space needed for children subtrees
            child_ids = children.get(nid, [])
            children_h = 0
            for cid in child_ids:
                children_h += calc_footprint(cid)
            if child_ids:
                children_h += (len(child_ids) - 1) * NODE_GAP_Y

            footprint[nid] = max(MIN_NODE_FOOT, bins_h, children_h)
            return footprint[nid]

        calc_footprint(graph.root_id)
        for nid in graph.nodes:
            if nid not in footprint:
                calc_footprint(nid)

        # --- Step 3: Compute horizontal positions ---
        # Scale levels to fit in graph area
        n_levels = max_level + 1
        available_w = GRAPH_W - 2 * GRAPH_PAD - NODE_W
        level_gap = min(LEVEL_GAP_X, available_w // max(1, n_levels)) if n_levels > 1 else 0

        # --- Step 4: Assign vertical positions top-down ---
        # Root gets centered; children placed relative to parent
        total_root_foot = footprint.get(graph.root_id, MIN_NODE_FOOT)
        available_h = GRAPH_H - 2 * GRAPH_PAD
        scale = min(1.0, available_h / total_root_foot) if total_root_foot > 0 else 1.0

        def place_node(nid, x, y_start, y_end):
            """Place nid centered in [y_start, y_end], then recurse into children."""
            mid_y = (y_start + y_end) / 2
            self._node_rects[nid] = pygame.Rect(
                int(x), int(mid_y - NODE_H / 2), NODE_W, NODE_H
            )

            node = graph.nodes.get(nid)
            if not node:
                return

            child_ids = children.get(nid, [])
            if child_ids:
                child_x = x + level_gap
                total_cf = sum(footprint[c] for c in child_ids)
                total_cf += (len(child_ids) - 1) * NODE_GAP_Y
                child_top = mid_y - total_cf * scale / 2
                for cid in child_ids:
                    cf = footprint[cid] * scale
                    place_node(cid, child_x, child_top, child_top + cf)
                    child_top += cf + NODE_GAP_Y * scale

        root_top = GRAPH_Y + GRAPH_PAD + (available_h - total_root_foot * scale) / 2
        root_bot = root_top + total_root_foot * scale
        place_node(graph.root_id, GRAPH_PAD, root_top, root_bot)

        # Place orphan nodes (not reachable from root)
        orphan_y = GRAPH_Y + GRAPH_PAD
        for nid in graph.nodes:
            if nid not in self._node_rects:
                self._node_rects[nid] = pygame.Rect(GRAPH_PAD, int(orphan_y), NODE_W, NODE_H)
                orphan_y += MIN_NODE_FOOT

        # --- Step 5: Position bins ---
        for nid, node in graph.nodes.items():
            bin_edges = [e for e in node.edges if e.target.startswith("BIN:")]
            if not bin_edges:
                continue
            src_rect = self._node_rects.get(nid)
            if not src_rect:
                continue
            n = len(bin_edges)
            total_h = (n - 1) * BIN_LINE_H
            start_y = src_rect.centery - total_h / 2
            for i, edge in enumerate(bin_edges):
                bx = src_rect.right + 130
                by = int(start_y + i * BIN_LINE_H)
                self._bin_positions[f"{nid}:{edge.target}"] = (bx, by)

        # --- Step 6: Clamp everything to graph area ---
        min_y = GRAPH_Y + 5
        max_y = GRAPH_Y + GRAPH_H - NODE_H - 5
        for nid, rect in self._node_rects.items():
            rect.y = max(min_y, min(max_y, rect.y))
        for key, (bx, by) in list(self._bin_positions.items()):
            self._bin_positions[key] = (bx, max(min_y, min(max_y, by)))

        self._layout_dirty = False

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, events, dt):
        self._click_pos = None
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self._click_pos = e.pos

        # Auto-tick (only if factory has a root node)
        if not self.paused and self.world.graph.root_id:
            self.tick_timer += dt
            if self.tick_timer >= self.tick_interval:
                results = self.world.tick()
                self.tick_timer = 0
                self._spawn_flow_shapes(results)

        # Animate flow shapes
        self._flow_shapes = [s for s in self._flow_shapes if s.alive]
        for shape in self._flow_shapes:
            shape.update(dt)

        # Status message timer
        if self._status_timer > 0:
            self._status_timer -= dt

        # State-specific input
        if self.state == DIALOG_TEXT:
            self._handle_text_dialog(events)
        elif self.state == DIALOG_SELECT:
            pass  # handled in draw via buttons
        elif self.state == TRAINING:
            if self._training_typing_new_class:
                for e in events:
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_RETURN and self._training_new_class_input.strip():
                            new_name = self._training_new_class_input.strip()
                            if self._training_worker and new_name not in self._training_worker.class_names:
                                self._training_worker.class_names.append(new_name)
                                self._training_label_idx = len(self._training_worker.class_names) - 1
                            self._training_new_class_input = ""
                            self._training_typing_new_class = False
                        elif e.key == pygame.K_ESCAPE:
                            self._training_new_class_input = ""
                            self._training_typing_new_class = False
                        elif e.key == pygame.K_BACKSPACE:
                            self._training_new_class_input = self._training_new_class_input[:-1]
                        elif e.unicode and len(self._training_new_class_input) < 20:
                            self._training_new_class_input += e.unicode
            else:
                canvas_rect = pygame.Rect(270, 70, CANVAS_SIZE, CANVAS_SIZE)
                self._canvas.handle_events(events, canvas_rect)
        elif self.state == IDLE:
            self._handle_idle_click()
        elif self.state == CONNECTING:
            self._handle_connecting_click()

    def _spawn_flow_shapes(self, results):
        """Spawn animated shape thumbnails showing actual objects on actual paths."""
        graph = self.world.graph
        root_rect = self._node_rects.get(graph.root_id) if graph.root_id else None
        correct_set = set(id(o) for o, _ in results.correct)
        wrong_set = set(id(o) for o, _, _ in results.wrong)

        # Show each actual flow: the real object on the real edge it traveled
        stagger = 0.0
        for obj, from_nid, target_str, prediction in results.flows:
            src_rect = self._node_rects.get(from_nid)
            if not src_rect:
                continue

            thumb = _tensor_to_thumb(obj.tensor)

            # Determine border color based on final outcome
            obj_id = id(obj)
            if obj_id in correct_set:
                border = BORDER_CORRECT
            elif obj_id in wrong_set:
                border = BORDER_WRONG
            else:
                border = BORDER_TRANSIT

            # Find destination position
            if target_str.startswith("BIN:"):
                key = f"{from_nid}:{target_str}"
                if key in self._bin_positions:
                    bx, by = self._bin_positions[key]
                    dest = (bx + 15, by + 8)
                else:
                    continue
            elif target_str in self._node_rects:
                dest = self._node_rects[target_str].midleft
            else:
                continue

            fs = FlowShape(src_rect.midright, dest, thumb, border)
            fs.t = -stagger  # stagger so shapes don't overlap
            stagger += 0.08
            self._flow_shapes.append(fs)

        # Objects entering the system from the left edge
        all_entering = ([o for o, _ in results.correct] +
                        [o for o, _, _ in results.wrong] +
                        results.dropped)
        if root_rect:
            for i, obj in enumerate(all_entering[:5]):
                thumb = _tensor_to_thumb(obj.tensor)
                y_jitter = (i - len(all_entering[:5]) / 2) * (THUMB_SIZE + 4)
                entry = (GRAPH_X + 2, root_rect.centery + int(y_jitter))
                fs = FlowShape(entry, root_rect.midleft, thumb, BORDER_TRANSIT)
                fs.t = -i * 0.06
                self._flow_shapes.append(fs)

        # Dropped objects — fall off from where they were
        for obj, drop_nid in results.dropped_at:
            src_rect = self._node_rects.get(drop_nid)
            if not src_rect:
                continue
            thumb = _tensor_to_thumb(obj.tensor)
            # Fall downward from the node
            start = (src_rect.centerx + random.randint(-20, 20), src_rect.bottom)
            end = (start[0] + random.randint(-10, 10), start[1] + 120)
            fs = FlowShape(start, end, thumb, BORDER_WRONG)
            fs.duration = 0.8  # slow fall
            fs.t = random.uniform(-0.1, 0.0)
            self._flow_shapes.append(fs)

        # Cap to prevent lag
        if len(self._flow_shapes) > 80:
            self._flow_shapes = self._flow_shapes[-60:]

    def _handle_text_dialog(self, events):
        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_RETURN and self._dialog_input.strip():
                    result = self._dialog_input.strip()
                    self._dialog_input = ""
                    self.state = IDLE
                    if self._dialog_callback:
                        self._dialog_callback(result)
                elif e.key == pygame.K_ESCAPE:
                    self._dialog_input = ""
                    self.state = IDLE
                    self._pending_action = None
                elif e.key == pygame.K_BACKSPACE:
                    self._dialog_input = self._dialog_input[:-1]
                elif e.unicode and len(self._dialog_input) < 20:
                    self._dialog_input += e.unicode

    def _handle_idle_click(self):
        if not self._click_pos:
            return
        # Check if clicked a node
        for nid, rect in self._node_rects.items():
            if rect.collidepoint(self._click_pos):
                self.selected_node = nid
                return
        # Clicked empty graph area — deselect
        graph_area = pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)
        if graph_area.collidepoint(self._click_pos):
            self.selected_node = None

    def _handle_connecting_click(self):
        if not self._click_pos:
            return
        for nid, rect in self._node_rects.items():
            if rect.collidepoint(self._click_pos) and nid != self._connect_from:
                output_label = self._pending_data.get("output_label", "???")
                self.world.graph.connect(self._connect_from, output_label, nid)
                self._layout_dirty = True
                self._show_status(f"Connected \"{output_label}\" → {nid}")
                self._connect_from = None
                self.state = IDLE
                return
        # Click outside — cancel
        self.state = IDLE
        self._connect_from = None

    # ------------------------------------------------------------------
    # Actions (triggered by side panel buttons)
    # ------------------------------------------------------------------

    def _action_add_router(self):
        self._pending_action = "add_router"
        self._pending_data = {}
        self._show_text_dialog("Router node name:", self._on_router_name)

    def _on_router_name(self, name):
        self._pending_data["name"] = name
        self._show_text_dialog("Output label A:", self._on_router_label_a)

    def _on_router_label_a(self, label_a):
        self._pending_data["label_a"] = label_a
        self._show_text_dialog("Output label B:", self._on_router_label_b)

    def _on_router_label_b(self, label_b):
        name = self._pending_data["name"]
        label_a = self._pending_data["label_a"]

        worker = self.world.hire_worker(name, binary=True)
        if worker is None:
            self._pending_action = None
            self._show_status(f"Not enough coins! Need {self.world.economy.HIRE_COST:.0f}")
            return

        worker.class_names = [label_a, label_b]

        node_id = f"node_{len(self.world.graph.nodes)}"
        self.world.graph.add_node(
            node_id,
            queue_capacity=10,
        )
        self.world.assign_worker(worker, node_id)

        if len(self.world.graph.nodes) == 1:
            self.world.graph.set_root(node_id)

        self.selected_node = node_id
        self._layout_dirty = True
        self._pending_action = None
        self._show_status(f"Added router '{name}'")

    def _action_add_specialist(self):
        self._pending_action = "add_specialist"
        self._pending_data = {}
        self._show_text_dialog("Specialist node name:", self._on_specialist_name)

    def _on_specialist_name(self, name):
        worker = self.world.hire_worker(name, binary=False)
        if worker is None:
            self._pending_action = None
            self._show_status(f"Not enough coins! Need {self.world.economy.HIRE_COST:.0f}")
            return

        # No preset classes — player defines them during training
        node_id = f"node_{len(self.world.graph.nodes)}"
        self.world.graph.add_node(
            node_id,
            queue_capacity=10,
        )
        self.world.assign_worker(worker, node_id)

        if len(self.world.graph.nodes) == 1:
            self.world.graph.set_root(node_id)

        self.selected_node = node_id
        self._layout_dirty = True
        self._pending_action = None
        self._show_status(f"Added specialist '{name}' — train it to define classes!")

    def _action_connect(self):
        """Connect a worker output to another node. Step 1: pick which output."""
        if not self.selected_node:
            return
        node = self.world.graph.nodes.get(self.selected_node)
        if not node or not node.worker or not node.worker.class_names:
            self._show_status("Train the worker first to define outputs!")
            return
        self._pending_action = "connect_node"
        self._pending_data = {"from_node": self.selected_node}
        self._dialog_title = "Which output to connect?"
        self._dialog_options = list(node.worker.class_names)
        self._pending_data["selected_cats"] = []
        self.state = DIALOG_SELECT

    def _finish_connect_node_step1(self):
        """Step 1 done: user picked which output. Step 2: click target node."""
        selected = self._pending_data.get("selected_cats", [])
        if not selected:
            self.state = IDLE
            self._pending_action = None
            return
        self._pending_data["output_label"] = selected[0]
        self._connect_from = self._pending_data["from_node"]
        self._show_status(f"Click the target node for output \"{selected[0]}\"")
        self.state = CONNECTING
        self._pending_action = None

    def _action_connect_to_bin(self):
        """Connect a worker output to a bin. Step 1: pick which output."""
        if not self.selected_node:
            return
        node = self.world.graph.nodes.get(self.selected_node)
        if not node or not node.worker or not node.worker.class_names:
            self._show_status("Train the worker first to define outputs!")
            return
        self._pending_action = "connect_bin_step1"
        self._pending_data = {"from_node": self.selected_node}
        self._dialog_title = "Which output to route to a bin?"
        self._dialog_options = list(node.worker.class_names)
        self._pending_data["selected_cats"] = []
        self.state = DIALOG_SELECT

    def _finish_connect_bin_step1(self):
        """Step 1 done: user picked which output. Step 2: pick which bin."""
        selected = self._pending_data.get("selected_cats", [])
        if not selected:
            self.state = IDLE
            self._pending_action = None
            return
        self._pending_data["output_label"] = selected[0]
        self._pending_data["selected_cats"] = []
        self._pending_action = "connect_bin_step2"
        self._dialog_title = f"Route \"{selected[0]}\" to which bin?"
        self._dialog_options = list(self.world.active_categories)
        self.state = DIALOG_SELECT

    def _finish_connect_bin_step2(self):
        """Step 2 done: create the edge from output to bin."""
        selected = self._pending_data.get("selected_cats", [])
        output_label = self._pending_data.get("output_label")
        from_node = self._pending_data.get("from_node")
        if selected and output_label and from_node:
            bin_cat = selected[0]
            self.world.graph.connect(from_node, output_label, f"BIN:{bin_cat}")
            self._layout_dirty = True
            self._show_status(f"Connected \"{output_label}\" → [{bin_cat}] bin")
        self._pending_action = None
        self.state = IDLE

    def _action_set_root(self):
        if self.selected_node:
            self.world.graph.set_root(self.selected_node)

    def _action_remove_node(self):
        if self.selected_node:
            node = self.world.graph.nodes.get(self.selected_node)
            if node and node.worker:
                self.world.fire_worker(node.worker)
            self.world.graph.remove_node(self.selected_node)
            self.selected_node = None
            self._layout_dirty = True

    def _action_train(self):
        if not self.selected_node:
            return
        node = self.world.graph.nodes.get(self.selected_node)
        if not node or not node.worker:
            return
        self._training_worker = node.worker
        self._training_label_idx = 0
        self._training_typing_new_class = False
        self._training_new_class_input = ""
        self._canvas.clear()
        self.state = TRAINING

    def _train_submit_drawing(self):
        """Submit the current drawing as a training example."""
        if not self._training_worker or not self._training_worker.class_names:
            self._show_status("Add a class first!")
            return
        labels = self._training_worker.class_names
        if self._training_label_idx >= len(labels):
            self._training_label_idx = 0
        label = labels[self._training_label_idx]
        surf84 = self._canvas.get_surface_84()
        tensor = surface_to_tensor(surf84)
        self._training_worker.teach(tensor, label)
        self._canvas.clear()

    def _train_done(self):
        """Finish training and re-estimate accuracy."""
        if self._training_worker:
            worker = self._training_worker

            # Only estimate accuracy if category_mapping is set
            # (maps object categories to worker's class names)
            if worker.category_mapping:
                cats = list(set(worker.category_mapping.keys()))
                valid_cats = [c for c in cats if c in self.gen.ALL_CATEGORIES]
                if valid_cats:
                    test = self.gen.generate_balanced_batch(30, valid_cats)
                    worker.estimate_accuracy(test)

            n_examples = worker.get_support_set_size()
            if not worker.category_mapping:
                self._show_status(f"Trained {n_examples} examples. Connect outputs to set up routing!")
            else:
                self._show_status(f"Trained {n_examples} examples. Accuracy: {worker.cached_accuracy*100:.0f}%")

        self._training_worker = None
        self.state = IDLE

    # ------------------------------------------------------------------
    # Dialog helpers
    # ------------------------------------------------------------------

    def _show_text_dialog(self, title, callback):
        self._dialog_title = title
        self._dialog_input = ""
        self._dialog_callback = callback
        self.state = DIALOG_TEXT

    def _show_status(self, msg, duration=2.5):
        self._status_msg = msg
        self._status_timer = duration

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, surface):
        self._init_fonts()
        if self._layout_dirty:
            self._layout_graph()

        surface.fill(BG)
        self._draw_top_bar(surface)
        self._draw_graph(surface)
        self._draw_side_panel(surface)
        result = self._draw_bottom_bar(surface)

        # Overlays
        if self.state == DIALOG_TEXT:
            self._draw_text_dialog(surface)
        elif self.state == DIALOG_SELECT:
            self._draw_select_dialog(surface)
        elif self.state == TRAINING:
            self._draw_training(surface)

        # Status message (shown in all states except training which draws its own)
        if self.state != TRAINING and self._status_timer > 0:
            st = self.font.render(self._status_msg, True, (255, 200, 80))
            surface.blit(st, (GRAPH_W // 2 - st.get_width() // 2, GRAPH_Y + GRAPH_H - 30))

        return result

    def _draw_top_bar(self, surface):
        pygame.draw.rect(surface, TOP_BG, (0, 0, W, TOP_H))
        # Title
        t = self.font_lg.render("BabyBrain Factory", True, TEXT_LIGHT)
        surface.blit(t, (15, 8))
        # Coins
        coins = self.world.economy.coins
        cpt = self.world.economy.coins_per_tick
        ct = self.font.render(f"${coins:.0f}", True, (255, 220, 80))
        surface.blit(ct, (W // 2 - ct.get_width() // 2, 12))
        # $/tick
        cpt_color = GREEN if cpt > 0 else RED if cpt < 0 else TEXT_LIGHT
        cpt_t = self.font_sm.render(f"{cpt:+.1f} $/tick", True, cpt_color)
        surface.blit(cpt_t, (W // 2 + 60, 15))

    def _draw_graph(self, surface):
        # Background
        pygame.draw.rect(surface, BG, (GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H))

        graph = self.world.graph

        if not graph.nodes:
            msg = self.font.render("No nodes yet. Add a router or specialist.", True, (150, 150, 150))
            surface.blit(msg, (GRAPH_W // 2 - msg.get_width() // 2, GRAPH_Y + GRAPH_H // 2))
            return

        # Draw edges first (behind nodes)
        for nid, node in graph.nodes.items():
            src_rect = self._node_rects.get(nid)
            if not src_rect:
                continue
            for edge in node.edges:
                if edge.target.startswith("BIN:"):
                    key = f"{nid}:{edge.target}"
                    if key in self._bin_positions:
                        bx, by = self._bin_positions[key]
                        # Spread edge start points vertically across the node
                        bin_edges = [e for e in node.edges if e.target.startswith("BIN:")]
                        edge_i = bin_edges.index(edge)
                        n_bins = len(bin_edges)
                        y_spread = (edge_i - (n_bins - 1) / 2) * 10
                        start = (src_rect.right, src_rect.centery + int(y_spread))
                        end = (bx, by + 8)
                        pygame.draw.line(surface, EDGE_COLOR, start, end, 2)
                        # Bin label
                        cat_name = edge.target[4:]
                        bt = self.font_sm.render(f"[{cat_name}]", True, (60, 120, 60))
                        surface.blit(bt, (bx, by))
                        # Edge label: category name centered on line with background
                        label_text = edge.output_label
                        lt = self.font_sm.render(label_text, True, (70, 70, 130))
                        lw, lh = lt.get_size()
                        lx = int(start[0] + (end[0] - start[0]) * 0.5) - lw // 2
                        ly = int(start[1] + (end[1] - start[1]) * 0.5) - lh // 2
                        pygame.draw.rect(surface, BG, (lx - 3, ly - 1, lw + 6, lh + 2))
                        surface.blit(lt, (lx, ly))
                elif edge.target in self._node_rects:
                    dst_rect = self._node_rects[edge.target]
                    # Offset start/end vertically if multiple edges between same pair
                    edges_to_target = [e for e in node.edges
                                       if e.target == edge.target or
                                       (not e.target.startswith("BIN:") and e.target in self._node_rects)]
                    edge_idx = next((i for i, e in enumerate(
                        [e for e in node.edges if not e.target.startswith("BIN:")]) if e is edge), 0)
                    y_off = (edge_idx - len([e for e in node.edges if not e.target.startswith("BIN:")]) / 2) * 12

                    start = (src_rect.right, src_rect.centery + int(y_off))
                    end = (dst_rect.left, dst_rect.centery + int(y_off))
                    pygame.draw.line(surface, EDGE_COLOR, start, end, 2)
                    # Arrow head
                    ax, ay = end
                    pygame.draw.polygon(surface, EDGE_COLOR, [
                        (ax, ay), (ax - 8, ay - 5), (ax - 8, ay + 5)
                    ])
                    # Edge label centered on line with background
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

            # Fill color
            is_selected = (nid == self.selected_node)
            if is_selected:
                fill = NODE_SELECTED
            elif node.worker:
                fill = NODE_FILL
            else:
                fill = NODE_EMPTY

            pygame.draw.rect(surface, fill, rect, border_radius=6)
            pygame.draw.rect(surface, (0, 0, 0), rect, 2, border_radius=6)

            # Root indicator
            if nid == graph.root_id:
                pygame.draw.circle(surface, ACCENT, (rect.left + 10, rect.top + 10), 5)

            # Node text
            if node.worker:
                name_t = self.font_sm.render(node.worker.name[:18], True, TEXT_DARK)
                surface.blit(name_t, (rect.x + 5, rect.y + 5))

                acc = node.worker.cached_accuracy * 100
                spd = node.processing_speed
                info = f"{acc:.0f}% spd={spd}"
                info_t = self.font_sm.render(info, True, (80, 80, 80))
                surface.blit(info_t, (rect.x + 5, rect.y + 22))

                # Queue indicator
                qlen = len(node.queue)
                if qlen > 0:
                    q_t = self.font_sm.render(f"q:{qlen}", True, RED if qlen > 5 else (100, 100, 100))
                    surface.blit(q_t, (rect.x + 5, rect.y + 37))
            else:
                name_t = self.font_sm.render(nid, True, (120, 120, 120))
                surface.blit(name_t, (rect.x + 5, rect.y + 18))

        # Flow shapes
        for fs in self._flow_shapes:
            fs.draw(surface)

        # Connecting mode indicator
        if self.state == CONNECTING and self._connect_from:
            msg = self.font.render(f"Click target node (from {self._connect_from})", True, ACCENT)
            surface.blit(msg, (GRAPH_PAD, GRAPH_Y + GRAPH_H - 25))

    def _draw_side_panel(self, surface):
        panel_rect = pygame.Rect(SIDE_X, SIDE_Y, SIDE_W, GRAPH_H)
        pygame.draw.rect(surface, PANEL_BG, panel_rect)
        pygame.draw.line(surface, SEPARATOR, (SIDE_X, SIDE_Y), (SIDE_X, SIDE_Y + GRAPH_H), 2)

        y = SIDE_Y + 10

        if self.selected_node and self.selected_node in self.world.graph.nodes:
            node = self.world.graph.nodes[self.selected_node]
            y = self._draw_node_info(surface, node, y)
        else:
            t = self.font.render("No node selected", True, (120, 120, 120))
            surface.blit(t, (BTN_X, y))
            y += 30

        # Separator
        y += 5
        pygame.draw.line(surface, SEPARATOR, (BTN_X, y), (BTN_X + BTN_W, y), 1)
        y += 10

        # Global actions
        if self._draw_btn(surface, y, "Add Router Node", (100, 150, 200)):
            self._action_add_router()
        y += BTN_H + BTN_GAP

        if self._draw_btn(surface, y, "Add Specialist Node", (100, 150, 200)):
            self._action_add_specialist()
        y += BTN_H + BTN_GAP

        # Factory stats
        y += 10
        pygame.draw.line(surface, SEPARATOR, (BTN_X, y), (BTN_X + BTN_W, y), 1)
        y += 10

        stats = self.world.get_stats()
        for label, val in [
            ("Workers", str(stats["num_workers"])),
            ("Nodes", str(stats["nodes"])),
            ("Categories", str(len(stats["active_categories"]))),
            ("Tick", str(stats["tick_count"])),
        ]:
            t = self.font_sm.render(f"{label}: {val}", True, TEXT_DARK)
            surface.blit(t, (BTN_X, y))
            y += 18

    def _draw_node_info(self, surface, node, y):
        """Draw info for the selected node. Returns updated y."""
        # Node ID
        t = self.font.render(f"Node: {node.node_id}", True, TEXT_DARK)
        surface.blit(t, (BTN_X, y))
        y += 22

        if node.worker:
            w = node.worker
            t = self.font_sm.render(f"Worker: {w.name}", True, TEXT_DARK)
            surface.blit(t, (BTN_X, y))
            y += 18

            # Accuracy bar
            acc = w.cached_accuracy
            t = self.font_sm.render(f"Accuracy: {acc*100:.0f}%", True, TEXT_DARK)
            surface.blit(t, (BTN_X, y))
            y += 16
            bar_w = BTN_W - 4
            pygame.draw.rect(surface, (180, 180, 180), (BTN_X, y, bar_w, 10))
            fill_w = int(acc * bar_w)
            bar_color = GREEN if acc > 0.7 else (220, 180, 40) if acc > 0.4 else RED
            pygame.draw.rect(surface, bar_color, (BTN_X, y, fill_w, 10))
            y += 14

            t = self.font_sm.render(f"Classes: {', '.join(w.class_names[:4])}", True, (80, 80, 80))
            surface.blit(t, (BTN_X, y))
            y += 16

            t = self.font_sm.render(f"Support: {w.get_support_set_size()} examples", True, (80, 80, 80))
            surface.blit(t, (BTN_X, y))
            y += 16

            t = self.font_sm.render(f"Speed: {node.processing_speed} | Queue: {len(node.queue)}", True, (80, 80, 80))
            surface.blit(t, (BTN_X, y))
            y += 20

            # Action buttons
            if self._draw_btn(surface, y, "Train Worker", (80, 160, 80)):
                self._action_train()
            y += BTN_H + BTN_GAP
        else:
            t = self.font_sm.render("No worker assigned", True, (150, 100, 100))
            surface.blit(t, (BTN_X, y))
            y += 20

        if self._draw_btn(surface, y, "Connect To Node...", (130, 140, 170)):
            self._action_connect()
        y += BTN_H + BTN_GAP

        if self._draw_btn(surface, y, "Connect To Bin...", (130, 160, 130)):
            self._action_connect_to_bin()
        y += BTN_H + BTN_GAP

        is_root = (node.node_id == self.world.graph.root_id)
        if not is_root:
            if self._draw_btn(surface, y, "Set as Root", (150, 150, 100)):
                self._action_set_root()
            y += BTN_H + BTN_GAP

        if self._draw_btn(surface, y, "Remove Node", (190, 100, 100)):
            self._action_remove_node()
        y += BTN_H + BTN_GAP

        return y

    def _draw_bottom_bar(self, surface):
        by = H - BOTTOM_H
        pygame.draw.rect(surface, BOTTOM_BG, (0, by, W, BOTTOM_H))

        cats = self.world.active_categories
        cat_str = ", ".join(cats[:6])
        if len(cats) > 6:
            cat_str += f" +{len(cats)-6}"
        t = self.font_sm.render(f"Active: {cat_str}", True, TEXT_LIGHT)
        surface.blit(t, (15, by + 8))

        t = self.font_sm.render(f"Objects/tick: {self.world.objects_per_tick}", True, TEXT_LIGHT)
        surface.blit(t, (15, by + 28))

        # Play/pause
        label = "Paused" if self.paused else "Running"
        color = RED if self.paused else GREEN
        t = self.font.render(label, True, color)
        surface.blit(t, (W - 160, by + 15))

        pp_label = "Resume" if self.paused else "Pause"
        pp_rect = pygame.Rect(W - 80, by + 10, 65, 30)
        if self._draw_btn_raw(surface, pp_rect, pp_label, (100, 120, 150)):
            self.paused = not self.paused

        # Back button
        back_rect = pygame.Rect(W - 250, by + 10, 65, 30)
        if self._draw_btn_raw(surface, back_rect, "Quit", (150, 120, 100)):
            return "MENU"

        return None

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _draw_text_dialog(self, surface):
        self._draw_overlay_bg(surface)
        bx, by = W // 2 - 180, H // 2 - 60
        bw, bh = 360, 120
        pygame.draw.rect(surface, (250, 248, 242), (bx, by, bw, bh), border_radius=8)
        pygame.draw.rect(surface, (0, 0, 0), (bx, by, bw, bh), 2, border_radius=8)

        t = self.font.render(self._dialog_title, True, TEXT_DARK)
        surface.blit(t, (bx + 15, by + 15))

        # Input box
        input_rect = pygame.Rect(bx + 15, by + 45, bw - 30, 28)
        pygame.draw.rect(surface, (255, 255, 255), input_rect)
        pygame.draw.rect(surface, (0, 0, 0), input_rect, 2)
        it = self.font.render(self._dialog_input + "|", True, TEXT_DARK)
        surface.blit(it, (input_rect.x + 5, input_rect.y + 4))

        ht = self.font_sm.render("Enter to confirm, Esc to cancel", True, (120, 120, 120))
        surface.blit(ht, (bx + 15, by + 85))

    def _draw_select_dialog(self, surface):
        self._draw_overlay_bg(surface)
        bx, by = W // 2 - 200, H // 2 - 180
        bw, bh = 400, 360
        pygame.draw.rect(surface, (250, 248, 242), (bx, by, bw, bh), border_radius=8)
        pygame.draw.rect(surface, (0, 0, 0), (bx, by, bw, bh), 2, border_radius=8)

        t = self.font.render(self._dialog_title, True, TEXT_DARK)
        surface.blit(t, (bx + 15, by + 15))

        selected = self._pending_data.get("selected_cats", [])
        oy = by + 45
        for i, opt in enumerate(self._dialog_options):
            is_sel = opt in selected
            opt_rect = pygame.Rect(bx + 15, oy, bw - 30, 28)
            fill = (200, 230, 200) if is_sel else (240, 240, 240)
            pygame.draw.rect(surface, fill, opt_rect, border_radius=4)
            pygame.draw.rect(surface, (0, 0, 0), opt_rect, 1, border_radius=4)
            prefix = "[x] " if is_sel else "[ ] "
            ot = self.font_sm.render(prefix + opt, True, TEXT_DARK)
            surface.blit(ot, (opt_rect.x + 8, opt_rect.y + 5))

            if self._click_pos and opt_rect.collidepoint(self._click_pos):
                if opt in selected:
                    selected.remove(opt)
                else:
                    selected.append(opt)

            oy += 32
            if oy > by + bh - 60:
                break

        # Done button
        done_rect = pygame.Rect(bx + bw // 2 - 50, by + bh - 45, 100, 32)
        if self._draw_btn_raw(surface, done_rect, "Done", (80, 150, 80)):
            if self._pending_action == "add_specialist":
                self._finish_add_specialist()
            elif self._pending_action == "connect_bin_step1":
                self._finish_connect_bin_step1()
            elif self._pending_action == "connect_bin_step2":
                self._finish_connect_bin_step2()
            elif self._pending_action == "connect_node":
                self._finish_connect_node_step1()
            else:
                self.state = IDLE

    def _draw_training(self, surface):
        self._draw_overlay_bg(surface, opaque=True)
        worker = self._training_worker
        if not worker:
            self.state = IDLE
            return

        title_color = (220, 220, 230)
        subtle = (160, 160, 170)

        # Title
        t = self.font_lg.render(f"Training: {worker.name}", True, title_color)
        surface.blit(t, (W // 2 - t.get_width() // 2, 20))

        # Canvas
        canvas_rect = pygame.Rect(270, 70, CANVAS_SIZE, CANVAS_SIZE)
        surface.blit(self._canvas.canvas, canvas_rect.topleft)
        pygame.draw.rect(surface, (120, 120, 120), canvas_rect, 2)
        self._canvas.draw_cursor_preview(surface, canvas_rect)

        # Preview
        preview = self._canvas.get_surface_84()
        preview_scaled = pygame.transform.scale(preview, (70, 70))
        surface.blit(preview_scaled, (25, 80))
        pygame.draw.rect(surface, (120, 120, 120), (25, 80, 70, 70), 1)
        t = self.font_sm.render("Preview", True, subtle)
        surface.blit(t, (33, 155))

        # Toolbar (fits in x=20..260, canvas starts at 270)
        self._canvas.draw_toolbar(surface, self, 20, 178)

        # --- Right panel: labels + add class ---
        label_x = canvas_rect.right + 25
        label_y = 80

        if worker.class_names:
            t = self.font.render("This drawing is:", True, title_color)
            surface.blit(t, (label_x, label_y))
            label_y += 26

            for i, cls_name in enumerate(worker.class_names):
                is_active = (i == self._training_label_idx)
                color = (80, 160, 80) if is_active else (100, 100, 110)
                btn_rect = pygame.Rect(label_x, label_y, 180, 28)
                if self._draw_btn_raw(surface, btn_rect, cls_name, color):
                    self._training_label_idx = i
                    self._training_typing_new_class = False
                label_y += 34
        else:
            t = self.font.render("No classes yet.", True, title_color)
            surface.blit(t, (label_x, label_y))
            label_y += 22
            t = self.font_sm.render("Add a class to start", True, subtle)
            surface.blit(t, (label_x, label_y))
            label_y += 22
            t = self.font_sm.render("training this worker.", True, subtle)
            surface.blit(t, (label_x, label_y))
            label_y += 30

        # Add class button / text input
        label_y += 8
        if self._training_typing_new_class:
            t = self.font_sm.render("New class name:", True, title_color)
            surface.blit(t, (label_x, label_y))
            label_y += 18
            input_rect = pygame.Rect(label_x, label_y, 180, 26)
            pygame.draw.rect(surface, (255, 255, 255), input_rect)
            pygame.draw.rect(surface, (0, 0, 0), input_rect, 2)
            it = self.font_sm.render(self._training_new_class_input + "|", True, TEXT_DARK)
            surface.blit(it, (input_rect.x + 4, input_rect.y + 4))
            label_y += 30
            t = self.font_sm.render("Enter to add, Esc to cancel", True, subtle)
            surface.blit(t, (label_x, label_y))
            label_y += 20
        else:
            if self._draw_btn_raw(surface, pygame.Rect(label_x, label_y, 180, 28),
                                  "+ Add Class", (100, 130, 170)):
                self._training_typing_new_class = True
                self._training_new_class_input = ""
            label_y += 36

        # Support set counts
        if worker.class_names:
            label_y += 5
            t = self.font_sm.render("Examples:", True, subtle)
            surface.blit(t, (label_x, label_y))
            label_y += 16
            for cls_name in worker.class_names:
                count = len(worker._support_set.get(cls_name, []))
                t = self.font_sm.render(f"  {cls_name}: {count}", True, title_color)
                surface.blit(t, (label_x, label_y))
                label_y += 15

        # Submit / Done buttons
        btn_y = canvas_rect.bottom + 15
        has_classes = bool(worker.class_names)

        if has_classes:
            if self._draw_btn_raw(surface, pygame.Rect(310, btn_y, 150, 36),
                                  "Add Example", (80, 160, 80)):
                self._train_submit_drawing()

        if self._draw_btn_raw(surface, pygame.Rect(490, btn_y, 120, 36),
                              "Done", (160, 100, 100)):
            self._train_done()

        # Status message
        if self._status_timer > 0:
            st = self.font.render(self._status_msg, True, (255, 200, 80))
            surface.blit(st, (W // 2 - st.get_width() // 2, btn_y + 45))

    def _draw_overlay_bg(self, surface, opaque=False):
        if opaque:
            surface.fill((40, 42, 48))
        else:
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 210))
            surface.blit(overlay, (0, 0))

    # ------------------------------------------------------------------
    # Button helpers
    # ------------------------------------------------------------------

    def _draw_btn(self, surface, y, text, color):
        """Draw a standard sidebar button. Returns True if clicked."""
        rect = pygame.Rect(BTN_X, y, BTN_W, BTN_H)
        return self._draw_btn_raw(surface, rect, text, color)

    def _draw_btn_raw(self, surface, rect, text, color):
        """Draw a button at arbitrary rect. Returns True if clicked."""
        mouse = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse)
        c = tuple(min(255, v + 25) for v in color) if hover else color
        pygame.draw.rect(surface, c, rect, border_radius=5)
        pygame.draw.rect(surface, (0, 0, 0), rect, 1, border_radius=5)
        t = self.font_sm.render(text, True, TEXT_DARK)
        tx = rect.x + (rect.width - t.get_width()) // 2
        ty = rect.y + (rect.height - t.get_height()) // 2
        surface.blit(t, (tx, ty))
        return self._click_pos is not None and rect.collidepoint(self._click_pos)

    # DrawingCanvas.draw_toolbar expects a renderer-like interface
    def draw_text(self, x, y, text, color=TEXT_DARK, font=None):
        f = font or self.font_sm
        t = f.render(text, True, color)
        pygame.display.get_surface().blit(t, (x, y))

    def draw_button(self, rect, text, color=(100, 150, 220)):
        return self._draw_btn_raw(pygame.display.get_surface(), rect, text, color)
