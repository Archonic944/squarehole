# TV-Remote-Style Button Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the uniform column of identical rectangular buttons with visually distinct controls — a radial context menu around selected nodes, side-by-side build buttons, a pill-shaped speed upgrade, and a circular play/pause button.

**Architecture:** All changes are in `app/ui/factory_floor.py`. We add icon-drawing helpers, a radial menu system, and reshape the side panel and bottom bar. Node-specific action buttons move from the side panel into the radial menu that appears around the selected node in the graph area.

**Tech Stack:** Python 3.11, pygame

---

### Task 1: Add Icon Drawing Helpers

**Files:**
- Modify: `app/ui/factory_floor.py` (add functions after the `_tensor_to_thumb` function, around line 106)

- [ ] **Step 1: Add all icon drawing functions**

Add these functions after `_tensor_to_thumb` (line 106), before the `FlowShape` class:

```python
# ---------------------------------------------------------------------------
# Icon drawing helpers (SVG-style procedural icons)
# ---------------------------------------------------------------------------

def _draw_icon_mallet(surface, cx, cy, r, color=(255, 255, 255)):
    """Mallet icon for Train Worker. Drawn relative to center (cx, cy) within radius r."""
    import math
    # Handle: diagonal line from bottom-left to center
    handle_len = r * 0.7
    hx1 = cx - handle_len * 0.5
    hy1 = cy + handle_len * 0.5
    hx2 = cx + handle_len * 0.15
    hy2 = cy - handle_len * 0.15
    pygame.draw.line(surface, color, (int(hx1), int(hy1)), (int(hx2), int(hy2)), 3)
    # Head: rectangle rotated 45°, at the top end of the handle
    head_w = r * 0.65
    head_h = r * 0.3
    angle = math.radians(-45)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Head center is slightly past hx2/hy2
    hcx = hx2 + cos_a * head_w * 0.1
    hcy = hy2 + sin_a * head_w * 0.1
    # Four corners of the head rectangle
    points = []
    for dx, dy in [(-head_w/2, -head_h/2), (head_w/2, -head_h/2),
                   (head_w/2, head_h/2), (-head_w/2, head_h/2)]:
        rx = hcx + dx * cos_a - dy * sin_a
        ry = hcy + dx * sin_a + dy * cos_a
        points.append((int(rx), int(ry)))
    pygame.draw.polygon(surface, color, points)


def _draw_icon_arrow_node(surface, cx, cy, r, color=(255, 255, 255)):
    """Arrow pointing to a square — Connect To Node."""
    # Horizontal arrow shaft
    x1 = cx - r * 0.5
    x2 = cx + r * 0.2
    y_mid = cy
    pygame.draw.line(surface, color, (int(x1), int(y_mid)), (int(x2), int(y_mid)), 2)
    # Arrowhead
    ah = r * 0.25
    pygame.draw.polygon(surface, color, [
        (int(x2 + ah), int(y_mid)),
        (int(x2), int(y_mid - ah * 0.6)),
        (int(x2), int(y_mid + ah * 0.6)),
    ])
    # Small square at destination
    sq = r * 0.25
    sq_x = cx + r * 0.55
    pygame.draw.rect(surface, color,
                     (int(sq_x - sq/2), int(cy - sq/2), int(sq), int(sq)), 2)


def _draw_icon_arrow_bin(surface, cx, cy, r, color=(255, 255, 255)):
    """Downward arrow into a bucket — Connect To Bin."""
    # Vertical arrow shaft
    y1 = cy - r * 0.45
    y2 = cy + r * 0.1
    pygame.draw.line(surface, color, (int(cx), int(y1)), (int(cx), int(y2)), 2)
    # Arrowhead pointing down
    ah = r * 0.25
    pygame.draw.polygon(surface, color, [
        (int(cx), int(y2 + ah)),
        (int(cx - ah * 0.6), int(y2)),
        (int(cx + ah * 0.6), int(y2)),
    ])
    # Bucket/trapezoid below
    bw_top = r * 0.6
    bw_bot = r * 0.4
    bh = r * 0.3
    by = cy + r * 0.35
    pygame.draw.lines(surface, color, False, [
        (int(cx - bw_top/2), int(by)),
        (int(cx - bw_bot/2), int(by + bh)),
        (int(cx + bw_bot/2), int(by + bh)),
        (int(cx + bw_top/2), int(by)),
    ], 2)


def _draw_icon_x(surface, cx, cy, r, color=(255, 255, 255)):
    """X mark — Remove Node."""
    d = r * 0.4
    pygame.draw.line(surface, color, (int(cx - d), int(cy - d)), (int(cx + d), int(cy + d)), 3)
    pygame.draw.line(surface, color, (int(cx + d), int(cy - d)), (int(cx - d), int(cy + d)), 3)


def _draw_icon_star(surface, cx, cy, r, color=(255, 255, 255)):
    """5-point star — Set as Root."""
    import math
    outer_r = r * 0.55
    inner_r = r * 0.22
    points = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)  # start from top
        rad = outer_r if i % 2 == 0 else inner_r
        points.append((int(cx + rad * math.cos(angle)),
                       int(cy + rad * math.sin(angle))))
    pygame.draw.polygon(surface, color, points, 2)


def _draw_icon_play(surface, cx, cy, r, color=(255, 255, 255)):
    """Right-pointing play triangle."""
    d = r * 0.45
    pygame.draw.polygon(surface, color, [
        (int(cx - d * 0.5), int(cy - d)),
        (int(cx + d), int(cy)),
        (int(cx - d * 0.5), int(cy + d)),
    ])


def _draw_icon_pause(surface, cx, cy, r, color=(255, 255, 255)):
    """Two vertical pause bars."""
    bar_w = max(2, int(r * 0.2))
    bar_h = int(r * 0.7)
    gap = int(r * 0.2)
    pygame.draw.rect(surface, color,
                     (int(cx - gap - bar_w), int(cy - bar_h), bar_w, bar_h * 2))
    pygame.draw.rect(surface, color,
                     (int(cx + gap), int(cy - bar_h), bar_w, bar_h * 2))


def _draw_icon_router(surface, cx, cy, r, color=(255, 255, 255)):
    """Branching lines — one input splitting into two outputs."""
    # Single line from left
    pygame.draw.line(surface, color,
                     (int(cx - r * 0.5), int(cy)),
                     (int(cx), int(cy)), 2)
    # Branch up
    pygame.draw.line(surface, color,
                     (int(cx), int(cy)),
                     (int(cx + r * 0.5), int(cy - r * 0.35)), 2)
    # Branch down
    pygame.draw.line(surface, color,
                     (int(cx), int(cy)),
                     (int(cx + r * 0.5), int(cy + r * 0.35)), 2)


def _draw_icon_specialist(surface, cx, cy, r, color=(255, 255, 255)):
    """Single focused line with dot — specialist focus."""
    pygame.draw.line(surface, color,
                     (int(cx - r * 0.5), int(cy)),
                     (int(cx + r * 0.3), int(cy)), 2)
    pygame.draw.circle(surface, color, (int(cx + r * 0.45), int(cy)), max(2, int(r * 0.12)))
```

- [ ] **Step 2: Verify the game still launches**

Run: `.venv/bin/python -m app.main`
Expected: Game launches normally, no import errors. Close immediately.

- [ ] **Step 3: Commit**

```bash
git add app/ui/factory_floor.py
git commit -m "feat: add procedural icon drawing helpers for TV-remote button redesign"
```

---

### Task 2: Implement the Radial Context Menu

**Files:**
- Modify: `app/ui/factory_floor.py`
  - Add `_draw_radial_menu` method to `FactoryFloorUI`
  - Modify `_handle_idle_click` to handle radial button clicks
  - Modify `_draw_graph` to call radial menu drawing
  - Remove node action buttons from `_draw_node_info`

- [ ] **Step 1: Add radial menu constants and define the button specs**

Add these constants after the existing `FLOW_SPEED = 150` line (line 66):

```python
# Radial menu
RADIAL_RING_R = 65       # distance from node center to button center
RADIAL_BUTTONS = [
    # (key, label, radius, base_color, icon_fn_name, default_angle_deg)
    ("train",       "Train Worker",     22, (80, 160, 80),   "_draw_icon_mallet",     -90),   # top
    ("connect_node","Connect To Node",  16, (130, 140, 170), "_draw_icon_arrow_node",  -30),   # upper-right
    ("connect_bin", "Connect To Bin",   16, (130, 160, 130), "_draw_icon_arrow_bin",    30),   # lower-right
    ("remove",      "Remove Node",      13, (190, 100, 100), "_draw_icon_x",          150),   # lower-left
    ("set_root",    "Set as Root",      14, (150, 150, 100), "_draw_icon_star",        210),   # left
]
```

- [ ] **Step 2: Add the `_compute_radial_positions` method**

Add this method to `FactoryFloorUI`, after `_draw_node_info` (after line 1118):

```python
def _compute_radial_positions(self, cx, cy, buttons):
    """Compute (bx, by) for each radial button, adapting the arc if near edges.
    
    Args:
        cx, cy: center of the selected node
        buttons: list of (key, label, radius, color, icon_fn, default_angle_deg) tuples
    
    Returns:
        list of (bx, by, key, label, radius, color, icon_fn) for each visible button
    """
    import math
    margin = max(b[2] for b in buttons) + 4  # largest button radius + border

    # Available space in each direction from node center
    space_left = cx - GRAPH_X
    space_right = (GRAPH_X + GRAPH_W) - cx
    space_up = cy - GRAPH_Y
    space_down = (GRAPH_Y + GRAPH_H) - cy

    needed = RADIAL_RING_R + margin

    # Determine blocked angular ranges (in degrees, 0=right, 90=down, -90=up)
    blocked = []
    if space_right < needed:
        blocked.append((-70, 70))     # block right side
    if space_left < needed:
        blocked.append((110, 250))    # block left side
    if space_up < needed:
        blocked.append((-160, -20))   # block top
    if space_down < needed:
        blocked.append((20, 160))     # block bottom

    def angle_blocked(deg):
        """Check if a given angle (degrees) falls in any blocked range."""
        for lo, hi in blocked:
            # Normalize to check
            a = deg % 360
            lo_n = lo % 360
            hi_n = hi % 360
            if lo_n <= hi_n:
                if lo_n <= a <= hi_n:
                    return True
            else:
                if a >= lo_n or a <= hi_n:
                    return True
        return False

    if not blocked:
        # Full circle — use default angles
        result = []
        for key, label, radius, color, icon_fn, default_deg in buttons:
            angle_rad = math.radians(default_deg)
            bx = cx + RADIAL_RING_R * math.cos(angle_rad)
            by = cy + RADIAL_RING_R * math.sin(angle_rad)
            result.append((int(bx), int(by), key, label, radius, color, icon_fn))
        return result

    # Find the widest unblocked arc and distribute buttons evenly
    # Sample angles to find valid range
    valid_angles = []
    for deg in range(0, 360, 5):
        if not angle_blocked(deg):
            valid_angles.append(deg)

    if not valid_angles:
        # Fallback: just use defaults (shouldn't happen with normal layouts)
        result = []
        for key, label, radius, color, icon_fn, default_deg in buttons:
            angle_rad = math.radians(default_deg)
            bx = cx + RADIAL_RING_R * math.cos(angle_rad)
            by = cy + RADIAL_RING_R * math.sin(angle_rad)
            result.append((int(bx), int(by), key, label, radius, color, icon_fn))
        return result

    # Distribute buttons evenly across valid arc
    n = len(buttons)
    # Use the valid angles' midpoint range
    arc_start = valid_angles[0]
    arc_end = valid_angles[-1]
    # Handle wrap-around
    if arc_end - arc_start > 300:
        # The valid range wraps around 0. Find the gap.
        gaps = []
        for i in range(len(valid_angles) - 1):
            if valid_angles[i + 1] - valid_angles[i] > 10:
                gaps.append(i)
        if gaps:
            gap_idx = gaps[0]
            arc_start = valid_angles[gap_idx + 1]
            arc_end = valid_angles[gap_idx] + 360
    
    arc_span = arc_end - arc_start
    if n > 1:
        step = arc_span / (n - 1) if arc_span > 0 else 0
    else:
        step = 0

    result = []
    for i, (key, label, radius, color, icon_fn, _) in enumerate(buttons):
        deg = arc_start + i * step
        angle_rad = math.radians(deg)
        bx = cx + RADIAL_RING_R * math.cos(angle_rad)
        by = cy + RADIAL_RING_R * math.sin(angle_rad)
        result.append((int(bx), int(by), key, label, radius, color, icon_fn))
    return result
```

- [ ] **Step 3: Add the `_draw_radial_menu` method**

Add this method right after `_compute_radial_positions`:

```python
def _draw_radial_menu(self, surface):
    """Draw the radial context menu around the selected node."""
    if not self.selected_node or self.selected_node not in self._node_rects:
        return
    
    node = self.world.graph.nodes.get(self.selected_node)
    if not node:
        return
    
    rect = self._node_rects[self.selected_node]
    cx, cy = rect.centerx, rect.centery

    # Filter buttons based on node state
    is_root = (node.node_id == self.world.graph.root_id)
    has_worker = node.worker is not None
    
    buttons = []
    for spec in RADIAL_BUTTONS:
        key = spec[0]
        if key == "set_root" and is_root:
            continue
        if key == "train" and not has_worker:
            continue
        buttons.append(spec)

    positions = self._compute_radial_positions(cx, cy, buttons)
    mouse = pygame.mouse.get_pos()
    self._radial_hit = None  # track which button is hovered for tooltip

    for bx, by, key, label, radius, color, icon_fn_name in positions:
        # Check hover
        dist_sq = (mouse[0] - bx) ** 2 + (mouse[1] - by) ** 2
        hover = dist_sq <= radius * radius
        
        # Draw drop shadow
        shadow_surf = pygame.Surface((radius * 2 + 6, radius * 2 + 6), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surf, (0, 0, 0, 60),
                          (radius + 3, radius + 3), radius)
        surface.blit(shadow_surf, (bx - radius - 3 + 2, by - radius - 3 + 2))

        # Draw button circle
        fill = tuple(min(255, v + 25) for v in color) if hover else color
        pygame.draw.circle(surface, fill, (bx, by), radius)
        pygame.draw.circle(surface, (40, 40, 40), (bx, by), radius, 2)

        # Draw icon
        icon_fn = globals()[icon_fn_name]
        icon_fn(surface, bx, by, radius)

        # Check click
        if self._click_pos and (self._click_pos[0] - bx) ** 2 + (self._click_pos[1] - by) ** 2 <= radius * radius:
            self._radial_hit = key

        # Draw tooltip on hover
        if hover:
            tip_text = self.font_sm.render(label, True, (230, 230, 230))
            tw, th = tip_text.get_width(), tip_text.get_height()
            pad = 4
            # Position tooltip outward from node center
            import math
            angle = math.atan2(by - cy, bx - cx)
            tip_cx = bx + int(math.cos(angle) * (radius + tw // 2 + 12))
            tip_cy = by + int(math.sin(angle) * (radius + th // 2 + 8))
            # Clamp to screen
            tip_x = max(GRAPH_X + 2, min(GRAPH_X + GRAPH_W - tw - pad * 2 - 2, tip_cx - tw // 2 - pad))
            tip_y = max(GRAPH_Y + 2, min(GRAPH_Y + GRAPH_H - th - pad * 2 - 2, tip_cy - th // 2 - pad))
            tip_rect = pygame.Rect(tip_x, tip_y, tw + pad * 2, th + pad * 2)
            pygame.draw.rect(surface, (50, 50, 50), tip_rect, border_radius=4)
            surface.blit(tip_text, (tip_x + pad, tip_y + pad))
```

- [ ] **Step 4: Handle radial button clicks in `_handle_idle_click`**

Replace the existing `_handle_idle_click` method (lines 547-558) with:

```python
def _handle_idle_click(self):
    if not self._click_pos:
        return
    
    # If radial menu is showing, check radial buttons first
    if self.selected_node and self._radial_hit:
        action = self._radial_hit
        self._radial_hit = None
        if action == "train":
            self._action_train()
        elif action == "connect_node":
            self._action_connect()
        elif action == "connect_bin":
            self._action_connect_to_bin()
        elif action == "set_root":
            self._action_set_root()
        elif action == "remove":
            self._action_remove_node()
        return
    
    # Check if clicked a radial button area (consume click if inside any button)
    if self.selected_node and self.selected_node in self._node_rects:
        node = self.world.graph.nodes.get(self.selected_node)
        if node:
            rect = self._node_rects[self.selected_node]
            cx, cy = rect.centerx, rect.centery
            is_root = (node.node_id == self.world.graph.root_id)
            has_worker = node.worker is not None
            buttons = []
            for spec in RADIAL_BUTTONS:
                key = spec[0]
                if key == "set_root" and is_root:
                    continue
                if key == "train" and not has_worker:
                    continue
                buttons.append(spec)
            positions = self._compute_radial_positions(cx, cy, buttons)
            for bx, by, key, label, radius, color, icon_fn_name in positions:
                if (self._click_pos[0] - bx) ** 2 + (self._click_pos[1] - by) ** 2 <= radius * radius:
                    return  # click was inside a radial button area, consumed
    
    # Check if clicked a node
    for nid, rect in self._node_rects.items():
        if rect.collidepoint(self._click_pos):
            self.selected_node = nid
            self._radial_hit = None
            return
    # Clicked empty graph area — deselect
    graph_area = pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)
    if graph_area.collidepoint(self._click_pos):
        self.selected_node = None
```

- [ ] **Step 5: Add `_radial_hit` init and draw call**

In `__init__` (around line 210, after `self._click_pos = None`), add:

```python
        # Radial menu state
        self._radial_hit = None
```

In `_draw_graph` method, add the radial menu draw call at the very end of the method (after all nodes and edges are drawn, after the flow shapes). Find the end of `_draw_graph` — it's just before `_draw_side_panel`. Add:

```python
        # Radial context menu (drawn last for highest z-order)
        if self.state == IDLE:
            self._draw_radial_menu(surface)
```

- [ ] **Step 6: Remove node action buttons from `_draw_node_info`**

Replace the `_draw_node_info` method (lines 1054-1118) with this version that only shows info, no buttons:

```python
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
    else:
        t = self.font_sm.render("No worker assigned", True, (150, 100, 100))
        surface.blit(t, (BTN_X, y))
        y += 20

    return y
```

- [ ] **Step 7: Verify the game launches and radial menu appears**

Run: `.venv/bin/python -m app.main`
Expected: Game launches. Add a node, click it — radial buttons fan out around it. Hover shows tooltips. Clicking a radial button triggers the action. Clicking elsewhere dismisses.

- [ ] **Step 8: Commit**

```bash
git add app/ui/factory_floor.py
git commit -m "feat: add radial context menu for node actions"
```

---

### Task 3: Reshape the Side Panel — Build Buttons and Speed Upgrade

**Files:**
- Modify: `app/ui/factory_floor.py` — update `_draw_side_panel` method (lines 996-1052)

- [ ] **Step 1: Replace `_draw_side_panel` with the new layout**

Replace the `_draw_side_panel` method with:

```python
def _draw_side_panel(self, surface):
    panel_rect = pygame.Rect(SIDE_X, SIDE_Y, SIDE_W, GRAPH_H)
    pygame.draw.rect(surface, PANEL_BG, panel_rect)
    pygame.draw.line(surface, SEPARATOR, (SIDE_X, SIDE_Y), (SIDE_X, SIDE_Y + GRAPH_H), 2)

    y = SIDE_Y + 10

    # --- Build buttons (side by side) ---
    build_btn_w = 108
    build_btn_h = 38
    build_btn_gap = 9
    bx1 = BTN_X
    bx2 = BTN_X + build_btn_w + build_btn_gap

    # Add Router button (left)
    router_rect = pygame.Rect(bx1, y, build_btn_w, build_btn_h)
    if self._draw_btn_with_icon(surface, router_rect, "Router", (100, 150, 200),
                                _draw_icon_router):
        self._action_add_router()

    # Add Specialist button (right)
    spec_rect = pygame.Rect(bx2, y, build_btn_w, build_btn_h)
    if self._draw_btn_with_icon(surface, spec_rect, "Specialist", (80, 160, 160),
                                _draw_icon_specialist):
        self._action_add_specialist()

    y += build_btn_h + 12

    # --- Speed upgrade (pill shape) ---
    cost = self.world.get_speed_upgrade_cost()
    spd = self.world.speed_level
    label = f"Spd {spd}→{spd+1}  (${cost:.0f})"
    can_afford = self.world.economy.can_afford(cost)
    pill_color = (100, 170, 100) if can_afford else (140, 140, 140)
    pill_rect = pygame.Rect(BTN_X, y, BTN_W, 34)
    pill_radius = 17  # half of height for fully rounded ends

    mouse = pygame.mouse.get_pos()
    hover = pill_rect.collidepoint(mouse)
    fill = tuple(min(255, v + 25) for v in pill_color) if hover else pill_color
    pygame.draw.rect(surface, fill, pill_rect, border_radius=pill_radius)
    border_color = tuple(max(0, v - 40) for v in pill_color)
    pygame.draw.rect(surface, border_color, pill_rect, 2, border_radius=pill_radius)
    t = self.font_sm.render(label, True, TEXT_DARK)
    tx = pill_rect.x + (pill_rect.width - t.get_width()) // 2
    ty = pill_rect.y + (pill_rect.height - t.get_height()) // 2
    surface.blit(t, (tx, ty))
    if self._click_pos and pill_rect.collidepoint(self._click_pos):
        if self.world.buy_speed_upgrade():
            self._show_status(f"Speed upgraded to Lv.{self.world.speed_level}!")

    y += 34 + 10

    # --- Separator ---
    pygame.draw.line(surface, SEPARATOR, (BTN_X, y), (BTN_X + BTN_W, y), 1)
    y += 10

    # --- Node info (when selected) ---
    if self.selected_node and self.selected_node in self.world.graph.nodes:
        node = self.world.graph.nodes[self.selected_node]
        y = self._draw_node_info(surface, node, y)
    else:
        t = self.font.render("No node selected", True, (120, 120, 120))
        surface.blit(t, (BTN_X, y))
        y += 30

    # --- Separator ---
    y += 5
    pygame.draw.line(surface, SEPARATOR, (BTN_X, y), (BTN_X + BTN_W, y), 1)
    y += 10

    # --- Factory stats ---
    stats = self.world.get_stats()
    for label, val in [
        ("Workers", str(stats["num_workers"])),
        ("Nodes", str(stats["nodes"])),
        ("Speed", f"Lv.{stats.get('speed_level', 1)}"),
        ("Categories", str(len(stats["active_categories"]))),
        ("Tick", str(stats["tick_count"])),
    ]:
        t = self.font_sm.render(f"{label}: {val}", True, TEXT_DARK)
        surface.blit(t, (BTN_X, y))
        y += 18
```

- [ ] **Step 2: Add the `_draw_btn_with_icon` helper**

Add this method after `_draw_btn_raw` (after line 1418):

```python
def _draw_btn_with_icon(self, surface, rect, text, color, icon_fn):
    """Draw a button with a small icon to the left of the text. Returns True if clicked."""
    mouse = pygame.mouse.get_pos()
    hover = rect.collidepoint(mouse)
    c = tuple(min(255, v + 25) for v in color) if hover else color
    pygame.draw.rect(surface, c, rect, border_radius=6)
    pygame.draw.rect(surface, (0, 0, 0), rect, 1, border_radius=6)
    # Icon on the left
    icon_size = min(rect.height - 8, 18)
    icon_cx = rect.x + 6 + icon_size // 2
    icon_cy = rect.y + rect.height // 2
    icon_fn(surface, icon_cx, icon_cy, icon_size, color=TEXT_DARK)
    # Text to the right of icon
    t = self.font_sm.render(text, True, TEXT_DARK)
    tx = rect.x + 6 + icon_size + 6
    ty = rect.y + (rect.height - t.get_height()) // 2
    surface.blit(t, (tx, ty))
    return self._click_pos is not None and rect.collidepoint(self._click_pos)
```

- [ ] **Step 3: Verify game launches and side panel looks correct**

Run: `.venv/bin/python -m app.main`
Expected: Side panel shows two side-by-side build buttons at the top with icons, a pill-shaped speed upgrade below, then node info (no action buttons), then stats.

- [ ] **Step 4: Commit**

```bash
git add app/ui/factory_floor.py
git commit -m "feat: reshape side panel with side-by-side build buttons and pill speed upgrade"
```

---

### Task 4: Reshape the Bottom Bar — Circular Play/Pause

**Files:**
- Modify: `app/ui/factory_floor.py` — update `_draw_bottom_bar` method (lines 1120-1150)

- [ ] **Step 1: Replace `_draw_bottom_bar` with the new layout**

Replace the `_draw_bottom_bar` method with:

```python
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

    # --- Quit button (small rounded square, leftmost of right controls) ---
    quit_rect = pygame.Rect(W - 175, by + 10, 55, 30)
    mouse = pygame.mouse.get_pos()
    quit_hover = quit_rect.collidepoint(mouse)
    quit_color = (170, 140, 120) if quit_hover else (150, 120, 100)
    pygame.draw.rect(surface, quit_color, quit_rect, border_radius=6)
    pygame.draw.rect(surface, (80, 60, 50), quit_rect, 1, border_radius=6)
    qt = self.font_sm.render("Quit", True, TEXT_LIGHT)
    surface.blit(qt, (quit_rect.x + (quit_rect.width - qt.get_width()) // 2,
                      quit_rect.y + (quit_rect.height - qt.get_height()) // 2))
    if self._click_pos and quit_rect.collidepoint(self._click_pos):
        return "MENU"

    # --- Status label ---
    label = "Paused" if self.paused else "Running"
    label_color = RED if self.paused else GREEN
    lt = self.font.render(label, True, label_color)
    surface.blit(lt, (W - 110, by + 15))

    # --- Circular play/pause button ---
    circle_r = 18
    circle_cx = W - 37
    circle_cy = by + 25  # vertically centered in 50px bar

    circle_hover = (mouse[0] - circle_cx) ** 2 + (mouse[1] - circle_cy) ** 2 <= circle_r * circle_r
    if self.paused:
        fill = (230, 80, 80) if circle_hover else (200, 60, 60)
    else:
        fill = (80, 195, 80) if circle_hover else (60, 170, 60)
    
    # Drop shadow
    shadow_surf = pygame.Surface((circle_r * 2 + 6, circle_r * 2 + 6), pygame.SRCALPHA)
    pygame.draw.circle(shadow_surf, (0, 0, 0, 60),
                      (circle_r + 3, circle_r + 3), circle_r)
    surface.blit(shadow_surf, (circle_cx - circle_r - 3 + 2, circle_cy - circle_r - 3 + 2))

    pygame.draw.circle(surface, fill, (circle_cx, circle_cy), circle_r)
    pygame.draw.circle(surface, (40, 40, 40), (circle_cx, circle_cy), circle_r, 2)

    # Icon
    if self.paused:
        _draw_icon_play(surface, circle_cx, circle_cy, circle_r)
    else:
        _draw_icon_pause(surface, circle_cx, circle_cy, circle_r)

    # Click
    if self._click_pos and (self._click_pos[0] - circle_cx) ** 2 + (self._click_pos[1] - circle_cy) ** 2 <= circle_r * circle_r:
        self.paused = not self.paused

    # --- Status message ---
    if self._status_timer > 0:
        st = self.font.render(self._status_msg, True, (255, 200, 80))
        surface.blit(st, (W // 2 - st.get_width() // 2, by + 15))

    return None
```

- [ ] **Step 2: Remove status message from `draw` method if it's rendered elsewhere**

Check if the status message is also drawn in `draw()` — the current code only draws it in `_draw_bottom_bar` via the old button system. The new `_draw_bottom_bar` includes it, so nothing else to change.

- [ ] **Step 3: Verify game launches and bottom bar looks correct**

Run: `.venv/bin/python -m app.main`
Expected: Bottom bar shows the Quit button as a small rounded square on the left of the right controls, "Paused"/"Running" label, and a circular play/pause button with the appropriate icon. Clicking the circle toggles pause.

- [ ] **Step 4: Commit**

```bash
git add app/ui/factory_floor.py
git commit -m "feat: reshape bottom bar with circular play/pause and distinct quit button"
```

---

### Task 5: Visual Testing and Edge Case Fixes

**Files:**
- Modify: `app/ui/factory_floor.py` (minor fixes as needed)

- [ ] **Step 1: Test radial menu with node near right edge**

Run: `.venv/bin/python -m app.main`
- Add multiple nodes so the tree layout pushes some nodes toward the right edge of the graph area (close to the side panel boundary at x=769).
- Select a right-edge node.
Expected: Radial buttons adapt — they fan out to the left instead of overlapping the side panel.

- [ ] **Step 2: Test radial menu with node near top edge**

- Select a node near the top of the graph area (close to y=45).
Expected: Radial buttons adapt — they fan out downward.

- [ ] **Step 3: Test clicking through all radial actions**

For a selected node:
1. Click "Train Worker" radial button → training overlay opens.
2. Close training. Click "Connect To Node" → dialog opens asking which output.
3. Click "Connect To Bin" → dialog opens.
4. Click "Set as Root" → node becomes root.
5. Click "Remove Node" → node is removed, radial disappears.

- [ ] **Step 4: Test that clicking a different node switches the radial**

Click node A (radial shows). Then click node B directly. Expected: radial moves to node B.

- [ ] **Step 5: Fix any issues found, then commit**

```bash
git add app/ui/factory_floor.py
git commit -m "fix: polish radial menu edge cases and visual tweaks"
```
