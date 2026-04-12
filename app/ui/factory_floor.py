"""Factory floor UI — graph editor, training, and live simulation view."""

import math
import random
import pygame
import torch
from collections import deque

from app.ui.drawing_canvas import DrawingCanvas, CANVAS_SIZE
from app.ui.graph_layout import ForceLayout, BIN_W, BIN_H


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
GRAPH_PAD = 30      # padding inside graph area

# Minimap
MINIMAP_MAX_W = 160
MINIMAP_MAX_H = 120
MINIMAP_PAD = 8

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


class FloatingText:
    """A text label that floats upward and fades out."""

    __slots__ = ("x", "y", "text", "color", "t", "duration", "alive")

    def __init__(self, x, y, text, color, duration=1.2):
        self.x = float(x)
        self.y = float(y)
        self.text = text
        self.color = color
        self.t = 0.0
        self.duration = duration
        self.alive = True

    def update(self, dt):
        self.t += dt
        self.y -= 30 * dt
        if self.t >= self.duration:
            self.alive = False

    def draw(self, surface, font):
        if not self.alive or self.t / self.duration > 0.85:
            return
        t = font.render(self.text, True, self.color)
        surface.blit(t, (int(self.x), int(self.y)))

    def draw_offset(self, surface, font, off_x, off_y):
        if not self.alive or self.t / self.duration > 0.85:
            return
        t = font.render(self.text, True, self.color)
        surface.blit(t, (int(self.x - off_x), int(self.y - off_y)))

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


# ---------------------------------------------------------------------------
# Icon drawing helpers (SVG-style procedural icons)
# ---------------------------------------------------------------------------

def _draw_icon_mallet(surface, cx, cy, r, color=(255, 255, 255)):
    """Train Worker — a mallet: diagonal handle with rectangular head."""
    # Handle: lower-left to upper-right area
    hx1 = cx - int(r * 0.45)
    hy1 = cy + int(r * 0.45)
    hx2 = cx + int(r * 0.15)
    hy2 = cy - int(r * 0.15)
    pygame.draw.line(surface, color, (hx1, hy1), (hx2, hy2), 3)
    # Head: filled rectangle rotated ~45 degrees at the top of the handle
    hw = r * 0.45  # half-width of head
    hh = r * 0.15  # half-height of head
    angle = math.radians(45)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Center of the head, slightly past the handle end
    mx = cx + int(r * 0.25)
    my = cy - int(r * 0.25)
    corners = []
    for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        corners.append((mx + int(rx), my + int(ry)))
    pygame.draw.polygon(surface, color, corners)


def _draw_icon_arrow_node(surface, cx, cy, r, color=(255, 255, 255)):
    """Connect To Node — horizontal arrow with a small square at destination."""
    # Horizontal arrow from left toward right
    x1 = cx - int(r * 0.5)
    x2 = cx + int(r * 0.25)
    pygame.draw.line(surface, color, (x1, cy), (x2, cy), 2)
    # Arrowhead
    ah = int(r * 0.2)
    pygame.draw.polygon(surface, color, [
        (x2 + ah, cy),
        (x2 - 1, cy - ah),
        (x2 - 1, cy + ah),
    ])
    # Small outlined square at the destination end
    sq = int(r * 0.22)
    sx = cx + int(r * 0.5) - sq
    sy = cy - sq
    pygame.draw.rect(surface, color, (sx, sy, sq * 2, sq * 2), 2)


def _draw_icon_arrow_bin(surface, cx, cy, r, color=(255, 255, 255)):
    """Connect To Bin — downward arrow with open-top trapezoid/bucket below."""
    # Vertical downward arrow
    y1 = cy - int(r * 0.5)
    y2 = cy + int(r * 0.05)
    pygame.draw.line(surface, color, (cx, y1), (cx, y2), 2)
    # Arrowhead
    ah = int(r * 0.18)
    pygame.draw.polygon(surface, color, [
        (cx, y2 + ah),
        (cx - ah, y2 - 1),
        (cx + ah, y2 - 1),
    ])
    # Open-top trapezoid (bucket) — left side, bottom, right side
    bw_top = int(r * 0.35)
    bw_bot = int(r * 0.25)
    bt = cy + int(r * 0.3)
    bb = cy + int(r * 0.55)
    pygame.draw.lines(surface, color, False, [
        (cx - bw_top, bt),
        (cx - bw_bot, bb),
        (cx + bw_bot, bb),
        (cx + bw_top, bt),
    ], 2)


def _draw_icon_x(surface, cx, cy, r, color=(255, 255, 255)):
    """Remove Node — two crossing diagonal lines."""
    d = int(r * 0.4)
    pygame.draw.line(surface, color, (cx - d, cy - d), (cx + d, cy + d), 3)
    pygame.draw.line(surface, color, (cx + d, cy - d), (cx - d, cy + d), 3)


def _draw_icon_star(surface, cx, cy, r, color=(255, 255, 255)):
    """Set as Root — 5-point star outline."""
    outer_r = r * 0.55
    inner_r = r * 0.22
    points = []
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        rad = outer_r if i % 2 == 0 else inner_r
        points.append((
            cx + int(rad * math.cos(angle)),
            cy + int(rad * math.sin(angle)),
        ))
    pygame.draw.polygon(surface, color, points, 2)


def _draw_icon_play(surface, cx, cy, r, color=(255, 255, 255)):
    """Play — right-pointing filled equilateral triangle."""
    h = r * 0.5
    pygame.draw.polygon(surface, color, [
        (cx + int(h * 0.9), cy),
        (cx - int(h * 0.5), cy - int(h * 0.8)),
        (cx - int(h * 0.5), cy + int(h * 0.8)),
    ])


def _draw_icon_pause(surface, cx, cy, r, color=(255, 255, 255)):
    """Pause — two vertical filled rectangles."""
    bw = max(2, int(r * 0.15))
    bh = int(r * 0.5)
    gap = int(r * 0.15)
    pygame.draw.rect(surface, color, (cx - gap - bw, cy - bh, bw, bh * 2))
    pygame.draw.rect(surface, color, (cx + gap, cy - bh, bw, bh * 2))


def _draw_icon_router(surface, cx, cy, r, color=(255, 255, 255)):
    """Router — Y-shape rotated 90° (one line splitting into two)."""
    # Stem from left to center
    x1 = cx - int(r * 0.5)
    xm = cx
    pygame.draw.line(surface, color, (x1, cy), (xm, cy), 2)
    # Upper branch
    x2 = cx + int(r * 0.5)
    yu = cy - int(r * 0.35)
    pygame.draw.line(surface, color, (xm, cy), (x2, yu), 2)
    # Lower branch
    yd = cy + int(r * 0.35)
    pygame.draw.line(surface, color, (xm, cy), (x2, yd), 2)


def _draw_icon_specialist(surface, cx, cy, r, color=(255, 255, 255)):
    """Specialist — horizontal line with a filled circle at the end."""
    x1 = cx - int(r * 0.5)
    x2 = cx + int(r * 0.3)
    pygame.draw.line(surface, color, (x1, cy), (x2, cy), 2)
    dot_r = max(3, int(r * 0.18))
    pygame.draw.circle(surface, color, (x2 + dot_r + 1, cy), dot_r)


class FlowShape:
    """An animated shape thumbnail that moves along an edge."""

    __slots__ = ("sx", "sy", "ex", "ey", "thumb", "border", "t", "duration", "alive")

    def __init__(self, start, end, thumb_surface, border_color=BORDER_TRANSIT, speed_mult=1):
        self.sx, self.sy = float(start[0]), float(start[1])
        self.ex, self.ey = float(end[0]), float(end[1])
        self.thumb = thumb_surface
        self.border = border_color
        self.t = 0.0
        dist = math.hypot(self.ex - self.sx, self.ey - self.sy)
        self.duration = max(0.1, dist / (FLOW_SPEED * speed_mult))
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

    def draw_offset(self, surface, off_x, off_y):
        if not self.alive or self.t < 0:
            return
        x, y = self.pos
        x -= off_x
        y -= off_y
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
        self.base_tick_interval = 1.5  # seconds between ticks at speed level 1

        # Fonts (initialized on first draw since pygame must be init'd)
        self._fonts_ready = False
        self.font = None
        self.font_sm = None
        self.font_lg = None

        # Graph layout cache
        self._node_rects: dict[str, pygame.Rect] = {}
        self._bin_positions: dict[str, tuple[int, int]] = {}
        self._layout_dirty = True

        # Viewport for virtual canvas (panning)
        self._viewport_x = 0.0
        self._viewport_y = 0.0
        self._canvas_x = 0.0
        self._canvas_y = 0.0
        self._canvas_w = float(GRAPH_W)
        self._canvas_h = float(GRAPH_H)
        self._viewport_initialized = False

        # Flow animation
        self._flow_shapes: list[FlowShape] = []
        self._floating_texts: list[FloatingText] = []

        # Connecting state
        self._connect_from = None

        # Training state
        self._canvas = DrawingCanvas()
        self._training_worker = None
        self._training_label_idx = 0
        self._training_new_class_input = ""
        self._training_typing_new_class = False

        # Gallery: all drawings ever submitted, reusable across workers
        self._gallery: list[tuple[pygame.Surface, torch.Tensor]] = []  # (thumb, tensor)
        self._show_gallery = False
        self._gallery_scroll = 0

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

        # Center viewport on root on first layout; clamp on subsequent
        if not self._viewport_initialized:
            self._center_viewport_on_root()
            self._viewport_initialized = True
        else:
            self._clamp_viewport()

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
            max_x = max(max_x, bx + BIN_W)
            max_y = max(max_y, by + BIN_H)

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

    def _vx(self, virtual_x: float) -> int:
        """Transform virtual canvas x to screen x."""
        return int(virtual_x - self._viewport_x) + GRAPH_X

    def _vy(self, virtual_y: float) -> int:
        """Transform virtual canvas y to screen y."""
        return int(virtual_y - self._viewport_y) + GRAPH_Y

    def _screen_to_virtual(self, sx: int, sy: int) -> tuple[float, float]:
        """Convert screen coords to virtual canvas coords."""
        return (sx - GRAPH_X + self._viewport_x, sy - GRAPH_Y + self._viewport_y)

    # ------------------------------------------------------------------
    # Minimap
    # ------------------------------------------------------------------

    def _minimap_params(self) -> tuple[float, float, float, float, float] | None:
        """Return (scale, mm_x, mm_y, mm_w, mm_h) or None if minimap not needed."""
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

    def _handle_minimap_click(self, mx, my) -> bool:
        """Handle click on minimap. Returns True if click was consumed."""
        params = self._minimap_params()
        if not params:
            return False
        scale, mm_x, mm_y, mm_w, mm_h = params
        if not (mm_x <= mx <= mm_x + mm_w and mm_y <= my <= mm_y + mm_h):
            return False
        cvx, cvy = self._minimap_to_virtual(mx, my, scale, mm_x, mm_y)
        self._viewport_x = cvx - GRAPH_W / 2
        self._viewport_y = cvy - GRAPH_H / 2
        self._clamp_viewport()
        return True

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
            mx_b, my_b = v2m(bx, by)
            pygame.draw.circle(surface, (60, 120, 60), (mx_b, my_b), 2)

        # Viewport rectangle
        vp_x1, vp_y1 = v2m(self._viewport_x, self._viewport_y)
        vp_x2, vp_y2 = v2m(self._viewport_x + GRAPH_W, self._viewport_y + GRAPH_H)
        pygame.draw.rect(surface, ACCENT,
                         (vp_x1, vp_y1, vp_x2 - vp_x1, vp_y2 - vp_y1), 2)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, events, dt):
        self._click_pos = None
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self._click_pos = e.pos

        # Minimap drag — pan viewport when mouse is held on minimap
        if pygame.mouse.get_pressed()[0]:
            mmx, mmy = pygame.mouse.get_pos()
            params = self._minimap_params()
            if params:
                scale, mm_x, mm_y, mm_w, mm_h = params
                if mm_x <= mmx <= mm_x + mm_w and mm_y <= mmy <= mm_y + mm_h:
                    cvx, cvy = self._minimap_to_virtual(mmx, mmy, scale, mm_x, mm_y)
                    self._viewport_x = cvx - GRAPH_W / 2
                    self._viewport_y = cvy - GRAPH_H / 2
                    self._clamp_viewport()

        # Auto-tick (only if factory has a root node)
        if not self.paused and self.world.graph.root_id:
            tick_interval = self.base_tick_interval / self.world.speed_level
            self.tick_timer += dt
            if self.tick_timer >= tick_interval:
                results = self.world.tick()
                self.tick_timer = 0
                self._spawn_flow_shapes(results)

        # Animate flow shapes and floating texts
        self._flow_shapes = [s for s in self._flow_shapes if s.alive]
        for shape in self._flow_shapes:
            shape.update(dt)
        self._floating_texts = [ft for ft in self._floating_texts if ft.alive]
        for ft in self._floating_texts:
            ft.update(dt)

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
                                self._training_worker._rebuild_head()
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

        # Suppress click for background UI when overlay is active
        if self.state in (TRAINING, DIALOG_TEXT, DIALOG_SELECT):
            self._bg_click_suppressed = True
        else:
            self._bg_click_suppressed = False

    def _spawn_flow_shapes(self, results):
        """Spawn animated shape thumbnails showing actual objects on actual paths."""
        graph = self.world.graph
        root_rect = self._node_rects.get(graph.root_id) if graph.root_id else None
        correct_set = set(id(o) for o, _ in results.correct)
        wrong_set = set(id(o) for o, _, _ in results.wrong)
        speed_mult = self.world.speed_level

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

            fs = FlowShape(src_rect.midright, dest, thumb, border, speed_mult)
            fs.t = -stagger  # stagger so shapes don't overlap
            stagger += 0.08 / speed_mult
            self._flow_shapes.append(fs)

        # Objects entering the system from the left edge
        all_entering = ([o for o, _ in results.correct] +
                        [o for o, _, _ in results.wrong] +
                        results.dropped)
        if root_rect:
            for i, obj in enumerate(all_entering[:5]):
                thumb = _tensor_to_thumb(obj.tensor)
                y_jitter = (i - len(all_entering[:5]) / 2) * (THUMB_SIZE + 4)
                entry_x = self._viewport_x  # left edge of visible area
                entry = (entry_x, root_rect.centery + int(y_jitter))
                fs = FlowShape(entry, root_rect.midleft, thumb, BORDER_TRANSIT, speed_mult)
                fs.t = -i * 0.06 / speed_mult
                self._flow_shapes.append(fs)

        # Floating money text at bins for correct/wrong
        reward = self.world.economy.CORRECT_REWARD
        penalty = self.world.economy.WRONG_PENALTY
        for obj, bin_name in results.correct:
            for key, (bx, by) in self._bin_positions.items():
                if key.endswith(f"BIN:{bin_name}"):
                    self._floating_texts.append(
                        FloatingText(bx + 50, by - 5, f"+${reward:.0f}", (40, 220, 40)))
                    break
        for obj, pred_bin, true_cat in results.wrong:
            for key, (bx, by) in self._bin_positions.items():
                if key.endswith(f"BIN:{pred_bin}"):
                    self._floating_texts.append(
                        FloatingText(bx + 50, by - 5, f"-${penalty:.0f}", (220, 50, 50)))
                    break

        # Dropped objects — fall off from where they were
        for obj, drop_nid in results.dropped_at:
            src_rect = self._node_rects.get(drop_nid)
            if not src_rect:
                continue
            thumb = _tensor_to_thumb(obj.tensor)
            # Fall downward from the node
            start = (src_rect.centerx + random.randint(-20, 20), src_rect.bottom)
            end = (start[0] + random.randint(-10, 10), start[1] + 120)
            fs = FlowShape(start, end, thumb, BORDER_WRONG, speed_mult)
            fs.duration = 0.8 / speed_mult  # slow fall, scaled
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

        mx, my = self._click_pos

        # 0. Check minimap click first
        if self._handle_minimap_click(mx, my):
            return

        # 1. If radial menu is showing, check if click hit a radial button
        #    and dispatch the action directly
        if self.selected_node and self.selected_node in self.world.graph.nodes:
            node = self.world.graph.nodes[self.selected_node]
            rect = self._node_rects.get(self.selected_node)
            if rect:
                cx = self._vx(rect.centerx)
                cy = self._vy(rect.centery)
                buttons = []
                for b in RADIAL_BUTTONS:
                    key = b[0]
                    if key == "train" and not node.worker:
                        continue
                    if key == "set_root" and node.node_id == self.world.graph.root_id:
                        continue
                    buttons.append(b)
                positions = self._compute_radial_positions(cx, cy, buttons)
                for bx, by, key, label, radius, color, icon_fn_name in positions:
                    dist = math.hypot(mx - bx, my - by)
                    if dist <= radius:
                        if key == "train":
                            self._action_train()
                        elif key == "connect_node":
                            self._action_connect()
                        elif key == "connect_bin":
                            self._action_connect_to_bin()
                        elif key == "set_root":
                            self._action_set_root()
                        elif key == "remove":
                            self._action_remove_node()
                        return

        # 2. Check if clicked a node rect — translate to virtual coords
        vmx, vmy = self._screen_to_virtual(mx, my)
        for nid, rect in self._node_rects.items():
            if rect.collidepoint(vmx, vmy):
                self.selected_node = nid
                return

        # 3. Clicked empty graph area — deselect
        graph_area = pygame.Rect(GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H)
        if graph_area.collidepoint(self._click_pos):
            self.selected_node = None

    def _handle_connecting_click(self):
        if not self._click_pos:
            return
        vmx, vmy = self._screen_to_virtual(*self._click_pos)
        for nid, rect in self._node_rects.items():
            if rect.collidepoint(vmx, vmy) and nid != self._connect_from:
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
        worker._rebuild_head()

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
        # Save to gallery
        self._gallery.append((surf84.copy(), tensor.clone()))
        self._canvas.clear()

    def _train_submit_from_gallery(self, idx):
        """Reuse a gallery image as a training example."""
        if not self._training_worker or not self._training_worker.class_names:
            return
        if idx >= len(self._gallery):
            return
        labels = self._training_worker.class_names
        if self._training_label_idx >= len(labels):
            self._training_label_idx = 0
        label = labels[self._training_label_idx]
        _, tensor = self._gallery[idx]
        self._training_worker.teach(tensor.clone(), label)

    def _train_remove_example(self, class_name, example_idx):
        """Remove a specific example from the worker's support set."""
        if not self._training_worker:
            return
        ss = self._training_worker._support_set.get(class_name, [])
        if 0 <= example_idx < len(ss):
            ss.pop(example_idx)
            self._training_worker._needs_readapt = True

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

        # Suppress background button clicks when overlay is active
        saved_click = self._click_pos
        if getattr(self, '_bg_click_suppressed', False):
            self._click_pos = None

        self._draw_top_bar(surface)
        self._draw_graph(surface)
        self._draw_side_panel(surface)
        result = self._draw_bottom_bar(surface)

        # Restore click for overlay use
        self._click_pos = saved_click

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

        # Flow shapes
        off_x = self._viewport_x - GRAPH_X
        off_y = self._viewport_y - GRAPH_Y
        for fs in self._flow_shapes:
            fs.draw_offset(surface, off_x, off_y)

        # Floating money text
        for ft in self._floating_texts:
            ft.draw_offset(surface, self.font, off_x, off_y)

        # Connecting mode indicator
        if self.state == CONNECTING and self._connect_from:
            msg = self.font.render(f"Click target node (from {self._connect_from})", True, ACCENT)
            surface.blit(msg, (GRAPH_PAD, GRAPH_Y + GRAPH_H - 25))

        # Remove clip
        surface.set_clip(None)

        # Radial context menu (drawn after clip removed so it can extend outside graph)
        if self.state == IDLE:
            self._draw_radial_menu(surface)

        # Minimap (drawn in screen coords, after clip removed)
        self._draw_minimap(surface)

    def _draw_side_panel(self, surface):
        panel_rect = pygame.Rect(SIDE_X, SIDE_Y, SIDE_W, GRAPH_H)
        pygame.draw.rect(surface, PANEL_BG, panel_rect)
        pygame.draw.line(surface, SEPARATOR, (SIDE_X, SIDE_Y), (SIDE_X, SIDE_Y + GRAPH_H), 2)

        y = SIDE_Y + 10

        # Build buttons (side by side)
        router_rect = pygame.Rect(BTN_X, y, 108, 38)
        specialist_rect = pygame.Rect(BTN_X + 117, y, 108, 38)
        if self._draw_btn_with_icon(surface, router_rect, "Router", (100, 150, 200), _draw_icon_router):
            self._action_add_router()
        if self._draw_btn_with_icon(surface, specialist_rect, "Specialist", (80, 160, 160), _draw_icon_specialist):
            self._action_add_specialist()
        y += 38 + 12

        # Speed upgrade (pill shape)
        cost = self.world.get_speed_upgrade_cost()
        spd = self.world.speed_level
        label = f"Spd {spd}\u2192{spd+1}  (${cost:.0f})"
        can_afford = self.world.economy.can_afford(cost)
        pill_color = (100, 170, 100) if can_afford else (140, 140, 140)
        pill_rect = pygame.Rect(BTN_X, y, BTN_W, 34)
        mouse = pygame.mouse.get_pos()
        hover = pill_rect.collidepoint(mouse)
        c = tuple(min(255, v + 25) for v in pill_color) if hover else pill_color
        pygame.draw.rect(surface, c, pill_rect, border_radius=17)
        border_color = tuple(max(0, v - 40) for v in pill_color)
        pygame.draw.rect(surface, border_color, pill_rect, 2, border_radius=17)
        t = self.font_sm.render(label, True, TEXT_DARK)
        tx = pill_rect.x + (pill_rect.width - t.get_width()) // 2
        ty = pill_rect.y + (pill_rect.height - t.get_height()) // 2
        surface.blit(t, (tx, ty))
        if self._click_pos is not None and pill_rect.collidepoint(self._click_pos):
            if self.world.buy_speed_upgrade():
                self._show_status(f"Speed upgraded to Lv.{self.world.speed_level}!")
        y += 34 + 10

        # Separator
        pygame.draw.line(surface, SEPARATOR, (BTN_X, y), (BTN_X + BTN_W, y), 1)
        y += 10

        # Node info
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

        # Factory stats
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

    def _compute_radial_positions(self, cx, cy, buttons):
        """Compute radial button positions, adapting arc when near graph edges.

        Returns list of (bx, by, key, label, radius, color, icon_fn_name).
        """
        if not buttons:
            return []

        max_btn_r = max(b[2] for b in buttons)
        margin = RADIAL_RING_R + max_btn_r + 4

        # Check which directions are blocked
        space_left = cx - GRAPH_X
        space_right = (GRAPH_X + GRAPH_W) - cx
        space_up = cy - GRAPH_Y
        space_down = (GRAPH_Y + GRAPH_H) - cy

        blocked_left = space_left < margin
        blocked_right = space_right < margin
        blocked_up = space_up < margin
        blocked_down = space_down < margin

        any_blocked = blocked_left or blocked_right or blocked_up or blocked_down

        if not any_blocked:
            # Use default angles from button specs
            result = []
            for key, label, radius, color, icon_fn_name, angle_deg in buttons:
                angle_rad = math.radians(angle_deg)
                bx = cx + int(RADIAL_RING_R * math.cos(angle_rad))
                by = cy + int(RADIAL_RING_R * math.sin(angle_rad))
                result.append((bx, by, key, label, radius, color, icon_fn_name))
            return result

        # Find valid angular range by sampling every 5 degrees
        valid_angles = []
        for deg in range(0, 360, 5):
            rad = math.radians(deg)
            test_x = cx + RADIAL_RING_R * math.cos(rad)
            test_y = cy + RADIAL_RING_R * math.sin(rad)
            # Check if this point (plus max button radius + margin) fits in graph
            if (test_x - max_btn_r - 2 >= GRAPH_X and
                test_x + max_btn_r + 2 <= GRAPH_X + GRAPH_W and
                test_y - max_btn_r - 2 >= GRAPH_Y and
                test_y + max_btn_r + 2 <= GRAPH_Y + GRAPH_H):
                valid_angles.append(deg)

        if not valid_angles:
            # Fallback: place at default angles anyway
            result = []
            for key, label, radius, color, icon_fn_name, angle_deg in buttons:
                angle_rad = math.radians(angle_deg)
                bx = cx + int(RADIAL_RING_R * math.cos(angle_rad))
                by = cy + int(RADIAL_RING_R * math.sin(angle_rad))
                result.append((bx, by, key, label, radius, color, icon_fn_name))
            return result

        # Distribute buttons evenly across the valid arc
        n = len(buttons)
        if n == 1:
            mid_deg = valid_angles[len(valid_angles) // 2]
            spread = [mid_deg]
            ring_r = RADIAL_RING_R
        else:
            arc_start = valid_angles[0]
            arc_end = valid_angles[-1]
            arc_span = arc_end - arc_start
            if arc_span <= 0:
                arc_span = 360
            step = arc_span / (n - 1) if n > 1 else 0
            spread = [arc_start + i * step for i in range(n)]

            # Check if buttons would overlap and increase ring radius if needed
            ring_r = RADIAL_RING_R
            min_chord = max_btn_r * 2 + 8  # diameter + gap
            if step > 0:
                # chord length between adjacent buttons at current ring_r
                chord = 2 * ring_r * math.sin(math.radians(step / 2))
                if chord < min_chord:
                    # Increase ring_r so chord >= min_chord
                    ring_r = int(min_chord / (2 * math.sin(math.radians(step / 2)))) + 1

        result = []
        for i, (key, label, radius, color, icon_fn_name, _default_deg) in enumerate(buttons):
            angle_rad = math.radians(spread[i])
            bx = cx + int(ring_r * math.cos(angle_rad))
            by = cy + int(ring_r * math.sin(angle_rad))
            result.append((bx, by, key, label, radius, color, icon_fn_name))
        return result

    def _draw_radial_menu(self, surface):
        """Draw the radial context menu around the selected node."""
        if not self.selected_node:
            return
        if self.selected_node not in self.world.graph.nodes:
            return

        node = self.world.graph.nodes[self.selected_node]
        rect = self._node_rects.get(self.selected_node)
        if not rect:
            return

        cx = self._vx(rect.centerx)
        cy = self._vy(rect.centery)

        # Filter buttons based on node state
        buttons = []
        for b in RADIAL_BUTTONS:
            key = b[0]
            if key == "train" and not node.worker:
                continue
            if key == "set_root" and node.node_id == self.world.graph.root_id:
                continue
            buttons.append(b)

        positions = self._compute_radial_positions(cx, cy, buttons)
        if not positions:
            return

        mouse_pos = pygame.mouse.get_pos()
        mx, my = mouse_pos

        icon_fns = {
            "_draw_icon_mallet": _draw_icon_mallet,
            "_draw_icon_arrow_node": _draw_icon_arrow_node,
            "_draw_icon_arrow_bin": _draw_icon_arrow_bin,
            "_draw_icon_x": _draw_icon_x,
            "_draw_icon_star": _draw_icon_star,
        }

        hovered_idx = None

        for i, (bx, by, key, label, radius, color, icon_fn_name) in enumerate(positions):
            dist_mouse = math.hypot(mx - bx, my - by)
            is_hover = dist_mouse <= radius

            if is_hover:
                hovered_idx = i

            # 1. Drop shadow
            shadow_surf = pygame.Surface((radius * 2 + 8, radius * 2 + 8), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, (0, 0, 0, 60),
                               (radius + 4, radius + 4), radius)
            surface.blit(shadow_surf, (bx - radius - 4 + 2, by - radius - 4 + 2))

            # 2. Filled circle (lighten by 25 on hover)
            if is_hover:
                draw_color = (min(255, color[0] + 25),
                              min(255, color[1] + 25),
                              min(255, color[2] + 25))
            else:
                draw_color = color
            pygame.draw.circle(surface, draw_color, (bx, by), radius)

            # 3. Dark border (2px)
            pygame.draw.circle(surface, (40, 40, 40), (bx, by), radius, 2)

            # 4. Draw icon
            icon_fn = icon_fns.get(icon_fn_name)
            if icon_fn:
                icon_fn(surface, bx, by, radius)

        # 5. Tooltip for hovered button
        if hovered_idx is not None:
            bx, by, key, label, radius, color, icon_fn_name = positions[hovered_idx]
            # Render tooltip text
            tip_surf = self.font_sm.render(label, True, (255, 255, 255))
            tw, th = tip_surf.get_size()
            pad = 6
            tip_w = tw + pad * 2
            tip_h = th + pad * 2

            # Position outward from node center
            dx = bx - cx
            dy = by - cy
            dist = math.hypot(dx, dy)
            if dist > 0:
                nx, ny = dx / dist, dy / dist
            else:
                nx, ny = 0, -1
            tip_x = int(bx + nx * (radius + 8)) - tip_w // 2
            tip_y = int(by + ny * (radius + 8)) - tip_h // 2

            # Clamp to graph bounds
            tip_x = max(GRAPH_X + 2, min(tip_x, GRAPH_X + GRAPH_W - tip_w - 2))
            tip_y = max(GRAPH_Y + 2, min(tip_y, GRAPH_Y + GRAPH_H - tip_h - 2))

            # Draw tooltip background
            tip_rect = pygame.Rect(tip_x, tip_y, tip_w, tip_h)
            pygame.draw.rect(surface, (50, 50, 50), tip_rect, border_radius=4)
            surface.blit(tip_surf, (tip_x + pad, tip_y + pad))

    def _draw_bottom_bar(self, surface):
        by = H - BOTTOM_H
        pygame.draw.rect(surface, BOTTOM_BG, (0, by, W, BOTTOM_H))

        # -- Left side: info text (unchanged) --
        cats = self.world.active_categories
        cat_str = ", ".join(cats[:6])
        if len(cats) > 6:
            cat_str += f" +{len(cats)-6}"
        t = self.font_sm.render(f"Active: {cat_str}", True, TEXT_LIGHT)
        surface.blit(t, (15, by + 8))

        t = self.font_sm.render(f"Objects/tick: {self.world.objects_per_tick}", True, TEXT_LIGHT)
        surface.blit(t, (15, by + 28))

        mouse = pygame.mouse.get_pos()

        # -- Right side: Quit button --
        quit_rect = pygame.Rect(W - 175, by + 10, 55, 30)
        quit_hover = quit_rect.collidepoint(mouse)
        quit_color = (170, 140, 120) if quit_hover else (150, 120, 100)
        pygame.draw.rect(surface, quit_color, quit_rect, border_radius=6)
        pygame.draw.rect(surface, (80, 60, 50), quit_rect, 1, border_radius=6)
        qt = self.font_sm.render("Quit", True, TEXT_LIGHT)
        surface.blit(qt, (quit_rect.x + (quit_rect.width - qt.get_width()) // 2,
                          quit_rect.y + (quit_rect.height - qt.get_height()) // 2))
        if self._click_pos is not None and quit_rect.collidepoint(self._click_pos):
            return "MENU"

        # -- Status label --
        label = "Paused" if self.paused else "Running"
        color = RED if self.paused else GREEN
        lt = self.font_sm.render(label, True, color)
        surface.blit(lt, (W - 110, by + 15))

        # -- Circular play/pause button --
        circle_cx = W - 37
        circle_cy = by + 25
        circle_r = 18

        circle_hover = (mouse[0] - circle_cx) ** 2 + (mouse[1] - circle_cy) ** 2 <= circle_r ** 2

        if self.paused:
            fill = (230, 80, 80) if circle_hover else (200, 60, 60)
        else:
            fill = (80, 195, 80) if circle_hover else (60, 170, 60)

        # Drop shadow
        shadow_surf = pygame.Surface((circle_r * 2 + 8, circle_r * 2 + 8), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surf, (0, 0, 0, 60),
                           (circle_r + 2, circle_r + 2), circle_r)
        surface.blit(shadow_surf, (circle_cx - circle_r - 2 + 2,
                                   circle_cy - circle_r - 2 + 2))

        # Filled circle + border
        pygame.draw.circle(surface, fill, (circle_cx, circle_cy), circle_r)
        pygame.draw.circle(surface, (40, 40, 40), (circle_cx, circle_cy), circle_r, 2)

        # Icon
        if self.paused:
            _draw_icon_play(surface, circle_cx, circle_cy, circle_r)
        else:
            _draw_icon_pause(surface, circle_cx, circle_cy, circle_r)

        # Click detection (distance check)
        if self._click_pos is not None:
            dx = self._click_pos[0] - circle_cx
            dy = self._click_pos[1] - circle_cy
            if dx * dx + dy * dy <= circle_r * circle_r:
                self.paused = not self.paused

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

        # Title + live coins
        t = self.font_lg.render(f"Training: {worker.name}", True, title_color)
        surface.blit(t, (W // 2 - t.get_width() // 2, 20))
        coins = self.world.economy.coins
        cpt = self.world.economy.coins_per_tick
        ct = self.font.render(f"${coins:.0f}", True, (255, 220, 80))
        surface.blit(ct, (W - 120, 8))
        cpt_color = (40, 220, 40) if cpt > 0 else (220, 50, 50) if cpt < 0 else title_color
        cpt_t = self.font_sm.render(f"{cpt:+.1f}/t", True, cpt_color)
        surface.blit(cpt_t, (W - 120, 28))

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

        # Support set preview with thumbnails + delete buttons
        if worker.class_names:
            label_y += 5
            t = self.font_sm.render("Support set:", True, subtle)
            surface.blit(t, (label_x, label_y))
            label_y += 16
            thumb_sz = 28
            for cls_name in worker.class_names:
                examples = worker._support_set.get(cls_name, [])
                t = self.font_sm.render(f"{cls_name} ({len(examples)}):", True, title_color)
                surface.blit(t, (label_x, label_y))
                label_y += 15
                # Show thumbnails in a row
                for ei, ex_tensor in enumerate(examples[:6]):  # cap display at 6
                    tx = label_x + ei * (thumb_sz + 3)
                    ty = label_y
                    # Convert tensor to small surface
                    arr = (ex_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(
                        __import__('numpy').uint8)
                    thumb_surf = pygame.surfarray.make_surface(arr.transpose(1, 0, 2))
                    thumb_surf = pygame.transform.smoothscale(thumb_surf, (thumb_sz, thumb_sz))
                    surface.blit(thumb_surf, (tx, ty))
                    # Delete button (small X)
                    xr = pygame.Rect(tx + thumb_sz - 10, ty, 10, 10)
                    pygame.draw.rect(surface, (180, 50, 50), xr)
                    xt = self.font_sm.render("x", True, (255, 255, 255))
                    surface.blit(xt, (xr.x + 1, xr.y - 2))
                    if self._click_pos and xr.collidepoint(self._click_pos):
                        self._train_remove_example(cls_name, ei)
                if len(examples) > 6:
                    t = self.font_sm.render(f"+{len(examples)-6}", True, subtle)
                    surface.blit(t, (label_x + 6 * (thumb_sz + 3), label_y + 8))
                label_y += thumb_sz + 5 if examples else 2

        # Gallery button + gallery view
        label_y += 5
        gallery_label = f"Gallery ({len(self._gallery)})"
        if self._draw_btn_raw(surface, pygame.Rect(label_x, label_y, 180, 26),
                              gallery_label, (100, 120, 160)):
            self._show_gallery = not self._show_gallery
        label_y += 30

        if self._show_gallery and self._gallery:
            t = self.font_sm.render("Click to add as current class:", True, subtle)
            surface.blit(t, (label_x, label_y))
            label_y += 15
            gx = label_x
            for gi, (g_surf, g_tensor) in enumerate(self._gallery):
                if gi >= 12:  # cap display
                    break
                col = gi % 4
                row = gi // 4
                tx = label_x + col * (thumb_sz + 3)
                ty = label_y + row * (thumb_sz + 3)
                scaled = pygame.transform.smoothscale(g_surf, (thumb_sz, thumb_sz))
                surface.blit(scaled, (tx, ty))
                pygame.draw.rect(surface, (100, 100, 100), (tx, ty, thumb_sz, thumb_sz), 1)
                if self._click_pos and pygame.Rect(tx, ty, thumb_sz, thumb_sz).collidepoint(self._click_pos):
                    self._train_submit_from_gallery(gi)
            rows = min(3, (min(len(self._gallery), 12) + 3) // 4)
            label_y += rows * (thumb_sz + 3) + 5

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

    # DrawingCanvas.draw_toolbar expects a renderer-like interface
    def draw_text(self, x, y, text, color=TEXT_DARK, font=None):
        f = font or self.font_sm
        t = f.render(text, True, color)
        pygame.display.get_surface().blit(t, (x, y))

    def draw_button(self, rect, text, color=(100, 150, 220)):
        return self._draw_btn_raw(pygame.display.get_surface(), rect, text, color)
