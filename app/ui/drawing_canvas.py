"""Shared drawing canvas with brush, fill, line, and stamp tools."""

import math
import pygame
import numpy as np

CANVAS_SCALE = 4  # draw on 336x336, downscale to 84x84
CANVAS_SIZE = 84 * CANVAS_SCALE
BRUSH_SIZES = [2, 4, 8, 12]
DRAW_COLORS = [
    ("black", (0, 0, 0)),
    ("red", (220, 50, 50)),
    ("blue", (50, 80, 220)),
    ("green", (50, 180, 50)),
    ("orange", (240, 140, 30)),
    ("purple", (150, 50, 200)),
    ("white", (255, 255, 255)),
]

TOOL_BRUSH = 0
TOOL_FILL = 1
TOOL_LINE = 2
TOOL_STAMP = 3

STAMP_SHAPES = ["circle", "square", "triangle", "star", "diamond"]


def _stamp_points(shape, cx, cy, r):
    """Return polygon points for a stamp shape centered at (cx, cy) with radius r."""
    if shape == "circle":
        return None  # handled separately with pygame.draw.circle
    elif shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    elif shape == "triangle":
        return [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
    elif shape == "star":
        pts = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            rad = r if i % 2 == 0 else r * 0.4
            pts.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
        return pts
    elif shape == "diamond":
        return [(cx, cy - r), (cx + r * 0.6, cy), (cx, cy + r), (cx - r * 0.6, cy)]
    return None


class DrawingCanvas:
    """Reusable drawing canvas with brush, fill, line, and stamp tools."""

    def __init__(self):
        self.canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        self.canvas.fill((255, 255, 255))
        self._drawing = False
        self._last_mouse_pos = None
        self.brush_size_idx = 1
        self.color_idx = 0
        self.tool = TOOL_BRUSH
        self.stamp_idx = 0

        # Line tool state
        self._line_start = None   # first click position (canvas-local)
        self._line_chain = False  # shift held = chain mode

    def clear(self):
        self.canvas.fill((255, 255, 255))
        self._drawing = False
        self._last_mouse_pos = None
        self._line_start = None

    def get_surface_84(self) -> pygame.Surface:
        """Return canvas downscaled to 84x84."""
        return pygame.transform.smoothscale(self.canvas, (84, 84))

    @property
    def current_color(self):
        return DRAW_COLORS[self.color_idx][1]

    @property
    def current_brush_radius(self):
        return BRUSH_SIZES[self.brush_size_idx] * CANVAS_SCALE // 2

    def _flood_fill(self, x, y, fill_color):
        """Stack-based flood fill using numpy for speed."""
        arr = pygame.surfarray.pixels3d(self.canvas)
        target = arr[x, y].copy()
        fill = np.array(fill_color, dtype=np.uint8)
        if np.array_equal(target, fill):
            return
        tolerance = 30
        w, h = arr.shape[0], arr.shape[1]
        visited = np.zeros((w, h), dtype=bool)
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                continue
            if visited[cx, cy]:
                continue
            diff = np.abs(arr[cx, cy].astype(int) - target.astype(int))
            if diff.max() > tolerance:
                continue
            visited[cx, cy] = True
            arr[cx, cy] = fill
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))
        del arr

    def _draw_stamp(self, cx, cy):
        """Draw a stamp shape at (cx, cy) on the canvas."""
        shape = STAMP_SHAPES[self.stamp_idx]
        r = self.current_brush_radius * 2
        color = self.current_color
        if shape == "circle":
            pygame.draw.circle(self.canvas, color, (cx, cy), r)
        else:
            pts = _stamp_points(shape, cx, cy, r)
            if pts:
                pygame.draw.polygon(self.canvas, color, pts)

    def handle_events(self, events, canvas_rect):
        """Process mouse events for drawing. canvas_rect is the screen-space rect."""
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if canvas_rect.collidepoint(e.pos):
                    local = (e.pos[0] - canvas_rect.x, e.pos[1] - canvas_rect.y)

                    if self.tool == TOOL_FILL:
                        self._flood_fill(local[0], local[1], self.current_color)

                    elif self.tool == TOOL_STAMP:
                        self._draw_stamp(local[0], local[1])

                    elif self.tool == TOOL_LINE:
                        if self._line_start is None:
                            self._line_start = local
                        else:
                            # Draw line from start to here
                            w = self.current_brush_radius * 2
                            pygame.draw.line(self.canvas, self.current_color,
                                             self._line_start, local, max(1, w))
                            # Endcaps
                            r = max(1, self.current_brush_radius)
                            pygame.draw.circle(self.canvas, self.current_color, self._line_start, r)
                            pygame.draw.circle(self.canvas, self.current_color, local, r)
                            # Chain if shift held
                            mods = pygame.key.get_mods()
                            if mods & pygame.KMOD_SHIFT:
                                self._line_start = local
                            else:
                                self._line_start = None

                    else:  # TOOL_BRUSH
                        self._drawing = True
                        self._last_mouse_pos = local
                        pygame.draw.circle(self.canvas, self.current_color, local, self.current_brush_radius)

            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                if self.tool == TOOL_BRUSH:
                    self._drawing = False
                    self._last_mouse_pos = None

            elif e.type == pygame.MOUSEMOTION and self._drawing and self.tool == TOOL_BRUSH:
                if canvas_rect.collidepoint(e.pos):
                    cur = (e.pos[0] - canvas_rect.x, e.pos[1] - canvas_rect.y)
                    r = self.current_brush_radius
                    if self._last_mouse_pos:
                        pygame.draw.line(self.canvas, self.current_color, self._last_mouse_pos, cur, r * 2)
                    pygame.draw.circle(self.canvas, self.current_color, cur, r)
                    self._last_mouse_pos = cur
                else:
                    self._drawing = False
                    self._last_mouse_pos = None

    def draw_toolbar(self, surface, renderer, x, y):
        """Draw the tool palette starting at (x, y). Returns the total height used."""
        r = renderer
        cur_y = y

        # Tool row 1: Brush, Fill
        r.draw_text(x, cur_y, "Tool:", font=r.font_sm)
        cur_y += 16
        tools = [
            (TOOL_BRUSH, "Brush"),
            (TOOL_FILL, "Fill"),
            (TOOL_LINE, "Line"),
            (TOOL_STAMP, "Stamp"),
        ]
        for i, (tid, tname) in enumerate(tools):
            col = i % 2
            row = i // 2
            color = (80, 80, 80) if self.tool == tid else (150, 150, 155)
            bx = x + col * 78
            by = cur_y + row * 32
            if r.draw_button(pygame.Rect(bx, by, 72, 26), tname, color):
                self.tool = tid
                self._line_start = None
        cur_y += 32 * 2 + 6

        # Stamp shape picker (only if stamp selected)
        if self.tool == TOOL_STAMP:
            r.draw_text(x, cur_y, "Shape:", font=r.font_sm)
            cur_y += 16
            for i, sname in enumerate(STAMP_SHAPES):
                col = i % 3
                row = i // 3
                color = (80, 80, 80) if i == self.stamp_idx else (150, 150, 155)
                bx = x + col * 56
                by = cur_y + row * 28
                if r.draw_button(pygame.Rect(bx, by, 50, 24), sname[:5], color):
                    self.stamp_idx = i
            rows = (len(STAMP_SHAPES) + 2) // 3
            cur_y += rows * 28 + 6

        # Line tool hint
        if self.tool == TOOL_LINE:
            if self._line_start:
                r.draw_text(x, cur_y, "Click end pt", font=r.font_sm)
                r.draw_text(x, cur_y + 14, "Shift=chain", font=r.font_sm)
            else:
                r.draw_text(x, cur_y, "Click start pt", font=r.font_sm)
            cur_y += 30

        # Brush size (for brush, line, and stamp)
        if self.tool in (TOOL_BRUSH, TOOL_LINE, TOOL_STAMP):
            r.draw_text(x, cur_y, "Size:", font=r.font_sm)
            cur_y += 16
            for i, sz in enumerate(BRUSH_SIZES):
                color = (80, 80, 80) if i == self.brush_size_idx else (150, 150, 155)
                btn_rect = pygame.Rect(x + i * 42, cur_y, 36, 24)
                if r.draw_button(btn_rect, str(sz), color):
                    self.brush_size_idx = i
            cur_y += 32

        # Color palette
        r.draw_text(x, cur_y, "Color:", font=r.font_sm)
        cur_y += 16
        for i, (cname, cval) in enumerate(DRAW_COLORS):
            col = i % 4
            row = i // 4
            btn_rect = pygame.Rect(x + col * 44, cur_y + row * 28, 38, 24)
            border = (255, 255, 0) if i == self.color_idx else (0, 0, 0)
            pygame.draw.rect(surface, cval, btn_rect)
            pygame.draw.rect(surface, border, btn_rect, 3 if i == self.color_idx else 1)
            if r._click_pos and btn_rect.collidepoint(r._click_pos):
                self.color_idx = i
        rows = (len(DRAW_COLORS) + 3) // 4
        cur_y += rows * 28 + 8

        # Clear
        if r.draw_button(pygame.Rect(x, cur_y, 80, 26), "Clear", (200, 150, 150)):
            self.clear()
        cur_y += 34

        return cur_y - y
