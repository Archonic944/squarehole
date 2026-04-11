"""Shared drawing canvas with brush, fill, and color tools."""

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


class DrawingCanvas:
    """Reusable drawing canvas with brush and fill tools."""

    def __init__(self):
        self.canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        self.canvas.fill((255, 255, 255))
        self._drawing = False
        self._last_mouse_pos = None
        self.brush_size_idx = 1
        self.color_idx = 0
        self.tool = TOOL_BRUSH

    def clear(self):
        self.canvas.fill((255, 255, 255))
        self._drawing = False
        self._last_mouse_pos = None

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
        arr = pygame.surfarray.pixels3d(self.canvas)  # (W, H, 3) view
        target = arr[x, y].copy()
        fill = np.array(fill_color, dtype=np.uint8)

        if np.array_equal(target, fill):
            return

        # Tolerance for anti-aliased edges
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

        del arr  # release surface lock

    def handle_events(self, events, canvas_rect):
        """Process mouse events for drawing. canvas_rect is the screen-space rect."""
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if canvas_rect.collidepoint(e.pos):
                    local = (e.pos[0] - canvas_rect.x, e.pos[1] - canvas_rect.y)
                    if self.tool == TOOL_FILL:
                        self._flood_fill(local[0], local[1], self.current_color)
                    else:
                        self._drawing = True
                        self._last_mouse_pos = local
                        pygame.draw.circle(self.canvas, self.current_color, local, self.current_brush_radius)

            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
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
        """Draw the tool palette (brush/fill, colors, sizes) starting at (x, y).
        Returns the total height used.
        """
        r = renderer
        cur_y = y

        # Tool selection
        r.draw_text(x, cur_y, "Tool:", font=r.font_sm)
        cur_y += 18
        brush_color = (80, 80, 80) if self.tool == TOOL_BRUSH else (180, 180, 180)
        fill_color = (80, 80, 80) if self.tool == TOOL_FILL else (180, 180, 180)
        if r.draw_button(pygame.Rect(x, cur_y, 70, 28), "Brush", brush_color):
            self.tool = TOOL_BRUSH
        if r.draw_button(pygame.Rect(x + 78, cur_y, 70, 28), "Fill", fill_color):
            self.tool = TOOL_FILL
        cur_y += 38

        # Brush size (only relevant for brush tool)
        if self.tool == TOOL_BRUSH:
            r.draw_text(x, cur_y, "Size:", font=r.font_sm)
            cur_y += 18
            for i, sz in enumerate(BRUSH_SIZES):
                color = (80, 80, 80) if i == self.brush_size_idx else (180, 180, 180)
                btn_rect = pygame.Rect(x + i * 45, cur_y, 38, 28)
                if r.draw_button(btn_rect, str(sz), color):
                    self.brush_size_idx = i
            cur_y += 38

        # Color palette
        r.draw_text(x, cur_y, "Color:", font=r.font_sm)
        cur_y += 18
        for i, (cname, cval) in enumerate(DRAW_COLORS):
            col = i % 4
            row = i // 4
            btn_rect = pygame.Rect(x + col * 52, cur_y + row * 32, 45, 26)
            border = (255, 255, 0) if i == self.color_idx else (0, 0, 0)
            pygame.draw.rect(surface, cval, btn_rect)
            pygame.draw.rect(surface, border, btn_rect, 3 if i == self.color_idx else 1)
            if r._click_pos and btn_rect.collidepoint(r._click_pos):
                self.color_idx = i
        rows = (len(DRAW_COLORS) + 3) // 4
        cur_y += rows * 32 + 10

        # Clear
        if r.draw_button(pygame.Rect(x, cur_y, 90, 30), "Clear", (200, 150, 150)):
            self.clear()
        cur_y += 40

        return cur_y - y
