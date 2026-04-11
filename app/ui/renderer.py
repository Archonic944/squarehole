"""PyGame renderer for BabyBrain UI elements."""

import math
import pygame

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
DARK_GRAY = (100, 100, 100)
LIGHT_BLUE = (200, 220, 255)
BLUE = (60, 120, 220)
GREEN = (80, 200, 80)
YELLOW = (230, 210, 40)
RED = (220, 60, 60)
BG_COLOR = (240, 235, 225)


class Renderer:
    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self.font = pygame.font.SysFont("Arial", 20)
        self.font_sm = pygame.font.SysFont("Arial", 16)
        self.font_lg = pygame.font.SysFont("Arial", 32, bold=True)
        self._click_pos = None
        self._milestone_timer = 0.0
        self._milestone_text = ""
        self._thinking_frame = 0

    def begin_frame(self, events: list):
        """Call at start of frame to capture click state."""
        self._click_pos = None
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self._click_pos = e.pos

    def clear(self, color=BG_COLOR):
        self.surface.fill(color)

    # --- Baby face ---

    def draw_baby(self, x: int, y: int, mood: float, thinking: bool = False):
        """Draw a simple circle face with mood-based expression."""
        radius = 60
        # Head
        pygame.draw.circle(self.surface, (255, 220, 180), (x, y), radius)
        pygame.draw.circle(self.surface, BLACK, (x, y), radius, 2)

        # Eyes
        eye_y = y - 15
        for ex in (x - 20, x + 20):
            pygame.draw.circle(self.surface, BLACK, (ex, eye_y), 6)
            pygame.draw.circle(self.surface, WHITE, (ex - 2, eye_y - 2), 2)

        # Mouth based on mood
        if mood > 0.6:
            # Happy smile
            pygame.draw.arc(self.surface, BLACK, (x - 25, y, 50, 30), math.pi, 2 * math.pi, 2)
        elif mood > 0.3:
            # Neutral line
            pygame.draw.line(self.surface, BLACK, (x - 18, y + 15), (x + 18, y + 15), 2)
        else:
            # Sad frown
            pygame.draw.arc(self.surface, BLACK, (x - 25, y + 10, 50, 30), 0, math.pi, 2)

        # Thinking animation
        if thinking:
            self._thinking_frame = (self._thinking_frame + 1) % 90
            dots = "." * (self._thinking_frame // 30 + 1)
            txt = self.font.render(dots, True, DARK_GRAY)
            self.surface.blit(txt, (x + radius + 10, y - 10))

    # --- Bars ---

    def draw_mood_bar(self, x: int, y: int, mood: float, width: int = 160):
        """Colored bar: green -> yellow -> red."""
        pygame.draw.rect(self.surface, DARK_GRAY, (x, y, width, 18), 2)
        fill_w = int(mood * (width - 4))
        if mood > 0.6:
            c = GREEN
        elif mood > 0.3:
            c = YELLOW
        else:
            c = RED
        pygame.draw.rect(self.surface, c, (x + 2, y + 2, fill_w, 14))
        label = self.font_sm.render("Mood", True, BLACK)
        self.surface.blit(label, (x, y - 18))

    def draw_energy_bar(self, x: int, y: int, energy: float, width: int = 160):
        pygame.draw.rect(self.surface, DARK_GRAY, (x, y, width, 18), 2)
        fill_w = int(energy * (width - 4))
        pygame.draw.rect(self.surface, BLUE, (x + 2, y + 2, fill_w, 14))
        label = self.font_sm.render("Energy", True, BLACK)
        self.surface.blit(label, (x, y - 18))

    def draw_age(self, x: int, y: int, age_days: int):
        txt = self.font.render(f"Age: {age_days} day{'s' if age_days != 1 else ''}", True, BLACK)
        self.surface.blit(txt, (x, y))

    # --- Milestone notification ---

    def show_milestone(self, text: str):
        self._milestone_text = text
        self._milestone_timer = 3.0  # seconds

    def draw_milestone_notification(self, dt: float):
        if self._milestone_timer <= 0:
            return
        self._milestone_timer -= dt
        alpha = min(1.0, self._milestone_timer / 0.5, (3.0 - (3.0 - self._milestone_timer)) / 0.5)
        alpha = max(0.0, min(1.0, alpha))
        if alpha <= 0:
            return
        txt = self.font_lg.render(f"★ {self._milestone_text}", True, (50, 50, 180))
        txt.set_alpha(int(alpha * 255))
        w = txt.get_width()
        self.surface.blit(txt, (400 - w // 2, 50))

    # --- Button ---

    def draw_button(self, rect: pygame.Rect, text: str, color=(100, 150, 220)) -> bool:
        """Draw a button and return True if clicked this frame."""
        mouse = pygame.mouse.get_pos()
        hover = rect.collidepoint(mouse)
        c = tuple(min(255, v + 30) for v in color) if hover else color
        pygame.draw.rect(self.surface, c, rect, border_radius=8)
        pygame.draw.rect(self.surface, BLACK, rect, 2, border_radius=8)
        txt = self.font.render(text, True, BLACK)
        tx = rect.x + (rect.width - txt.get_width()) // 2
        ty = rect.y + (rect.height - txt.get_height()) // 2
        self.surface.blit(txt, (tx, ty))
        return self._click_pos is not None and rect.collidepoint(self._click_pos)

    # --- Text ---

    def draw_text(self, x: int, y: int, text: str, color=BLACK, font=None):
        f = font or self.font
        surf = f.render(text, True, color)
        self.surface.blit(surf, (x, y))

    def draw_text_centered(self, y: int, text: str, color=BLACK, font=None):
        f = font or self.font
        surf = f.render(text, True, color)
        self.surface.blit(surf, (400 - surf.get_width() // 2, y))
