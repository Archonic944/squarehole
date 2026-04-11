"""UI for 'What's This?' mini-game."""

import pygame
from app.ui.renderer import Renderer
from app.ui.game_objects import generate_toy_box, surface_to_tensor
from app.ui.drawing_canvas import DrawingCanvas, CANVAS_SIZE

# States
SELECTING_OBJECT = 0
BABY_GUESSING = 1
PLAYER_CONFIRMING = 2
LEARNING = 3
DRAWING = 4

PRESET_CLASSES = ["circle", "square", "triangle", "star", "diamond", "blob", "thing"]


class WhatsThisUI:
    def __init__(self, renderer: Renderer, skill, baby):
        self.renderer = renderer
        self.skill = skill
        self.baby = baby
        self.state = SELECTING_OBJECT
        self.toy_box = generate_toy_box()
        self.selected_index = None
        self.selected_surface = None  # for drawn images
        self.guess_label = ""
        self.guess_confidence = 0.0
        self.thinking_timer = 0.0
        self.accuracy_log: list[bool] = []
        self.feedback_text = ""
        self.feedback_timer = 0.0
        self._new_class_input = ""
        self._typing_new_class = False
        self._canvas = DrawingCanvas()

    def _grid_rects(self) -> list[pygame.Rect]:
        rects = []
        cols = 5
        margin_x, margin_y = 50, 100
        pad = 8
        for i, _ in enumerate(self.toy_box):
            col = i % cols
            row = i // cols
            x = margin_x + col * (84 + pad)
            y = margin_y + row * (84 + pad)
            rects.append(pygame.Rect(x, y, 84, 84))
        return rects

    def _get_selected_surface(self):
        if self.selected_surface is not None:
            return self.selected_surface
        return self.toy_box[self.selected_index][0]

    def update(self, events: list):
        dt = 1.0 / 30.0
        if self.feedback_timer > 0:
            self.feedback_timer -= dt

        if self.state == SELECTING_OBJECT:
            pass

        elif self.state == DRAWING:
            canvas_rect = pygame.Rect(230, 80, CANVAS_SIZE, CANVAS_SIZE)
            self._canvas.handle_events(events, canvas_rect)

        elif self.state == BABY_GUESSING:
            self.thinking_timer -= dt
            if self.thinking_timer <= 0:
                surf = self._get_selected_surface()
                tensor = surface_to_tensor(surf)
                try:
                    self.guess_label, self.guess_confidence = self.skill.predict(tensor)
                except Exception:
                    self.guess_label = "???"
                    self.guess_confidence = 0.0
                self.state = PLAYER_CONFIRMING

        elif self.state == LEARNING:
            self.state = SELECTING_OBJECT

        if self._typing_new_class:
            for e in events:
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_RETURN and self._new_class_input:
                        self._confirm_label(self._new_class_input)
                        self._typing_new_class = False
                        self._new_class_input = ""
                    elif e.key == pygame.K_BACKSPACE:
                        self._new_class_input = self._new_class_input[:-1]
                    elif e.key == pygame.K_ESCAPE:
                        self._typing_new_class = False
                        self._new_class_input = ""
                    elif e.unicode and len(self._new_class_input) < 20:
                        self._new_class_input += e.unicode

    def _confirm_label(self, label: str):
        surf = self._get_selected_surface()
        tensor = surface_to_tensor(surf)
        correct = (label == self.guess_label)
        self.accuracy_log.append(correct)
        self.baby.update_mood(correct)
        self.baby.use_energy(0.05)
        self.baby.buffer_replay("whats_this", (tensor, label))
        try:
            self.skill.teach(tensor, label)
        except Exception:
            pass
        self.feedback_text = "Correct!" if correct else f"Learned: {label}"
        self.feedback_timer = 2.0
        if len(self.accuracy_log) % 10 == 0:
            acc = sum(self.accuracy_log[-10:]) / 10
            if acc >= 0.8:
                self.baby.add_milestone(f"What's This 80% acc ({len(self.accuracy_log)} tries)")
        self.state = LEARNING

    def draw(self, surface: pygame.Surface):
        r = self.renderer
        r.draw_text_centered(10, "What's This?", font=r.font_lg)

        if r.draw_button(pygame.Rect(10, 10, 80, 35), "← Back"):
            return "MENU"

        if self.accuracy_log:
            recent = self.accuracy_log[-10:]
            acc = sum(recent) / len(recent) * 100
            r.draw_text(600, 15, f"Accuracy: {acc:.0f}%")

        if self.state == SELECTING_OBJECT:
            r.draw_text_centered(70, "Click an object to show Baby!")
            rects = self._grid_rects()
            for i, (surf, _) in enumerate(self.toy_box):
                rect = rects[i]
                pygame.draw.rect(surface, (200, 200, 200), rect, 1)
                surface.blit(surf, rect.topleft)
                if r._click_pos and rect.collidepoint(r._click_pos):
                    self.selected_index = i
                    self.selected_surface = None
                    self.thinking_timer = 1.0
                    self.state = BABY_GUESSING

            if r.draw_button(pygame.Rect(580, 550, 200, 40), "Draw Your Own!", (220, 180, 100)):
                self._canvas.clear()
                self.state = DRAWING

        elif self.state == DRAWING:
            r.draw_text_centered(55, "Draw something for Baby!", font=r.font_lg)

            canvas_rect = pygame.Rect(230, 80, CANVAS_SIZE, CANVAS_SIZE)
            surface.blit(self._canvas.canvas, canvas_rect.topleft)
            pygame.draw.rect(surface, (0, 0, 0), canvas_rect, 2)

            # Preview
            preview = self._canvas.get_surface_84()
            preview_scaled = pygame.transform.scale(preview, (80, 80))
            surface.blit(preview_scaled, (10, 90))
            pygame.draw.rect(surface, (0, 0, 0), (10, 90, 80, 80), 1)
            r.draw_text(10, 175, "Preview", font=r.font_sm)

            # Toolbar
            self._canvas.draw_toolbar(surface, r, 10, 200)

            # Submit / Back
            btn_y = CANVAS_SIZE + 95
            if r.draw_button(pygame.Rect(420, btn_y, 160, 40), "Show Baby!", (130, 200, 130)):
                self.selected_surface = self._canvas.get_surface_84()
                self.selected_index = None
                self.thinking_timer = 1.0
                self.state = BABY_GUESSING
            if r.draw_button(pygame.Rect(10, btn_y, 80, 40), "← Back"):
                self.state = SELECTING_OBJECT

        elif self.state == BABY_GUESSING:
            surf = self._get_selected_surface()
            scaled = pygame.transform.scale(surf, (168, 168))
            surface.blit(scaled, (316, 150))
            r.draw_baby(400, 420, self.baby.mood, thinking=True)
            r.draw_text_centered(510, "Baby is thinking...")

        elif self.state == PLAYER_CONFIRMING:
            surf = self._get_selected_surface()
            scaled = pygame.transform.scale(surf, (168, 168))
            surface.blit(scaled, (50, 120))

            r.draw_baby(400, 200, self.baby.mood)
            guess_text = f'Baby says: "{self.guess_label}" ({self.guess_confidence:.0%})'
            r.draw_text_centered(290, guess_text)

            if self._typing_new_class:
                r.draw_text_centered(340, "Type new class name:")
                box_rect = pygame.Rect(300, 365, 200, 30)
                pygame.draw.rect(surface, (255, 255, 255), box_rect)
                pygame.draw.rect(surface, (0, 0, 0), box_rect, 2)
                r.draw_text(305, 368, self._new_class_input + "|")
            else:
                r.draw_text_centered(330, "What is it really?")
                try:
                    classes = self.skill.get_known_classes()
                except Exception:
                    classes = []
                all_classes = list(set(classes) | set(PRESET_CLASSES[:5]))
                btn_y = 360
                btn_x = 50
                for cls_name in all_classes:
                    rect = pygame.Rect(btn_x, btn_y, 120, 35)
                    if r.draw_button(rect, cls_name, (130, 200, 130)):
                        self._confirm_label(cls_name)
                    btn_x += 130
                    if btn_x > 650:
                        btn_x = 50
                        btn_y += 45

                btn_y += 50
                if r.draw_button(pygame.Rect(300, btn_y, 200, 40), "+ New Class", (200, 180, 130)):
                    self._typing_new_class = True

        if self.feedback_timer > 0:
            color = (0, 150, 0) if "Correct" in self.feedback_text else (0, 80, 180)
            r.draw_text_centered(560, self.feedback_text, color=color, font=r.font_lg)

        return None
