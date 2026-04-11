"""UI for 'Sort It Out' mini-game."""

import pygame
from app.ui.renderer import Renderer
from app.ui.game_objects import generate_sort_objects, surface_to_tensor
from app.ui.drawing_canvas import DrawingCanvas, CANVAS_SIZE

PLAYER_SORTING = 0
BABY_SORTING = 1
REVIEWING = 2
ROUND_RESULTS = 3
DRAWING = 4

BIN_A_RECT = pygame.Rect(80, 460, 220, 110)
BIN_B_RECT = pygame.Rect(500, 460, 220, 110)


class SortItOutUI:
    def __init__(self, renderer: Renderer, skill, baby):
        self.renderer = renderer
        self.skill = skill
        self.baby = baby
        self._canvas = DrawingCanvas()
        self._new_round()

    def _new_round(self):
        self.objects = generate_sort_objects(10)
        self.positions = []
        for i in range(len(self.objects)):
            x = 80 + (i % 5) * 130
            y = 100 + (i // 5) * 110
            self.positions.append([x, y])
        self.sorted_bins: dict[int, str] = {}
        self.dragging = None
        self.drag_offset = (0, 0)
        self.state = PLAYER_SORTING
        self.baby_predictions: list[tuple[int, str, float]] = []
        self.baby_pred_index = 0
        self.saliency_surface = None
        self.saliency_index = None
        self.feedback_text = ""
        self.feedback_timer = 0.0
        # Round stats
        self.round_correct = 0
        self.round_total = 0
        self.total_correct = 0
        self.total_attempts = 0
        try:
            self.skill.reset()
        except Exception:
            pass

    def _unsorted_indices(self):
        return [i for i in range(len(self.objects)) if i not in self.sorted_bins]

    def _bin_to_int(self, label: str) -> int:
        return 0 if label == "A" else 1

    def _int_to_bin(self, val: int) -> str:
        return "A" if val == 0 else "B"

    def _teach_object(self, idx: int, bin_label: str):
        surf, attrs = self.objects[idx]
        tensor = surface_to_tensor(surf)
        bin_int = self._bin_to_int(bin_label)
        try:
            self.skill.teach(tensor, bin_int)
        except Exception:
            pass
        self.baby.use_energy(0.03)
        self.baby.buffer_replay("sort_it_out", (tensor, bin_int))

    def _add_drawn_object(self):
        """Add the current canvas drawing as a new sortable object."""
        surf84 = self._canvas.get_surface_84()
        attrs = {"shape": "drawn", "color": "custom", "size": "custom"}
        idx = len(self.objects)
        self.objects.append((surf84, attrs))
        # Position in next available grid slot
        row = idx // 5
        col = idx % 5
        self.positions.append([80 + col * 130, 100 + row * 110])

    def _start_baby_sorting(self):
        unsorted = self._unsorted_indices()
        self.baby_predictions = []
        for idx in unsorted:
            surf, _ = self.objects[idx]
            tensor = surface_to_tensor(surf)
            try:
                bin_int, conf = self.skill.predict(tensor)
                label = self._int_to_bin(bin_int)
            except Exception:
                label, conf = "A", 0.5
            self.baby_predictions.append((idx, label, conf))
        self.baby_pred_index = 0
        self.round_correct = 0
        self.round_total = 0
        self.state = REVIEWING

    def _try_saliency(self):
        if not self.baby_predictions:
            return
        idx = self.baby_predictions[0][0]
        surf, _ = self.objects[idx]
        tensor = surface_to_tensor(surf)
        try:
            import numpy as np
            sal = self.skill.get_saliency(tensor)
            if sal is not None:
                sal = sal / (sal.max() + 1e-8)
                sal_rgb = (np.stack([sal, sal * 0.2, sal * 0.2], axis=-1) * 255).astype(np.uint8)
                self.saliency_surface = pygame.surfarray.make_surface(sal_rgb.transpose(1, 0, 2))
                self.saliency_surface = pygame.transform.scale(self.saliency_surface, (168, 168))
                self.saliency_index = idx
        except Exception:
            self.saliency_surface = None

    def _reposition_unsorted(self):
        """Re-layout unsorted objects into the grid."""
        unsorted = self._unsorted_indices()
        for slot, idx in enumerate(unsorted):
            row = slot // 5
            col = slot % 5
            self.positions[idx] = [80 + col * 130, 100 + row * 110]

    def _add_more_objects(self, n=5):
        """Generate and add more objects for continued sorting."""
        new_objs = generate_sort_objects(n)
        for surf, attrs in new_objs:
            idx = len(self.objects)
            self.objects.append((surf, attrs))
            row = idx // 5
            col = idx % 5
            self.positions.append([80 + col * 130, 100 + row * 110])
        self._reposition_unsorted()

    def update(self, events: list):
        dt = 1.0 / 30.0
        if self.feedback_timer > 0:
            self.feedback_timer -= dt

        for e in events:
            if self.state == PLAYER_SORTING:
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    for i in self._unsorted_indices():
                        rect = pygame.Rect(self.positions[i][0], self.positions[i][1], 84, 84)
                        if rect.collidepoint(e.pos):
                            self.dragging = i
                            self.drag_offset = (
                                self.positions[i][0] - e.pos[0],
                                self.positions[i][1] - e.pos[1],
                            )
                            break

                elif e.type == pygame.MOUSEMOTION and self.dragging is not None:
                    self.positions[self.dragging][0] = e.pos[0] + self.drag_offset[0]
                    self.positions[self.dragging][1] = e.pos[1] + self.drag_offset[1]

                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1 and self.dragging is not None:
                    idx = self.dragging
                    self.dragging = None
                    center = (self.positions[idx][0] + 42, self.positions[idx][1] + 42)
                    if BIN_A_RECT.collidepoint(center):
                        self.sorted_bins[idx] = "A"
                        self._teach_object(idx, "A")
                    elif BIN_B_RECT.collidepoint(center):
                        self.sorted_bins[idx] = "B"
                        self._teach_object(idx, "B")
                    else:
                        self._reposition_unsorted()

            elif self.state == DRAWING:
                canvas_rect = pygame.Rect(230, 80, CANVAS_SIZE, CANVAS_SIZE)
                self._canvas.handle_events([e], canvas_rect)

    def draw(self, surface: pygame.Surface):
        r = self.renderer
        r.draw_text_centered(10, "Sort It Out", font=r.font_lg)

        if r.draw_button(pygame.Rect(10, 10, 80, 35), "← Back"):
            return "MENU"

        # Score display
        if self.total_attempts > 0:
            r.draw_text(600, 15, f"Baby: {self.total_correct}/{self.total_attempts}", font=r.font_sm)

        if self.state == PLAYER_SORTING:
            r.draw_text_centered(65, "Drag objects into bins to teach Baby a sorting rule")

            # Draw bins
            self._draw_bins(surface, r)

            # Draw unsorted objects
            for i in self._unsorted_indices():
                surf, _ = self.objects[i]
                surface.blit(surf, (self.positions[i][0], self.positions[i][1]))
                pygame.draw.rect(surface, (0, 0, 0),
                                 (self.positions[i][0], self.positions[i][1], 84, 84), 1)

            # Action buttons
            unsorted = self._unsorted_indices()
            btn_y = 430
            if len(self.sorted_bins) >= 4 and unsorted:
                if r.draw_button(pygame.Rect(300, btn_y, 200, 35), "Let Baby Try!", (100, 200, 150)):
                    self._start_baby_sorting()

            # Draw your own button
            if r.draw_button(pygame.Rect(10, 555, 180, 35), "Draw & Add", (220, 180, 100)):
                self._canvas.clear()
                self.state = DRAWING

            # Add more objects
            if r.draw_button(pygame.Rect(610, 555, 180, 35), "+ More Objects", (160, 180, 220)):
                self._add_more_objects(5)

        elif self.state == DRAWING:
            r.draw_text_centered(55, "Draw an object to sort!", font=r.font_lg)

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

            btn_y = CANVAS_SIZE + 95
            if r.draw_button(pygame.Rect(350, btn_y, 200, 40), "Add to Sort!", (130, 200, 130)):
                self._add_drawn_object()
                self._reposition_unsorted()
                self.state = PLAYER_SORTING
            if r.draw_button(pygame.Rect(10, btn_y, 80, 40), "← Back"):
                self.state = PLAYER_SORTING

        elif self.state == REVIEWING:
            self._draw_bins(surface, r)

            if self.baby_pred_index < len(self.baby_predictions):
                idx, label, conf = self.baby_predictions[self.baby_pred_index]
                remaining = len(self.baby_predictions) - self.baby_pred_index
                r.draw_text_centered(65, f"Baby is sorting... ({remaining} left)")

                surf_obj, _ = self.objects[idx]
                scaled = pygame.transform.scale(surf_obj, (120, 120))
                surface.blit(scaled, (340, 120))

                r.draw_baby(180, 200, self.baby.mood, thinking=False)

                confidence_text = "confident" if conf > 0.7 else "unsure" if conf > 0.55 else "guessing"
                r.draw_text_centered(260, f'Baby says: Bin {label} ({conf:.0%} - {confidence_text})')
                r.draw_text_centered(290, "Is Baby correct?")

                if r.draw_button(pygame.Rect(230, 320, 140, 45), "Correct!", (100, 200, 100)):
                    self.sorted_bins[idx] = label
                    self._teach_object(idx, label)
                    self.baby.update_mood(True)
                    self.round_correct += 1
                    self.round_total += 1
                    self.total_correct += 1
                    self.total_attempts += 1
                    self.baby_pred_index += 1

                other = "B" if label == "A" else "A"
                if r.draw_button(pygame.Rect(420, 320, 140, 45), f"No, Bin {other}", (220, 100, 100)):
                    self.sorted_bins[idx] = other
                    self._teach_object(idx, other)
                    self.baby.update_mood(False)
                    self.round_total += 1
                    self.total_attempts += 1
                    self.baby_pred_index += 1
            else:
                self._try_saliency()
                self.state = ROUND_RESULTS

        elif self.state == ROUND_RESULTS:
            self._draw_bins(surface, r)

            # Score summary
            if self.round_total > 0:
                pct = self.round_correct / self.round_total * 100
                r.draw_text_centered(100, f"Baby got {self.round_correct}/{self.round_total} correct ({pct:.0f}%)", font=r.font_lg)
                if pct >= 80:
                    r.draw_text_centered(140, "Baby is getting the hang of it!", color=(0, 150, 0))
                elif pct >= 50:
                    r.draw_text_centered(140, "Baby is learning...", color=(0, 80, 180))
                else:
                    r.draw_text_centered(140, "Baby needs more examples!", color=(200, 50, 50))
            else:
                r.draw_text_centered(100, "No objects to sort!", font=r.font_lg)

            # Saliency
            if self.saliency_surface and self.saliency_index is not None:
                r.draw_text_centered(175, "Baby was looking at...")
                obj_surf, _ = self.objects[self.saliency_index]
                orig = pygame.transform.scale(obj_surf, (140, 140))
                surface.blit(orig, (220, 195))
                surface.blit(pygame.transform.scale(self.saliency_surface, (140, 140)), (430, 195))
                r.draw_text(240, 340, "Original", font=r.font_sm)
                r.draw_text(460, 340, "Saliency", font=r.font_sm)

            # Options
            btn_y = 370
            if r.draw_button(pygame.Rect(100, btn_y, 220, 45), "Keep Sorting", (100, 200, 150)):
                self._add_more_objects(6)
                self.state = PLAYER_SORTING

            if r.draw_button(pygame.Rect(480, btn_y, 220, 45), "New Round", (100, 180, 220)):
                self._new_round()

            if r.draw_button(pygame.Rect(290, btn_y + 55, 220, 45), "Draw & Add", (220, 180, 100)):
                self._canvas.clear()
                self.state = DRAWING

            # Milestone check
            if self.round_total >= 4 and self.round_correct / self.round_total >= 0.8:
                self.baby.add_milestone(f"Sort It Out {self.round_correct}/{self.round_total}")

        # Feedback
        if self.feedback_timer > 0:
            r.draw_text_centered(560, self.feedback_text, font=r.font_lg)

        return None

    def _draw_bins(self, surface, r):
        """Draw the two sorting bins with counts."""
        pygame.draw.rect(surface, (180, 220, 180), BIN_A_RECT, border_radius=10)
        pygame.draw.rect(surface, (0, 0, 0), BIN_A_RECT, 2, border_radius=10)
        r.draw_text(BIN_A_RECT.x + 70, BIN_A_RECT.y + 40, "Bin A")

        pygame.draw.rect(surface, (180, 180, 220), BIN_B_RECT, border_radius=10)
        pygame.draw.rect(surface, (0, 0, 0), BIN_B_RECT, 2, border_radius=10)
        r.draw_text(BIN_B_RECT.x + 70, BIN_B_RECT.y + 40, "Bin B")

        a_count = sum(1 for v in self.sorted_bins.values() if v == "A")
        b_count = sum(1 for v in self.sorted_bins.values() if v == "B")
        r.draw_text(BIN_A_RECT.x + 80, BIN_A_RECT.y + 5, f"({a_count})")
        r.draw_text(BIN_B_RECT.x + 80, BIN_B_RECT.y + 5, f"({b_count})")
