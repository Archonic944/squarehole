"""BabyBrain entry point and game loop."""

import sys
import pygame

from app.baby import Baby
from app.ui.renderer import Renderer
from app.ui.whats_this_ui import WhatsThisUI
from app.ui.sort_it_out_ui import SortItOutUI

MENU = "MENU"
WHATS_THIS = "WHATS_THIS"
SORT_IT_OUT = "SORT_IT_OUT"
FACTORY = "FACTORY"

BABY_W, BABY_H = 800, 600
FACTORY_W, FACTORY_H = 1024, 768
FPS = 30


def load_skills():
    skills = {}
    try:
        from app.skills.whats_this import WhatsThisSkill
        skills["whats_this"] = WhatsThisSkill()
    except Exception as e:
        print(f"[BabyBrain] WhatsThisSkill not available: {e}")
        skills["whats_this"] = None

    try:
        from app.skills.sort_it_out import SortItOutSkill
        skills["sort_it_out"] = SortItOutSkill()
    except Exception as e:
        print(f"[BabyBrain] SortItOutSkill not available: {e}")
        skills["sort_it_out"] = None

    return skills


def load_factory():
    try:
        from app.factory.objects import ObjectGenerator
        from app.factory.world import FactoryWorld
        from app.ui.factory_floor import FactoryFloorUI
        gen = ObjectGenerator(difficulty=0.0)
        world = FactoryWorld(gen)
        world.objects_per_tick = 5
        factory_ui = FactoryFloorUI(world, gen)
        return factory_ui
    except Exception as e:
        print(f"[BabyBrain] Factory not available: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    pygame.init()
    screen = pygame.display.set_mode((BABY_W, BABY_H))
    pygame.display.set_caption("BabyBrain")
    clock = pygame.time.Clock()

    baby = Baby.load()
    renderer = Renderer(screen)
    skills = load_skills()

    state = MENU
    whats_this_ui = None
    sort_it_out_ui = None
    factory_ui = None
    no_skill_msg = ""
    no_skill_timer = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        events = pygame.event.get()

        for e in events:
            if e.type == pygame.QUIT:
                running = False

        if state == MENU:
            renderer.begin_frame(events)
            renderer.clear()

            renderer.draw_baby(400, 180, baby.mood)
            renderer.draw_text_centered(50, "BabyBrain", font=renderer.font_lg)
            renderer.draw_age(340, 260, baby.age_days)
            renderer.draw_mood_bar(320, 300, baby.mood)
            renderer.draw_energy_bar(320, 350, baby.energy)
            renderer.draw_text(320, 380, f"Milestones: {len(baby.milestones)}")

            if renderer.draw_button(pygame.Rect(280, 415, 240, 40), "What's This?", (130, 190, 130)):
                if skills.get("whats_this") is None:
                    no_skill_msg = "WhatsThisSkill not found!"
                    no_skill_timer = 3.0
                else:
                    whats_this_ui = WhatsThisUI(renderer, skills["whats_this"], baby)
                    state = WHATS_THIS

            if renderer.draw_button(pygame.Rect(280, 462, 240, 40), "Sort It Out", (130, 130, 190)):
                if skills.get("sort_it_out") is None:
                    no_skill_msg = "SortItOutSkill not found!"
                    no_skill_timer = 3.0
                else:
                    sort_it_out_ui = SortItOutUI(renderer, skills["sort_it_out"], baby)
                    state = SORT_IT_OUT

            if renderer.draw_button(pygame.Rect(280, 509, 240, 40), "Factory Mode", (190, 160, 100)):
                factory_ui = load_factory()
                if factory_ui:
                    screen = pygame.display.set_mode((FACTORY_W, FACTORY_H))
                    state = FACTORY
                else:
                    no_skill_msg = "Factory failed to load!"
                    no_skill_timer = 3.0

            if renderer.draw_button(pygame.Rect(280, 556, 240, 35), "Sleep & Quit", (190, 130, 130)):
                running = False

            if no_skill_timer > 0:
                no_skill_timer -= dt
                renderer.draw_text_centered(420, no_skill_msg, color=(200, 50, 50), font=renderer.font_sm)

            renderer.draw_milestone_notification(dt)
            pygame.display.flip()

        elif state == WHATS_THIS:
            renderer.begin_frame(events)
            renderer.clear()
            whats_this_ui.update(events)
            result = whats_this_ui.draw(screen)
            if result == "MENU":
                state = MENU
            renderer.draw_milestone_notification(dt)
            pygame.display.flip()

        elif state == SORT_IT_OUT:
            renderer.begin_frame(events)
            renderer.clear()
            sort_it_out_ui.update(events)
            result = sort_it_out_ui.draw(screen)
            if result == "MENU":
                state = MENU
            renderer.draw_milestone_notification(dt)
            pygame.display.flip()

        elif state == FACTORY:
            factory_ui.update(events, dt)
            result = factory_ui.draw(screen)
            if result == "MENU":
                # Switch back to baby window size
                screen = pygame.display.set_mode((BABY_W, BABY_H))
                renderer = Renderer(screen)
                state = MENU
            pygame.display.flip()

    # Shutdown
    valid_skills = {k: v for k, v in skills.items() if v is not None}
    baby.sleep_and_replay(valid_skills)
    baby.save()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
