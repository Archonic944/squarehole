"""BabyBrain Factory — entry point."""

import sys
import pygame

from app.factory.objects import ObjectGenerator
from app.factory.world import FactoryWorld
from app.ui.factory_floor import FactoryFloorUI

WIDTH, HEIGHT = 1024, 768
FPS = 30


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BabyBrain Factory")
    clock = pygame.time.Clock()

    gen = ObjectGenerator(difficulty=0.0)
    world = FactoryWorld(gen)
    world.objects_per_tick = 5
    ui = FactoryFloorUI(world, gen)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        events = pygame.event.get()

        for e in events:
            if e.type == pygame.QUIT:
                running = False

        ui.update(events, dt)
        result = ui.draw(screen)
        if result == "MENU":
            running = False

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
