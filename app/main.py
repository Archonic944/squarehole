"""BabyBrain Factory — entry point."""

import argparse
import sys

import pygame

from app.factory.objects import ObjectGenerator
from app.factory.world import FactoryWorld
from app.ui.factory_floor import FactoryFloorUI

WIDTH, HEIGHT = 1024, 768
FPS = 30


def _detect_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="BabyBrain Factory")
    parser.add_argument(
        "--preset",
        choices=["simple_split", "two_level_tree", "stress_test"],
        help="Load a pre-built factory preset instead of starting blank",
    )
    parser.add_argument(
        "--load",
        metavar="PATH",
        help="Load a saved game from the given file path",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Save game state to this path when the game exits",
    )
    args = parser.parse_args()

    device = _detect_device()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BabyBrain Factory")
    clock = pygame.time.Clock()

    if args.load:
        from app.factory.save_load import load_game

        world = load_game(args.load, device=device)
        gen = world.object_generator
    elif args.preset:
        from app.factory.presets import simple_split, two_level_tree, stress_test

        presets = {
            "simple_split": simple_split,
            "two_level_tree": two_level_tree,
            "stress_test": stress_test,
        }
        world = presets[args.preset](device=device)
        gen = world.object_generator
    else:
        gen = ObjectGenerator(difficulty=0.0)
        world = FactoryWorld(gen)
        world.objects_per_tick = 1

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

    if args.save:
        from app.factory.save_load import save_game

        save_game(world, args.save)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
