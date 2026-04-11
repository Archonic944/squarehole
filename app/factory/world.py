"""Factory world — top-level game loop tying routing, economy, and workers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .economy import Economy
from .routing import RoutingGraph, TickResults
from .worker import FactoryWorker

if TYPE_CHECKING:
    pass  # FactoryObject imported at runtime via the generator

_CHECKPOINT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints")
)
SORT_CHECKPOINT = os.path.join(_CHECKPOINT_DIR, "sort_it_out_maml.pt")
WHATS_CHECKPOINT = os.path.join(_CHECKPOINT_DIR, "whats_this_maml.pt")


class FactoryWorld:
    """Main game state container and tick driver.

    Parameters
    ----------
    object_generator :
        An ``ObjectGenerator`` instance (from ``app.factory.objects``) that
        provides a ``.generate_batch(n, categories)`` method.
    """

    # Difficulty scaling knobs
    INITIAL_CATEGORIES: int = 4
    CATEGORY_UNLOCK_INTERVAL: int = 200  # ticks between new category unlocks

    def __init__(self, object_generator):
        self.object_generator = object_generator
        self.graph = RoutingGraph()
        self.economy = Economy()
        self.workers: list[FactoryWorker] = []
        self.tick_count: int = 0
        self.objects_per_tick: int = 3

        # Start with the first INITIAL_CATEGORIES from the generator's pool
        all_cats = getattr(object_generator, "ALL_CATEGORIES", [])
        self.active_categories: list[str] = list(
            all_cats[: self.INITIAL_CATEGORIES]
        )
        self._remaining_categories: list[str] = list(
            all_cats[self.INITIAL_CATEGORIES :]
        )

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> TickResults:
        """Run one game tick.

        1. Generate ``objects_per_tick`` new objects from active categories.
        2. Feed them into the routing graph.
        3. Process the graph (simulated inference by default).
        4. Update the economy.
        5. Maybe unlock a new category.
        6. Return the tick results.
        """
        self.tick_count += 1

        # 1. Generate objects
        new_objects = self.object_generator.generate_batch(
            self.objects_per_tick, categories=self.active_categories
        )

        # 2-3. Process routing graph (real inference for interactive play)
        results = self.graph.process_tick(new_objects, use_real_inference=True)

        # 4. Economy
        num_active_workers = sum(
            1
            for node in self.graph.nodes.values()
            if node.worker is not None
        )
        self.economy.process_tick_results(results, num_active_workers)

        # 5. Difficulty scaling
        self._maybe_unlock_category()

        return results

    def _maybe_unlock_category(self):
        """Unlock a new object category every CATEGORY_UNLOCK_INTERVAL ticks."""
        if (
            self._remaining_categories
            and self.tick_count % self.CATEGORY_UNLOCK_INTERVAL == 0
        ):
            new_cat = self._remaining_categories.pop(0)
            self.active_categories.append(new_cat)

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def hire_worker(
        self, name: str, binary: bool = True, device: str = "cpu"
    ) -> FactoryWorker | None:
        """Hire (create) a new worker if the factory can afford it.

        Parameters
        ----------
        name : str
            Display name for the worker.
        binary : bool
            If True, creates a 2-class worker (for routing).
            If False, creates an N-way worker (for terminal classification).
        device : str
            Torch device string.  Defaults to ``"cpu"`` for headless safety.

        Returns
        -------
        FactoryWorker or None
            The new worker, or ``None`` if insufficient coins.
        """
        if not self.economy.can_afford(self.economy.HIRE_COST):
            return None

        self.economy.spend(self.economy.HIRE_COST)

        # whats_this checkpoint transfers better to factory shapes
        checkpoint = WHATS_CHECKPOINT
        if binary:
            num_classes = 2
        else:
            num_classes = len(self.active_categories)

        worker = FactoryWorker(
            name=name,
            checkpoint_path=checkpoint,
            num_classes=num_classes,
            device=device,
        )
        self.workers.append(worker)
        return worker

    def fire_worker(self, worker: FactoryWorker):
        """Remove a worker from the factory.

        If the worker is assigned to a node, it is unassigned first.
        """
        self.unassign_worker(worker)
        if worker in self.workers:
            self.workers.remove(worker)

    def assign_worker(self, worker: FactoryWorker, node_id: str):
        """Place *worker* at the given routing node.

        If the worker was assigned elsewhere, it is removed from the old node
        first.
        """
        # Remove from any existing assignment
        self.unassign_worker(worker)

        if node_id in self.graph.nodes:
            self.graph.nodes[node_id].worker = worker

    def unassign_worker(self, worker: FactoryWorker):
        """Remove *worker* from whichever node it occupies (if any)."""
        for node in self.graph.nodes.values():
            if node.worker is worker:
                node.worker = None
                break

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a summary of the current factory state."""
        return {
            "tick_count": self.tick_count,
            "coins": self.economy.coins,
            "coins_per_tick": self.economy.coins_per_tick,
            "total_earned": self.economy.total_earned,
            "total_penalties": self.economy.total_penalties,
            "total_spent": self.economy.total_spent,
            "num_workers": len(self.workers),
            "active_categories": list(self.active_categories),
            "objects_per_tick": self.objects_per_tick,
            "nodes": len(self.graph.nodes),
            "workers": [
                {
                    "name": w.name,
                    "role": w.role,
                    "accuracy": w.cached_accuracy,
                    "processed": w.stats.total_processed,
                    "correct": w.stats.total_correct,
                }
                for w in self.workers
            ],
        }

    # ------------------------------------------------------------------
    # Batch simulation
    # ------------------------------------------------------------------

    def run_ticks(self, n: int) -> list[TickResults]:
        """Run *n* ticks and return all results. Useful for headless simulation."""
        return [self.tick() for _ in range(n)]
