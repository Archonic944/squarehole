"""Factory world — top-level game loop tying routing, economy, and workers."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .contracts import ALL_CONTRACTS, STARTER, get_contract, Contract
from .economy import Economy
from .routing import RoutingGraph, TickResults
from .worker import FactoryWorker

if TYPE_CHECKING:
    pass  # FactoryObject imported at runtime via the generator

_CHECKPOINT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints")
)
GENERAL_CHECKPOINT = os.path.join(_CHECKPOINT_DIR, "general_conv4_64_robust.pt")
GENERAL_CHECKPOINT_64 = os.path.join(_CHECKPOINT_DIR, "general_conv4_64.pt")
# Fallback to legacy task-specific checkpoint if no general checkpoint exists
WHATS_CHECKPOINT = os.path.join(_CHECKPOINT_DIR, "whats_this_maml.pt")


def resolve_worker_checkpoint() -> str:
    """Return the best available checkpoint path for runtime workers."""
    for path in (
        GENERAL_CHECKPOINT,
        GENERAL_CHECKPOINT_64,
        WHATS_CHECKPOINT,
    ):
        if os.path.exists(path):
            return path
    return WHATS_CHECKPOINT


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

    # Global speed upgrade
    SPEED_UPGRADE_BASE_COST: float = 200.0
    SPEED_UPGRADE_SCALE: float = 2.0  # each level costs 2x more

    def __init__(self, object_generator):
        self.object_generator = object_generator
        self.graph = RoutingGraph()
        self.economy = Economy()
        self.workers: list[FactoryWorker] = []
        self.tick_count: int = 0
        self.objects_per_tick: int = 3
        self.speed_level: int = 1  # global processing speed for all nodes

        # Progression is contract-based. The starter pack is accepted
        # automatically; additional contracts are accepted explicitly
        # from the Contracts UI.
        self.accepted_contract_ids: set[str] = set()
        self.active_categories: list[str] = []
        # Kept for legacy preset / save-load compatibility. The auto-
        # unlock path still works for presets that populate this list.
        self._remaining_categories: list[str] = []
        self.accept_contract(STARTER.id)

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

        # 2. Apply global speed level to all nodes
        for node in self.graph.nodes.values():
            node.processing_speed = self.speed_level

        # 3. Process routing graph (real inference for interactive play)
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
        self._maybe_ramp_throughput()

        return results

    # Throughput ramp: objects_per_tick increases over time
    THROUGHPUT_RAMP_INTERVAL: int = 100  # every N ticks
    MAX_OBJECTS_PER_TICK: int = 8

    def _maybe_unlock_category(self):
        """Unlock a new object category every CATEGORY_UNLOCK_INTERVAL ticks."""
        if (
            self._remaining_categories
            and self.tick_count % self.CATEGORY_UNLOCK_INTERVAL == 0
        ):
            new_cat = self._remaining_categories.pop(0)
            self.active_categories.append(new_cat)

    def _maybe_ramp_throughput(self):
        """Increase objects_per_tick over time so routing becomes necessary."""
        if (
            self.tick_count > 0
            and self.tick_count % self.THROUGHPUT_RAMP_INTERVAL == 0
            and self.objects_per_tick < self.MAX_OBJECTS_PER_TICK
        ):
            self.objects_per_tick += 1

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

        checkpoint = resolve_worker_checkpoint()
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

    # ------------------------------------------------------------------
    # Contracts
    # ------------------------------------------------------------------

    def accept_contract(self, contract_id: str) -> bool:
        """Accept a contract, adding its categories to the active pool.

        Returns True on success. Returns False if the contract is unknown,
        already accepted, or the factory cannot afford its cost.
        """
        if contract_id in self.accepted_contract_ids:
            return False
        contract = get_contract(contract_id)
        if contract is None:
            return False
        if contract.cost > 0 and not self.economy.can_afford(contract.cost):
            return False
        if contract.cost > 0:
            self.economy.spend(contract.cost)
        self.accepted_contract_ids.add(contract_id)
        for cat in contract.categories:
            if cat not in self.active_categories:
                self.active_categories.append(cat)
        return True

    def available_contracts(self) -> list[Contract]:
        """Contracts not yet accepted, in display order."""
        return [c for c in ALL_CONTRACTS if c.id not in self.accepted_contract_ids]

    def get_speed_upgrade_cost(self) -> float:
        return self.SPEED_UPGRADE_BASE_COST * (self.SPEED_UPGRADE_SCALE ** (self.speed_level - 1))

    def buy_speed_upgrade(self) -> bool:
        cost = self.get_speed_upgrade_cost()
        if self.economy.spend(cost):
            self.speed_level += 1
            return True
        return False

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
            "speed_level": self.speed_level,
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
