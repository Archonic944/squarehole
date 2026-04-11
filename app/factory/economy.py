"""Economy system — tracks coins, rewards, penalties, and upkeep."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .routing import TickResults


@dataclass
class Economy:
    """Manages the factory's coin balance and financial history."""

    coins: float = 500.0
    total_earned: float = 0.0
    total_spent: float = 0.0
    total_penalties: float = 0.0

    # Configurable rates — tuned so a 60%+ accurate factory profits
    CORRECT_REWARD: float = 15.0
    WRONG_PENALTY: float = 3.0
    DROPPED_PENALTY: float = 1.0
    HIRE_COST: float = 100.0
    UPKEEP_PER_WORKER: float = 0.3  # per tick

    # Rolling history for coins_per_tick
    _recent_earnings: deque = field(default_factory=lambda: deque(maxlen=100))

    def process_tick_results(self, results: "TickResults", num_workers: int) -> float:
        """Apply rewards and penalties from a tick, plus worker upkeep.

        Returns the net coin change for this tick.
        """
        reward = len(results.correct) * self.CORRECT_REWARD
        wrong_pen = len(results.wrong) * self.WRONG_PENALTY
        drop_pen = len(results.dropped) * self.DROPPED_PENALTY
        upkeep = num_workers * self.UPKEEP_PER_WORKER

        net = reward - wrong_pen - drop_pen - upkeep

        # Update totals
        self.total_earned += reward
        self.total_penalties += wrong_pen + drop_pen
        self.total_spent += upkeep
        self.coins += net

        self._recent_earnings.append(net)

        return net

    def can_afford(self, cost: float) -> bool:
        return self.coins >= cost

    def spend(self, amount: float) -> bool:
        """Deduct *amount* from coins. Returns False if insufficient funds."""
        if not self.can_afford(amount):
            return False
        self.coins -= amount
        self.total_spent += amount
        return True

    @property
    def coins_per_tick(self) -> float:
        """Rolling average net coins over the last 100 ticks."""
        if not self._recent_earnings:
            return 0.0
        return sum(self._recent_earnings) / len(self._recent_earnings)
