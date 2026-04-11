"""Baby state management for BabyBrain."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


SAVE_DIR = Path.home() / ".babybrain"
SAVE_JSON = SAVE_DIR / "save.json"
SAVE_PT = SAVE_DIR / "skill_data.pt"


class Baby:
    def __init__(self, name="Baby"):
        self.name = name
        self.creation_date = datetime.now(timezone.utc).isoformat()
        self.mood = 0.7
        self.energy = 1.0
        self.milestones: list[dict] = []
        self.skill_data: dict = {}
        self._mood_momentum = 0.0
        self._replay_buffer: list[tuple] = []  # (skill_name, data) pairs

    @property
    def age_days(self) -> int:
        return self.get_age_days()

    def get_age_days(self) -> int:
        created = datetime.fromisoformat(self.creation_date)
        now = datetime.now(timezone.utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return (now - created).days

    def update_mood(self, success: bool):
        direction = 0.05 if success else -0.08
        self._mood_momentum = 0.6 * self._mood_momentum + 0.4 * direction
        self.mood = max(0.0, min(1.0, self.mood + self._mood_momentum))

    def use_energy(self, amount: float) -> bool:
        if self.energy <= 0:
            return False
        self.energy = max(0.0, self.energy - amount)
        return True

    def rest(self):
        self.energy = 1.0

    def add_milestone(self, name: str):
        self.milestones.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def buffer_replay(self, skill_name: str, data):
        """Buffer training data for experience replay on sleep."""
        self._replay_buffer.append((skill_name, data))

    def sleep_and_replay(self, skills: dict):
        """Run experience replay on buffered data before quitting."""
        for skill_name, data in self._replay_buffer:
            if skill_name in skills:
                skill = skills[skill_name]
                try:
                    if hasattr(skill, "teach"):
                        skill.teach(*data)
                except Exception:
                    pass
        self._replay_buffer.clear()

    def save(self, path: str | None = None):
        save_dir = Path(path) if path else SAVE_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        json_path = save_dir / "save.json" if path else SAVE_JSON
        pt_path = save_dir / "skill_data.pt" if path else SAVE_PT

        state = {
            "name": self.name,
            "creation_date": self.creation_date,
            "mood": self.mood,
            "energy": self.energy,
            "milestones": self.milestones,
            "mood_momentum": self._mood_momentum,
        }
        with open(json_path, "w") as f:
            json.dump(state, f, indent=2)

        if self.skill_data:
            torch.save(self.skill_data, pt_path)

    @classmethod
    def load(cls, path: str | None = None) -> "Baby":
        save_dir = Path(path) if path else SAVE_DIR
        json_path = save_dir / "save.json" if path else SAVE_JSON
        pt_path = save_dir / "skill_data.pt" if path else SAVE_PT

        if not json_path.exists():
            return cls()

        with open(json_path) as f:
            state = json.load(f)

        baby = cls(name=state.get("name", "Baby"))
        baby.creation_date = state["creation_date"]
        baby.mood = state.get("mood", 0.7)
        baby.energy = state.get("energy", 1.0)
        baby.milestones = state.get("milestones", [])
        baby._mood_momentum = state.get("mood_momentum", 0.0)

        if pt_path.exists():
            baby.skill_data = torch.load(pt_path, weights_only=False)

        return baby
