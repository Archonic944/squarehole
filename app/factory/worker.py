"""Factory worker — wraps a MAML-adapted Conv4 model for object classification."""

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import learn2learn as l2l

from ..models.conv4 import Conv4WithHead

INNER_LR = 0.01
BASE_INNER_STEPS = 5  # for ~10 examples; scales up with support set size

_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models", "checkpoints"
)


@dataclass
class WorkerStats:
    total_processed: int = 0
    total_correct: int = 0
    coins_earned: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.total_correct / self.total_processed


class FactoryWorker:
    """A worker node that classifies objects using a MAML-adapted Conv4 model.

    Workers have three lifecycle stages:
      - trainee: freshly hired, no support set yet
      - worker: has been taught examples, can classify
      - teacher: experienced enough to train other workers
    """

    def __init__(
        self,
        name: str,
        checkpoint_path: str,
        num_classes: int,
        device: str = "cpu",
    ):
        self.name = name
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.role = "trainee"
        self.class_names: list[str] = []
        self.cached_accuracy: float = 0.0
        self.stats = WorkerStats()

        # Maps ground-truth object categories to this worker's class names.
        # For terminal workers (bins), the mapping is identity (cat -> cat).
        # For routing workers, it maps categories to user-defined concepts
        # e.g. {"star_5": "pointy", "circle": "not_pointy"}.
        self.category_mapping: dict[str, str] = {}

        # Support set stored per-class
        self._support_set: dict[str, list[torch.Tensor]] = defaultdict(list)

        # Build model and wrap in MAML
        self.hidden = 128  # match the general checkpoint
        self._base_model = Conv4WithHead(num_classes=num_classes, hidden=self.hidden).to(self.device)
        self._maml = l2l.algorithms.MAML(
            self._base_model, lr=INNER_LR, first_order=True
        )

        # Load checkpoint (backbone weights only; head will be rebuilt after teaching)
        resolved_path = os.path.normpath(checkpoint_path)
        if os.path.exists(resolved_path):
            state = torch.load(
                resolved_path, map_location=self.device, weights_only=True
            )
            # Only load backbone weights, skip head (size may mismatch)
            filtered = {
                k: v for k, v in state.items()
                if "head" not in k
            }
            self._maml.load_state_dict(filtered, strict=False)

        # Cached adapted learner — invalidated when support set changes
        self._adapted_learner = None
        self._needs_readapt = True

    # ------------------------------------------------------------------
    # Teaching
    # ------------------------------------------------------------------

    def teach(self, image_tensor: torch.Tensor, class_name: str):
        """Add a training example. Maps *class_name* to an integer index.

        If the class is new the classification head is rebuilt so the output
        dimension matches the current number of classes.  Any cached
        adaptation is invalidated.
        """
        if class_name not in self.class_names:
            self.class_names.append(class_name)
            self._rebuild_head()

        self._support_set[class_name].append(image_tensor.detach().cpu())
        self._needs_readapt = True

        # Promote from trainee once we have examples for every class
        if self.role == "trainee" and all(
            len(self._support_set[c]) > 0 for c in self.class_names
        ):
            self.role = "worker"

    def _rebuild_head(self):
        """Rebuild the classification head to match the current class count."""
        n = len(self.class_names)
        if n < 1:
            return
        self.num_classes = n
        self._base_model.rebuild_head(n)
        self._base_model.to(self.device)
        # Re-wrap so MAML tracks the new head parameters
        self._maml = l2l.algorithms.MAML(
            self._base_model, lr=INNER_LR, first_order=True
        )
        self._needs_readapt = True

    # ------------------------------------------------------------------
    # Adaptation (inner loop)
    # ------------------------------------------------------------------

    def _build_support_batch(self):
        images, labels = [], []
        for idx, name in enumerate(self.class_names):
            for img in self._support_set[name]:
                images.append(img)
                labels.append(idx)
        return (
            torch.stack(images).to(self.device),
            torch.tensor(labels, dtype=torch.long, device=self.device),
        )

    def _adapt(self):
        """Run MAML inner-loop adaptation on the current support set.

        Inner steps scale with support set size: more examples need
        more gradient steps to converge. This keeps both accuracy and
        stability high regardless of how many examples the player provides.

        The result is cached until the support set changes.
        """
        learner = self._maml.clone()
        if self._support_set:
            imgs, labels = self._build_support_batch()
            n_examples = len(imgs)
            # Scale: 5 steps for ~10 examples, ~15 steps for ~60 examples
            steps = max(BASE_INNER_STEPS, BASE_INNER_STEPS + (n_examples - 10) // 4)
            steps = min(steps, 20)  # cap to avoid slowness
            for _ in range(steps):
                preds = learner(imgs)
                loss = F.cross_entropy(preds, labels)
                learner.adapt(loss)
        self._adapted_learner = learner
        self._needs_readapt = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_real(self, image_tensor: torch.Tensor) -> tuple[str, float]:
        """Run actual MAML inference.

        Adapts if needed (result is cached for subsequent calls).
        Returns *(class_name, confidence)*.
        """
        if not self.class_names or not self._support_set:
            return ("unknown", 0.0)

        if self._needs_readapt:
            self._adapt()

        query = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._adapted_learner(query)
            probs = F.softmax(logits, dim=1)
            confidence, idx = probs.max(dim=1)

        i = idx.item()
        if i >= len(self.class_names):
            return self.class_names[0], 0.0
        return self.class_names[i], confidence.item()

    def predict_simulated(
        self, true_category: str, task_categories: list[str]
    ) -> tuple[str, bool]:
        """Fast probabilistic prediction for simulation ticks.

        Uses *cached_accuracy* to decide correctness stochastically.
        For routing workers, uses ``category_mapping`` to determine
        the correct output label for each object category.
        Returns *(predicted_label, is_correct)*.
        """
        # Determine what the correct output should be
        if self.category_mapping:
            correct_output = self.category_mapping.get(true_category)
            if correct_output is None:
                # Unknown category — random guess
                return random.choice(task_categories), False
        else:
            if true_category not in task_categories:
                # Misrouted object — worker doesn't know this category
                return random.choice(task_categories), False
            correct_output = true_category

        is_correct = random.random() < self.cached_accuracy

        if is_correct:
            return correct_output, True

        # Pick a random wrong output
        wrong_choices = [c for c in task_categories if c != correct_output]
        if not wrong_choices:
            return correct_output, True
        return random.choice(wrong_choices), False

    # ------------------------------------------------------------------
    # Accuracy estimation
    # ------------------------------------------------------------------

    def estimate_accuracy(self, test_objects: list) -> float:
        """Run real inference on *test_objects*, measure accuracy, and cache it.

        Each element in *test_objects* should have a *.tensor* and *.category*
        attribute (i.e. ``FactoryObject``).

        For routing workers with a ``category_mapping``, the expected label
        is looked up from the mapping.  For terminal workers the expected
        label is the object's own category.

        Use a large test set (50+ per class) for stable estimates.
        """
        if not test_objects or not self.class_names:
            self.cached_accuracy = 0.0
            return 0.0

        correct = 0
        total = 0
        for obj in test_objects:
            if self.category_mapping:
                expected = self.category_mapping.get(obj.category)
                if expected is None:
                    continue
            else:
                if obj.category not in self.class_names:
                    continue
                expected = obj.category

            pred, _ = self.predict_real(obj.tensor)
            total += 1
            if pred == expected:
                correct += 1

        self.cached_accuracy = correct / total if total > 0 else 0.0
        return self.cached_accuracy

    # ------------------------------------------------------------------
    # Knowledge distillation
    # ------------------------------------------------------------------

    def teach_from_teacher(self, teacher: "FactoryWorker", objects: list):
        """Learn from another worker's predictions (knowledge distillation).

        The *teacher* predicts on *objects* and the student adds those
        predictions as training examples.  Accuracy degrades naturally
        because the teacher's labels are not perfect.
        """
        for obj in objects:
            pred, _ = teacher.predict_real(obj.tensor)
            self.teach(obj.tensor, pred)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def natural_speed(self) -> int:
        """All workers process at the same speed. No artificial advantages."""
        return 1

    def get_support_set_size(self) -> int:
        return sum(len(v) for v in self._support_set.values())

    def __repr__(self) -> str:
        return (
            f"FactoryWorker(name={self.name!r}, role={self.role!r}, "
            f"classes={self.class_names}, "
            f"support={self.get_support_set_size()}, "
            f"accuracy={self.cached_accuracy:.2f})"
        )
