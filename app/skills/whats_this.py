"""'What's This?' skill — few-shot object recognition via MAML adaptation."""

import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import learn2learn as l2l

from ..models.conv4 import Conv4WithHead

INNER_LR = 0.01
INNER_STEPS = 5
DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "models", "checkpoints", "whats_this_maml.pt"
)


class WhatsThisSkill:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_path = checkpoint_path or DEFAULT_CKPT
        self.support_set: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.class_names: list[str] = []

        # Build model with a dummy head; will be rebuilt on first teach
        self._base_model = Conv4WithHead(num_classes=5).to(self.device)
        self._maml = l2l.algorithms.MAML(self._base_model, lr=INNER_LR, first_order=True)

        if os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self._maml.load_state_dict(state, strict=False)

    def _rebuild_head(self):
        n = len(self.class_names)
        if n < 1:
            return
        self._base_model.rebuild_head(n)
        self._base_model.to(self.device)
        # Re-wrap so MAML tracks the new parameters
        self._maml = l2l.algorithms.MAML(self._base_model, lr=INNER_LR, first_order=True)

    def teach(self, image_tensor: torch.Tensor, label: str):
        """Add an example to the support set. Rebuilds head if new class."""
        if label not in self.class_names:
            self.class_names.append(label)
            self._rebuild_head()
        self.support_set[label].append(image_tensor.detach())

    def _build_support_batch(self):
        images, labels = [], []
        for idx, name in enumerate(self.class_names):
            for img in self.support_set[name]:
                images.append(img)
                labels.append(idx)
        return (
            torch.stack(images).to(self.device),
            torch.tensor(labels, dtype=torch.long, device=self.device),
        )

    def predict(self, image_tensor: torch.Tensor):
        """Adapt on the full support set, then classify the query image.

        Returns: (label: str, confidence: float)
        """
        if not self.class_names:
            return ("unknown", 0.0)

        learner = self._maml.clone()
        support_imgs, support_labels = self._build_support_batch()

        # Inner-loop adaptation
        for _ in range(INNER_STEPS):
            preds = learner(support_imgs)
            loss = F.cross_entropy(preds, support_labels)
            learner.adapt(loss)

        # Query
        query = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = learner(query)
            probs = F.softmax(logits, dim=1)
            confidence, idx = probs.max(dim=1)

        return self.class_names[idx.item()], confidence.item()

    def get_known_classes(self) -> list[str]:
        return list(self.class_names)
