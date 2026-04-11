"""'Sort It Out' skill — binary sorting via MAML adaptation with saliency."""

import os

import numpy as np
import torch
import torch.nn.functional as F
import learn2learn as l2l

from ..models.conv4 import Conv4WithHead

INNER_LR = 0.01
INNER_STEPS = 5
DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "models", "checkpoints", "sort_it_out_maml.pt"
)


class SortItOutSkill:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.checkpoint_path = checkpoint_path or DEFAULT_CKPT

        # Support set: two bins
        self.support_images: list[torch.Tensor] = []
        self.support_labels: list[int] = []

        self._base_model = Conv4WithHead(num_classes=2).to(self.device)
        self._maml = l2l.algorithms.MAML(self._base_model, lr=INNER_LR, first_order=True)

        if os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self._maml.load_state_dict(state, strict=False)

    def teach(self, image_tensor: torch.Tensor, bin_label: int):
        """Add an example to the support set (bin 0 or 1)."""
        self.support_images.append(image_tensor.detach())
        self.support_labels.append(bin_label)

    def _get_adapted_learner(self):
        learner = self._maml.clone()
        if not self.support_images:
            return learner

        imgs = torch.stack(self.support_images).to(self.device)
        labels = torch.tensor(self.support_labels, dtype=torch.long, device=self.device)

        for _ in range(INNER_STEPS):
            preds = learner(imgs)
            loss = F.cross_entropy(preds, labels)
            learner.adapt(loss)
        return learner

    def predict(self, image_tensor: torch.Tensor):
        """Adapt and classify.

        Returns: (bin_label: int, confidence: float)
        """
        if not self.support_images:
            return (0, 0.5)

        learner = self._get_adapted_learner()
        query = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = learner(query)
            probs = F.softmax(logits, dim=1)
            confidence, idx = probs.max(dim=1)
        return idx.item(), confidence.item()

    def reset(self):
        """Clear support set for a new round."""
        self.support_images.clear()
        self.support_labels.clear()

    def get_saliency(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Compute gradient-based saliency map (absolute gradient of output w.r.t. input).

        Returns: numpy array of shape (84, 84) with saliency values.
        """
        learner = self._get_adapted_learner()
        query = image_tensor.unsqueeze(0).to(self.device).requires_grad_(True)
        logits = learner(query)
        score = logits.max()
        score.backward()

        # Absolute gradient, max across channels
        saliency = query.grad.data.abs().squeeze(0).max(dim=0)[0]
        return saliency.cpu().numpy()
