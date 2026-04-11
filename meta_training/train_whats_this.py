"""Meta-train Conv4 with MAML on Omniglot for the 'What's This?' skill."""

import os
import sys
import random

import torch
import torch.nn.functional as F
import learn2learn as l2l
from torchvision import transforms
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.models.conv4 import Conv4WithHead

WAYS = 5
SHOTS = 5
QUERY_PER_CLASS = 3
INNER_LR = 0.01
OUTER_LR = 0.001
INNER_STEPS = 5
ITERATIONS = 300
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "models", "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "whats_this_maml.pt")


def build_class_index(dataset):
    """Build a dict mapping class label -> list of indices."""
    print("Building class index...", flush=True)
    index = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset[i]
        index[label].append(i)
    print(f"Indexed {len(index)} classes, {len(dataset)} samples", flush=True)
    return index


def sample_episode(dataset, class_index, ways, shots, query_per_class, transform_to_3ch=True):
    """Manually sample an N-way K-shot episode."""
    classes = random.sample(list(class_index.keys()), ways)
    support_imgs, support_labels = [], []
    query_imgs, query_labels = [], []

    for new_label, cls in enumerate(classes):
        indices = random.sample(class_index[cls], shots + query_per_class)
        for i, idx in enumerate(indices):
            img, _ = dataset[idx]
            if i < shots:
                support_imgs.append(img)
                support_labels.append(new_label)
            else:
                query_imgs.append(img)
                query_labels.append(new_label)

    return (
        torch.stack(support_imgs),
        torch.tensor(support_labels, dtype=torch.long),
        torch.stack(query_imgs),
        torch.tensor(query_labels, dtype=torch.long),
    )


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        lambda x: x.expand(3, -1, -1),  # 1-ch -> 3-ch
    ])

    print("Loading Omniglot...", flush=True)
    omniglot = l2l.vision.datasets.FullOmniglot(
        root="~/data", transform=transform, download=True,
    )
    class_index = build_class_index(omniglot)

    model = Conv4WithHead(num_classes=WAYS).to(device)
    maml = l2l.algorithms.MAML(model, lr=INNER_LR, first_order=True)
    optimizer = torch.optim.Adam(maml.parameters(), lr=OUTER_LR)

    print("Starting training...", flush=True)
    for iteration in range(1, ITERATIONS + 1):
        optimizer.zero_grad()
        meta_loss = 0.0
        meta_acc = 0.0
        n_tasks = 4

        for _ in range(n_tasks):
            learner = maml.clone()
            s_imgs, s_labels, q_imgs, q_labels = sample_episode(
                omniglot, class_index, WAYS, SHOTS, QUERY_PER_CLASS,
            )
            s_imgs, s_labels = s_imgs.to(device), s_labels.to(device)
            q_imgs, q_labels = q_imgs.to(device), q_labels.to(device)

            # Inner loop
            for _ in range(INNER_STEPS):
                preds = learner(s_imgs)
                learner.adapt(F.cross_entropy(preds, s_labels))

            # Query eval
            preds = learner(q_imgs)
            task_loss = F.cross_entropy(preds, q_labels)
            task_acc = (preds.argmax(1) == q_labels).float().mean().item()
            task_loss.backward()
            meta_loss += task_loss.item()
            meta_acc += task_acc

        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.div_(n_tasks)
        optimizer.step()

        if iteration % 50 == 0:
            print(
                f"Iter {iteration}/{ITERATIONS} | "
                f"Loss: {meta_loss / n_tasks:.4f} | "
                f"Acc: {meta_acc / n_tasks:.4f}",
                flush=True,
            )

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(maml.state_dict(), CKPT_PATH)
    print(f"Checkpoint saved to {CKPT_PATH}")


if __name__ == "__main__":
    main()
