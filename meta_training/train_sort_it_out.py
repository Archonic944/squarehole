"""Meta-train Conv4 with MAML on synthetic shapes for the 'Sort It Out' skill."""

import os
import sys

import torch
import torch.nn.functional as F
import learn2learn as l2l

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.models.conv4 import Conv4WithHead
from meta_training.synthetic_shapes import SyntheticShapesDataset

WAYS = 2
SHOTS = 5
INNER_LR = 0.01
OUTER_LR = 0.001
INNER_STEPS = 5
ITERATIONS = 500
CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "models", "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "sort_it_out_maml.pt")


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    dataset = SyntheticShapesDataset(n_support=SHOTS, n_query=SHOTS)

    model = Conv4WithHead(num_classes=WAYS).to(device)
    maml = l2l.algorithms.MAML(model, lr=INNER_LR, first_order=True)
    optimizer = torch.optim.Adam(maml.parameters(), lr=OUTER_LR)
    loss_fn = F.cross_entropy

    for iteration in range(1, ITERATIONS + 1):
        optimizer.zero_grad()
        meta_loss = 0.0
        meta_acc = 0.0
        n_tasks = 4

        for _ in range(n_tasks):
            learner = maml.clone()
            support_imgs, support_labels, query_imgs, query_labels = dataset.sample_episode()
            support_imgs = support_imgs.to(device)
            support_labels = support_labels.to(device)
            query_imgs = query_imgs.to(device)
            query_labels = query_labels.to(device)

            # Inner loop on support set
            for _ in range(INNER_STEPS):
                preds = learner(support_imgs)
                train_loss = loss_fn(preds, support_labels)
                learner.adapt(train_loss)

            # Evaluate on query set
            preds = learner(query_imgs)
            task_loss = loss_fn(preds, query_labels)
            task_acc = (preds.argmax(dim=1) == query_labels).float().mean().item()

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
                f"Acc: {meta_acc / n_tasks:.4f}"
            )

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(maml.state_dict(), CKPT_PATH)
    print(f"Checkpoint saved to {CKPT_PATH}")


if __name__ == "__main__":
    main()
