"""Meta-train a general-purpose few-shot backbone on diverse procedural concepts.

This replaces the old shape-specific training. The backbone learns to
extract visual features for ANY binary concept — curved vs straight,
big vs small, pointy vs smooth, etc.

Usage:
    python meta_training/train_general.py [--hidden 128] [--iterations 2000]
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import learn2learn as l2l

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.models.conv4 import Conv4WithHead
from meta_training.procedural_meta import ProceduralMetaDataset

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "models", "checkpoints")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=128,
                        help="Conv4 hidden channels (64, 128, or 256)")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Meta-training iterations")
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--outer-lr", type=float, default=0.001)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--tasks-per-batch", type=int, default=4)
    parser.add_argument("--support", type=int, default=5,
                        help="Support examples per class")
    parser.add_argument("--query", type=int, default=5,
                        help="Query examples per class")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Output filename (default: general_conv4_{hidden}.pt)")
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save a snapshot checkpoint every N iterations (0 to disable)")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Backbone: Conv4-{args.hidden} "
          f"({args.hidden * 5 * 5}-dim features)")

    # Model
    model = Conv4WithHead(num_classes=2, hidden=args.hidden).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    maml = l2l.algorithms.MAML(model, lr=args.inner_lr, first_order=True)
    optimizer = torch.optim.Adam(maml.parameters(), lr=args.outer_lr)

    dataset = ProceduralMetaDataset(n_support=args.support, n_query=args.query)

    os.makedirs(CKPT_DIR, exist_ok=True)
    save_name = args.save_name or f"general_conv4_{args.hidden}.pt"
    stem, ext = os.path.splitext(save_name)
    if not ext:
        ext = ".pt"

    print(f"\nTraining: {args.iterations} iterations, "
          f"{args.tasks_per_batch} tasks/batch, "
          f"{args.inner_steps} inner steps")
    print(f"Episodes: {args.support}-shot support, {args.query}-shot query")
    print(f"Snapshot every {args.save_every} iters → {stem}_iterNNNN{ext}")
    print(f"Final → {save_name}   (pick true best via eval_features.py)")
    print()

    t0 = time.time()

    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad()
        meta_loss = 0.0
        meta_acc = 0.0

        for _ in range(args.tasks_per_batch):
            learner = maml.clone()
            s_img, s_lab, q_img, q_lab = dataset.sample_episode()
            s_img, s_lab = s_img.to(device), s_lab.to(device)
            q_img, q_lab = q_img.to(device), q_lab.to(device)

            # Inner loop
            for _ in range(args.inner_steps):
                preds = learner(s_img)
                learner.adapt(F.cross_entropy(preds, s_lab))

            # Query eval
            preds = learner(q_img)
            task_loss = F.cross_entropy(preds, q_lab)
            task_acc = (preds.argmax(1) == q_lab).float().mean().item()
            task_loss.backward()
            meta_loss += task_loss.item()
            meta_acc += task_acc

        for p in maml.parameters():
            if p.grad is not None:
                p.grad.data.div_(args.tasks_per_batch)
        optimizer.step()

        avg_acc = meta_acc / args.tasks_per_batch

        # Periodic snapshots — eval_features.py will pick the true best afterwards
        if args.save_every > 0 and iteration % args.save_every == 0:
            snap_path = os.path.join(
                CKPT_DIR, f"{stem}_iter{iteration:04d}{ext}")
            torch.save(maml.state_dict(), snap_path)
            print(f"  [snapshot: {os.path.basename(snap_path)}]", flush=True)

        if iteration % 50 == 0:
            elapsed = time.time() - t0
            avg_loss = meta_loss / args.tasks_per_batch
            print(f"Iter {iteration:>5}/{args.iterations} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.2%} | "
                  f"Time: {elapsed:.0f}s",
                  flush=True)

    # Save the final state (last iteration)
    ckpt_path = os.path.join(CKPT_DIR, save_name)
    torch.save(maml.state_dict(), ckpt_path)
    print(f"\nFinal checkpoint saved to {ckpt_path}")
    print(f"Total time: {time.time() - t0:.0f}s")
    print("\nNext: run eval_features.py on each snapshot to pick the winner.")


if __name__ == "__main__":
    main()
