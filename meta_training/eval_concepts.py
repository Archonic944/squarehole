"""Evaluate the trained checkpoint on each concept individually."""

import os, sys, torch, torch.nn.functional as F, learn2learn as l2l
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.models.conv4 import Conv4WithHead
from meta_training.procedural_meta import ALL_CONCEPTS, generate_episode

CKPT = os.path.join(os.path.dirname(__file__), "..", "app", "models", "checkpoints", "general_conv4_128.pt")

model = Conv4WithHead(num_classes=2, hidden=128)
maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)
state = torch.load(CKPT, map_location="cpu", weights_only=True)
filtered = {k: v for k, v in state.items() if "head" not in k}
maml.load_state_dict(filtered, strict=False)

for concept_fn in ALL_CONCEPTS:
    name = concept_fn.__name__.replace("concept_", "")
    accs = []
    for _ in range(20):
        # Manually generate episode from this specific concept
        class_0_fn, class_1_fn = concept_fn()
        from meta_training.procedural_meta import render_image
        import random
        s_imgs, s_labs, q_imgs, q_labs = [], [], [], []
        for imgs, labs, n in [(s_imgs, s_labs, 8), (q_imgs, q_labs, 8)]:
            for _ in range(n):
                label = random.randint(0, 1)
                shape_fn, attrs = class_0_fn() if label == 0 else class_1_fn()
                imgs.append(render_image(shape_fn, attrs))
                labs.append(label)
        s_imgs = torch.stack(s_imgs)
        s_labs = torch.tensor(s_labs)
        q_imgs = torch.stack(q_imgs)
        q_labs = torch.tensor(q_labs)

        learner = maml.clone()
        for _ in range(7):
            preds = learner(s_imgs)
            learner.adapt(F.cross_entropy(preds, s_labs))
        with torch.no_grad():
            preds = learner(q_imgs)
            acc = (preds.argmax(1) == q_labs).float().mean().item()
        accs.append(acc)

    mean = sum(accs) / len(accs) * 100
    bar = "#" * int(mean / 2.5)
    print(f"  {name:<25} {mean:5.1f}%  {bar}")
