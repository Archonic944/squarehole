# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the game (1024x768 pygame window)
.venv/bin/python -m app.main

# Run headless balance validation (tests generalist vs routing at scale)
.venv/bin/python app/simulation/headless.py

# Meta-train the general backbone (~10 min on MPS)
.venv/bin/python meta_training/train_general.py --hidden 128 --iterations 1000

# Evaluate per-concept accuracy of a trained checkpoint
.venv/bin/python meta_training/eval_concepts.py
```

Python 3.11 via `.venv/` (learn2learn has build issues on 3.12+). Torch uses MPS (Apple Silicon) when available.

## Architecture

**The game is a Factorio-style idle factory where workers are real MAML-adapted neural networks.** Objects (procedurally generated shapes) flow through a player-designed routing graph. Each node has a worker that classifies objects and routes them to bins or downstream nodes. Correct classification = coins, wrong = penalty.

### ML Pipeline

`Conv4WithHead` (configurable hidden=64/128/256) wrapped in learn2learn's `MAML`. Meta-trained on procedural episodes via `meta_training/procedural_meta.py` — 80% instance discrimination (tell apart two random shape types), 20% concept-based (abstract visual properties). The backbone learns general few-shot visual features. Checkpoint: `app/models/checkpoints/general_conv4_128.pt`.

Each `FactoryWorker` loads the shared backbone, gets a fresh classification head per task, and caches MAML-adapted weights. Inner steps scale with support set size (`BASE_INNER_STEPS=5`, up to 20). **Real inference runs on every object in the live game** — not simulated.

### Factory Simulation

`FactoryWorld.tick()` generates objects → feeds into `RoutingGraph.process_tick()` (BFS order, real inference) → updates `Economy`. Throughput ramps from 1→8 objects/tick over time. Global speed level (purchasable upgrade) sets `processing_speed` on all nodes.

Routing concepts are **user-defined** — the player teaches binary splits (e.g., "round" vs "angular") by drawing examples. No hardcoded category groupings. `category_mapping` on each worker maps ground-truth object categories to the worker's class names for simulated prediction mode (headless tests only).

### UI

`FactoryFloorUI` in `factory_floor.py` manages the entire game screen: graph rendering with auto-layout (subtree-aware vertical distribution), drawing canvas with brush/fill/line/stamp tools, training overlay, dialogs, and flow animation (actual shape thumbnails moving along edges with green/red borders for correct/wrong).

### Key Design Decisions

- All workers have speed=1 by default. Throughput scaling comes from the global speed upgrade, not per-node artificial speed formulas.
- Wrong penalty (8) is much higher than drop penalty (1), making **accuracy the dominant economic lever** — this is what makes routing trees valuable over generalists.
- The `ObjectGenerator` difficulty parameter (0.0–1.0) controls shape variation. Currently fixed at 0.0 (clean shapes) for reliable ML accuracy.
- `_rebuild_head()` must be called whenever `class_names` is modified directly (not through `teach()`).
