# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
.venv/bin/pip install -r requirements.txt

# Run the game (1024x768 pygame window)
.venv/bin/python -m app.main

# Run headless balance validation (tests generalist vs routing at scale)
.venv/bin/python app/simulation/headless.py

# Meta-train the general backbone (~10 min on MPS)
.venv/bin/python meta_training/train_general.py --hidden 128 --iterations 1000

# Evaluate per-concept accuracy of a trained checkpoint
.venv/bin/python meta_training/eval_concepts.py
```

Python 3.11 via `.venv/` (learn2learn has build issues on 3.12+). Torch uses MPS (Apple Silicon) when available. No linter or test suite configured — `tests/` exists but is empty.

## Module Layout

- `app/main.py` — entry point, pygame loop
- `app/factory/` — game logic: `objects.py` (procedural shape generation), `worker.py` (MAML wrapper), `routing.py` (DAG graph), `world.py` (tick driver), `economy.py` (coins/rewards)
- `app/ui/` — `factory_floor.py` (main game screen, ~1500 lines), `graph_layout.py` (force-directed layout engine), `drawing_canvas.py` (player drawing tools)
- `app/models/` — `conv4.py` (Conv4Backbone + Conv4WithHead), checkpoints in `checkpoints/`
- `meta_training/` — `procedural_meta.py` (episode generator), `train_general.py` (MAML training loop), `synthetic_shapes.py` (shape primitives)

## Architecture

**The game is a Factorio-style idle factory where workers are real MAML-adapted neural networks.** Objects (procedurally generated shapes) flow through a player-designed routing graph. Each node has a worker that classifies objects and routes them to bins or downstream nodes. Correct classification = coins, wrong = penalty.

### ML Pipeline

`FactoryObject` holds a `(3, 84, 84)` normalized float tensor plus `category` string and `attributes` dict — this is the universal data format flowing through the factory.

`Conv4WithHead` (configurable hidden=64/128/256) wrapped in learn2learn's `MAML`. Meta-trained on procedural episodes via `meta_training/procedural_meta.py` — 80% instance discrimination (tell apart two random shape types), 20% concept-based (abstract visual properties). The backbone learns general few-shot visual features. Checkpoint: `app/models/checkpoints/general_conv4_128.pt`.

Each `FactoryWorker` loads the shared backbone, gets a fresh classification head per task, and caches MAML-adapted weights. Inner steps scale with support set size (`BASE_INNER_STEPS=5`, up to 20). **Real inference runs on every object in the live game** — not simulated.

### Factory Simulation

`FactoryWorld.tick()` generates objects → feeds into `RoutingGraph.process_tick()` (BFS order, real inference) → updates `Economy`. Throughput ramps from 1→8 objects/tick over time. Global speed level (purchasable upgrade) sets `processing_speed` on all nodes.

Routing concepts are **user-defined** — the player teaches binary splits (e.g., "round" vs "angular") by drawing examples. No hardcoded category groupings. `category_mapping` on each worker maps ground-truth object categories to the worker's class names for simulated prediction mode (headless tests only).

### Graph Layout & UI

`ForceLayout` in `graph_layout.py` runs a force-directed physics simulation on topology changes (not per-frame). Five forces maintain a left-to-right DAG layout: depth-constrained x-positioning, node repulsion, edge springs (ideal length 200px for label readability), rectangular collision avoidance, and weak y-centering. Bins are simulated as smaller nodes (`BIN_W=60, BIN_H=20`) at `parent_depth + 1`. Converges in ~10-25 iterations (<1ms).

`FactoryFloorUI` in `factory_floor.py` manages the entire game screen. The graph lives on a virtual canvas larger than the viewport — all positions in `_node_rects` and `_bin_positions` are virtual canvas coordinates. Drawing uses `_vx()`/`_vy()` transforms; hit testing uses `_screen_to_virtual()`. A minimap thumbnail in the top-right corner (click/drag to pan) shows the full graph when it exceeds viewport size. Coordinates flow through `_virtual_to_minimap()`/`_minimap_to_virtual()` — no inline math.

Also includes: drawing canvas with brush/fill/line/stamp tools, training overlay, dialogs, radial context menu, and flow animation (actual shape thumbnails moving along edges with green/red borders for correct/wrong).

### Key Design Decisions

- All workers have speed=1 by default. Throughput scaling comes from the global speed upgrade, not per-node artificial speed formulas.
- Wrong penalty (8) is much higher than drop penalty (1), making **accuracy the dominant economic lever** — this is what makes routing trees valuable over generalists.
- The `ObjectGenerator` difficulty parameter (0.0–1.0) controls shape variation. Currently fixed at 0.0 (clean shapes) for reliable ML accuracy.
- `_rebuild_head()` must be called whenever `class_names` is modified directly (not through `teach()`).
