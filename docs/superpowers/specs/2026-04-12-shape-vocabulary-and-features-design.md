# Shape Vocabulary and Visual Features

## Problem

The current factory game has 8 starter shapes and a single meta-trained Conv4 backbone that performs poorly at runtime (see headless test results: binary routing workers at ~54% accuracy, terminal 3-class workers below chance). The `train_robust_contracts.py` approach — meta-training on abstract parameterized patterns (superellipses, spoke webs, holes) — produced a backbone that learned features for abstract mathematical patterns rather than features that transfer to realistic shape vocabulary.

Simultaneously, we want the game to support **more shapes** so that players can discover **more routing puzzle solutions**, and each new shape pack should **disrupt** existing routing trees to force players to discover new abstract features.

The core tension: the backbone must be **general enough** that it has never seen the specific game shapes (otherwise there's no learning for the player to do), but **grounded enough** that the features it learns actually transfer to realistic shapes.

## Solution

Define a fixed vocabulary of **19 shapes across 4 packs** (starter + 3 contract packs) and a fixed vocabulary of **8 binary visual features** that the backbone will meta-train to detect. Shapes are chosen so that:

1. Every shape has a **unique feature signature** (no routing collisions).
2. Every shape can be reached through **multiple routing paths** (2-3+ distinct trees).
3. Every feature is **concretely detectable** by a Conv4 backbone at 84x84.
4. Each new pack introduces shapes that **disrupt** existing routing trees by exercising features players haven't needed yet.

The backbone is meta-trained on episodes that exercise the 8 features using random, procedural shapes — **never the actual game shapes**. This preserves the game's "teach a worker from scratch" premise while ensuring the learned features transfer.

## The 8 Visual Features

Each feature is a binary classification task at meta-training time. The backbone learns to detect the presence/absence of each feature from few examples.

| # | Feature | What it measures | Detection signal |
|---|---------|------------------|------------------|
| 1 | **Curved edges / Straight edges** | Edge curvature | Smooth outline pixels vs straight line segments |
| 2 | **Sharp tips / Rounded tips** | Vertex acuteness | Acute corners vs obtuse/rounded corners |
| 3 | **Elongated / Compact** | Aspect ratio deviation | Bounding box width/height ratio |
| 4 | **Has-hole / Solid** | Interior void | Background color visible inside shape |
| 5 | **Branching / Unitary** | Radiating protrusions | Skeleton with multiple arms from center |
| 6 | **Lopsided / Balanced** | Asymmetric mass distribution | Center of mass offset from geometric center |
| 7 | **Bumpy / Smooth** | Outline regularity | Frequency of outline direction changes |
| 8 | **Multicolor / Mono** | Distinct color regions | 2+ dominant hue regions |

**Why these 8 (and no others)**:
- "Open/closed contour" was considered and rejected — topological, requires path tracing, not a CNN operation.
- "Internal detail" (e.g., concentric rings vs solid fill) was rejected — too subtle at 84x84 with pooled feature maps.
- "Multi-part / Unitary" was considered to distinguish mushroom from apple but rejected in favor of swapping apple for candy cane.
- "Concave / Convex" correlates too heavily with branching and sharp tips to add independent signal.

## The 19 Shapes

### Starter Pack (5 shapes)

Basic shapes that establish the core routing features (curved/straight, sharp/blunt, branching). Routable with just 2-3 features.

- **Circle** — `{curved}`
- **Triangle** — `{sharp}`
- **Square** — `{}` (no positive features — the "default" shape)
- **Star** — `{sharp, branching}`
- **Heart** — `{curved, sharp}`

### Pack 1: Tricky Silhouettes (5 shapes)

Introduces shapes that use *familiar* starter features in *unfamiliar* combinations. A player whose starter routing uses only "curved vs angular" will find their "curved" bucket suddenly overcrowded. This pack forces discovery of **lopsided**, **bumpy**, and **elongated** as routing features.

- **Arrow** — `{sharp, elongated}`
- **Crescent** — `{curved, lopsided}` (curved but asymmetric — breaks "curved = symmetric")
- **Cloud** — `{curved, bumpy}` (curved but wavy outline — breaks "curved = smooth")
- **Lightning** — `{sharp, elongated, lopsided}`
- **Teardrop** — `{curved, sharp, elongated}`

### Pack 2: Holes & Structure (4 shapes)

Introduces interior structure as a new routing axis. Previous features don't help here — the player must discover **has-hole** as a routing feature. Four shapes instead of five because no fifth "shape with a hole" renders cleanly at 84x84 without overlapping the existing four.

- **Donut** — `{curved, hole}`
- **Picture Frame** — `{hole}` (hollow square: angular, has square hole)
- **Key** — `{elongated, hole, lopsided}`
- **Gear** — `{curved, hole, bumpy}` (teeth = bumpy outline)

### Pack 3: Multicolor (5 shapes)

Forces yet another new routing axis: **multicolor**. Every shape here has 2+ distinct colored regions. Players who routed only by silhouette must now handle the color channel.

- **Mushroom** — `{curved, multicolor}` (dome cap + stem, different colors)
- **Tree** — `{sharp, elongated, multicolor}` (triangular canopy + trunk, Christmas-tree style)
- **Flower** — `{curved, branching, multicolor}` (petals radiating from center)
- **Candy Cane** — `{curved, elongated, multicolor}` (curved J-shape with red/white stripes)
- **Rainbow** — `{curved, bumpy, multicolor}` (concentric arcs in different colors)

## Full Feature Matrix

| Shape | Curved | Sharp | Elong | Hole | Branch | Lopsid | Bumpy | Multi |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Starter** | | | | | | | | |
| Circle | ✓ | | | | | | | |
| Triangle | | ✓ | | | | | | |
| Square | | | | | | | | |
| Star | | ✓ | | | ✓ | | | |
| Heart | ✓ | ✓ | | | | | | |
| **Pack 1: Tricky Silhouettes** | | | | | | | | |
| Arrow | | ✓ | ✓ | | | | | |
| Crescent | ✓ | | | | | ✓ | | |
| Cloud | ✓ | | | | | | ✓ | |
| Lightning | | ✓ | ✓ | | | ✓ | | |
| Teardrop | ✓ | ✓ | ✓ | | | | | |
| **Pack 2: Holes** | | | | | | | | |
| Donut | ✓ | | | ✓ | | | | |
| Picture Frame | | | | ✓ | | | | |
| Key | | | ✓ | ✓ | | ✓ | | |
| Gear | ✓ | | | ✓ | | | ✓ | |
| **Pack 3: Multicolor** | | | | | | | | |
| Mushroom | ✓ | | | | | | | ✓ |
| Tree | | ✓ | ✓ | | | | | ✓ |
| Flower | ✓ | | | | ✓ | | | ✓ |
| Candy Cane | ✓ | | ✓ | | | | | ✓ |
| Rainbow | ✓ | | | | | | ✓ | ✓ |

**Uniqueness verified**: Every shape has a unique feature signature. No two shapes collide.

## Routing Tree Analysis

Multiple valid routing trees exist for this shape list. Three examples:

### Tree A — "Color first"
Starts by isolating the multicolor pack, then routes each pack by its distinguishing features.

```
multicolor?
├── yes → [Pack 3]
│         curved?
│         ├── yes → branching? → flower
│         │        └── no → bumpy? → rainbow
│         │                 └── no → elongated? → candy cane / mushroom
│         └── no → tree
└── no → has-hole?
         ├── yes → [Pack 2 subtree: split by curved, elongated, bumpy]
         └── no → [Starter + Pack 1 subtree]
```

### Tree B — "Hole first"
Starts by isolating shapes with holes.

```
has-hole?
├── yes → [Pack 2]
│         curved?
│         ├── yes → bumpy? → gear / donut
│         └── no → elongated? → key / picture frame
└── no → multicolor? → [rest of tree]
```

### Tree C — "Shape first"
Starts with the most basic split. Imbalanced (11 curved vs 8 straight-edge shapes) but valid.

```
curved edges?
├── yes → [11 shapes] further split by multicolor, hole, sharp, branching, etc.
└── no → [8 shapes] further split by sharp, elongated, multicolor, etc.
```

**Per-shape routing path count**: Every shape (except the all-empty `square`) can be reached through at least 2-3 distinct routing paths depending on which feature the player splits on first. Even `square`, with no positive features, can be reached through different negative-feature sequences.

## Meta-Training Approach

The backbone is meta-trained on **episodes that exercise each of the 8 features using random procedural shapes — never the game shapes**. Each episode is a binary classification task:

- Support: 5 examples of class A (feature present), 5 examples of class B (feature absent)
- Query: 5 more examples of each
- Inner loop: MAML adapts the backbone's head to this specific binary task
- Outer loop: meta-gradient optimizes the backbone weights to be quickly adaptable

Example episodes:
- **Has-hole**: support = random shapes with circular/square holes cut out; query = similar
- **Multicolor**: support = random shapes rendered in 2 color regions; query = similar
- **Bumpy**: support = random shapes with wavy outlines; query = similar

Crucially, the shapes used in meta-training episodes (e.g., random polygons, blobs, irregular shapes) are **disjoint** from the 19 game shapes. The backbone never sees a circle, triangle, heart, mushroom, or candy cane during meta-training. It learns to detect the 8 features abstractly, and at runtime MAML adapts it to whatever shape vocabulary the player teaches.

## Success Criteria

1. **Headless test accuracy**: Binary routing workers (2-class) achieve ≥80% accuracy. Terminal workers (4-class) achieve ≥70% accuracy. Deep routing outperforms generalist at scale by ≥2×.
2. **Feature learnability**: Each of the 8 features, trained in isolation, achieves ≥85% meta-test accuracy on held-out episodes.
3. **Feature transfer**: A backbone meta-trained on random shapes achieves ≥75% accuracy when adapted (via few-shot) to detect each feature on the actual 19 game shapes.
4. **Routing tree flexibility**: A player can construct at least 3 distinct routing trees that successfully sort all 19 shapes.

## Non-Goals

- **Perfect classification** of every shape pair at the feature level. Some shapes (triangle vs square) may require terminal-level few-shot learning rather than routing.
- **Fine-grained polygon discrimination** (distinguishing pentagon from hexagon from heptagon). N-gon classification was explicitly dropped as too hard for few-shot.
- **Pre-trained visual recognition**. The backbone must not "already know" the game shapes — if it did, there would be no learning for the player to do.
- **Changes to the routing graph UI, economy, or worker lifecycle**. This spec covers only the shape vocabulary, the visual features, and the meta-training approach. The game's existing systems (FactoryWorker, RoutingGraph, Economy) are unchanged.
