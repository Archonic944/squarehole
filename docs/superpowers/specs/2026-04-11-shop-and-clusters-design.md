# Shop Upgrade + Worker Clusters — Design

**Status:** Approved 2026-04-11
**Scope:** Replace the speed upgrade with a shapes-per-tick shop upgrade, and introduce worker clusters (vertical columns of independently-trained workers that share routing schema and distribute incoming objects randomly).

## Motivation

The current game lacks a satisfying idle-game reward loop. The speed upgrade is a flat multiplier that scales tick rate and per-node processing speed together, which muddles two separate concepts. It also isn't a real progression lever — the auto-ramp in `_maybe_ramp_throughput` gives players throughput for free over time, undermining any shop.

This change introduces a real reward loop:

1. **Shapes-per-tick upgrade** — purchasable 1→2→3→4. The player chooses when to raise pressure.
2. **Worker clusters** — when one worker can't keep up, the player spends coins to train additional workers that share the same routing schema but are independently taught. Throughput scales by adding members; accuracy scales by teaching them well.

Together these create the loop: *earn coins → raise shape pressure → build clusters to keep up → train them well to stay profitable → earn more coins*.

## Non-regression constraint

**Players who never use clusters must see zero change in how single workers feel.** A size-1 cluster renders at the current `NODE_H = 50` with the current visual density. The slim-row rendering (`MEMBER_H = 28`) only activates when a cluster has 2 or more members. Radial menu, flow animation anchors, selection, and node info panel all behave identically for size-1 clusters.

## Data model

Approach A (cluster as first-class routing unit) was chosen over the thin-grouping alternative because edges, queue, and class_names all belong conceptually to the cluster, not to any individual member.

### `RoutingCluster`

Replaces `RoutingNode` as the primary entity in `RoutingGraph`.

```python
@dataclass
class ClusterMember:
    worker: FactoryWorker           # independently trained — own weights, own support set
    flash_timer: float = 0.0        # seconds remaining on aqua flash
    correct: int = 0                # per-member stats for debugging which member is weak
    wrong: int = 0

class RoutingCluster:
    cluster_id: str
    members: list[ClusterMember]    # at least 1
    class_names: list[str]          # shared schema — members' workers mirror this
    edges: list[RoutingEdge]        # shared routing — edges out of the cluster
    queue: deque                    # one queue for the whole cluster
    queue_capacity: int = 10
```

`RoutingGraph` stores `clusters: dict[str, RoutingCluster]` and `root_id: str | None` (points to a cluster_id). The old `nodes` attribute is removed.

**Invariant:** every `member.worker.class_names == cluster.class_names`. Mutations to `class_names` (rename, add, remove) propagate to all members and trigger `_rebuild_head()` and adaptation-cache invalidation on each. Support sets are per-member and untouched by schema changes.

### Processing speed

`processing_speed` is no longer a settable integer on a node driven by a global speed level. It is derived: `cluster.processing_speed == len(cluster.members)`. Adding members is the only way to raise per-cluster throughput.

## Tick & routing flow

### `RoutingGraph.process_tick()`

- New objects enqueue into the root **cluster**'s queue.
- BFS iterates over clusters.
- For each cluster with a non-empty queue, process up to `len(members)` objects:
  - For each object: `member = random.choice(cluster.members)`
  - Set `member.flash_timer = FLASH_DURATION` (0.35s)
  - Run `member.worker.predict_real(obj.tensor)`
  - Increment `member.correct` or `member.wrong` based on ground truth (same logic as current worker stats)
  - Route via `cluster.edges`; target is another `cluster_id` or `BIN:name`
- `TickResults.flows` gains a `member_idx: int` field so the UI can draw flow entry/exit at the correct row within the cluster rect.

### `FactoryWorld.tick()`

Changes:
- Remove the `for node in self.graph.nodes.values(): node.processing_speed = self.speed_level` loop entirely.
- Remove the call to `_maybe_ramp_throughput()`.
- `self.objects_per_tick` is set by the shop upgrade, initialized to 1.

Deleted: `_maybe_ramp_throughput`, `THROUGHPUT_RAMP_INTERVAL`, `MAX_OBJECTS_PER_TICK`, `speed_level`, `SPEED_UPGRADE_BASE_COST`, `SPEED_UPGRADE_SCALE`, `get_speed_upgrade_cost`, `buy_speed_upgrade`.

### Tick rate

`base_tick_interval = 1.5` is now fixed. The division by `speed_level` at `factory_floor.py:627` is removed; `tick_interval = self.base_tick_interval`. Flow animations also drop their `speed_mult` parameter (or hardcode it to 1.0).

## Shop upgrade

Added to `FactoryWorld`:

```python
MAX_OBJECTS_PER_TICK = 4
THROUGHPUT_COSTS = [200, 500, 1200]   # cost to reach level 2, 3, 4

objects_per_tick: int = 1

def get_throughput_upgrade_cost(self) -> float | None:
    if self.objects_per_tick >= self.MAX_OBJECTS_PER_TICK:
        return None
    return self.THROUGHPUT_COSTS[self.objects_per_tick - 1]

def buy_throughput_upgrade(self) -> bool:
    cost = self.get_throughput_upgrade_cost()
    if cost is None:
        return False
    if self.economy.spend(cost):
        self.objects_per_tick += 1
        return True
    return False
```

Flat cost curve (not exponential) because there are only three purchases total — the top end is the end-state, not a treadmill.

### Shop button (side panel)

Same pill location as the old speed button (`factory_floor.py:1320–1340`). New label logic:

```python
level = self.world.objects_per_tick
cost = self.world.get_throughput_upgrade_cost()
if cost is None:
    label = "Shapes/tick: 4 (MAX)"
    enabled = False
else:
    label = f"Shapes/tick {level}\u2192{level+1}  (${cost:.0f})"
    enabled = self.world.economy.can_afford(cost)
```

Same pill styling; disabled state uses the existing grey `(140, 140, 140)` color. No layout shift.

The factory stats section (lines 1361–1371) replaces `("Speed", f"Lv.{...}")` with `("Shapes/tick", str(objects_per_tick))`.

## Cluster UI

### Rendering constants

```python
CLUSTER_W = 140                    # replaces NODE_W
MEMBER_H_SLIM = 28                 # each member row in a multi-member cluster
MEMBER_H_SOLO = 50                 # size-1 cluster — matches old NODE_H
MEMBER_GAP = 2                     # gap between member rows
FLASH_DURATION = 0.35              # seconds
FLASH_COLOR = (120, 220, 230)      # aqua
```

### Cluster rect geometry

```python
def cluster_height(cluster):
    n = len(cluster.members)
    if n == 1:
        return MEMBER_H_SOLO
    return n * MEMBER_H_SLIM + (n - 1) * MEMBER_GAP
```

The cluster's bounding rect has width `CLUSTER_W` and the height above. The force-directed layout receives these rects directly — it already takes rects of arbitrary size, so no algorithmic change is needed.

Per-member rect: row `i` (0-indexed from top) sits at `rect.x, rect.y + i * (MEMBER_H_SLIM + MEMBER_GAP)` with size `CLUSTER_W × MEMBER_H_SLIM`. For size-1 clusters the single member rect == the cluster rect.

### Drawing

**Size-1 cluster:** draw identically to today's node. Worker name, accuracy, speed (always 1 now — likely drop the `spd=1` text since it's always 1 for size-1), queue badge, selection highlight, root dot. Visually indistinguishable from current.

**Multi-member cluster:**
- Draw cluster outline rect (rounded 6px) as a single shell behind the members.
- Draw each member as its own sub-rect with a 1px separator line between them.
- Per-member content: worker name on the left, accuracy `%` on the right. No per-member queue (queue is cluster-level).
- Queue badge drawn once at the top-right of the cluster shell.
- Root dot drawn at top-left of the cluster shell.
- Selection highlight: the whole cluster shell uses `NODE_SELECTED` fill.
- **Aqua flash:** during update(), decay each member's `flash_timer` by `dt`. When drawing, if `flash_timer > 0`, fill the member's sub-rect with `FLASH_COLOR` blended by `alpha = min(1.0, flash_timer / FLASH_DURATION) * 180/255`. Uses a per-frame alpha surface, not `rounded_rectangle` (per commit 9205ea9).

### Flow animation anchors

`TickResults.flows` entries now carry `member_idx`. When the UI spawns a `FlowShape`:

- **Incoming flow to a cluster:** destination is the top-left of the specific member row that processed the object (`rect.x, rect.y + member_idx * (MEMBER_H_SLIM + MEMBER_GAP) + MEMBER_H_SLIM/2` for multi-member; `rect.midleft` for size-1).
- **Outgoing flow from a cluster:** origin is the right edge of that same member row.
- This makes the aqua flash and the flow visually reinforce each other — the player sees *which* member routed the object.

### Radial menu

New button appended to `RADIAL_BUTTONS`:

```python
("add_to_cluster", "Add to Cluster", 14, (100, 180, 200), "_draw_icon_stack", 90),
```

Angle 90° (straight down) — currently unused. Existing 5 buttons stay at their angles; the radial-position logic already handles variable button counts.

New icon `_draw_icon_stack`: three stacked horizontal rounded bars. Simple, reads at 14px radius.

### Add-to-cluster flow

1. Player selects a cluster, opens radial menu, clicks "Add to Cluster".
2. Dialog: `"New cluster member name:"` (reuses `_show_text_dialog`).
3. On confirm: `world.hire_worker(name, binary=...)` using the same `binary` flag the cluster currently uses (inferred from `len(cluster.class_names) == 2`). This costs the normal `HIRE_COST = 100`.
4. Overwrite the new worker's `class_names` to match `cluster.class_names`; call `_rebuild_head()`; invalidate adaptation cache.
5. Append a new `ClusterMember` to `cluster.members`.
6. `_layout_dirty = True` so the layout re-runs to accommodate the taller cluster rect.
7. Status message: `"Added member '{name}' to cluster — train it!"`

### Schema propagation

Whenever cluster `class_names` is mutated (via training dialogs that add/rename/remove classes), a helper `_sync_cluster_schema(cluster)` iterates every member and:
- sets `member.worker.class_names = cluster.class_names.copy()`
- calls `member.worker._rebuild_head()`
- calls `member.worker.invalidate_adaptation_cache()` (existing method on `FactoryWorker`)

Support sets (teaching examples) are per-member and never touched by this sync.

The first member added to a newly created cluster inherits its schema from its own training (same as today). Subsequent members are synced down from the cluster on add (warning shown in the confirmation status if the incoming worker had any pre-existing class names that would be overwritten — but since new members come from `hire_worker`, they start blank, so the warning is rarely triggered).

## Migration & cleanup

Files touched:

- `app/factory/routing.py` — `RoutingNode` → `RoutingCluster` + `ClusterMember`. `RoutingGraph.nodes` → `RoutingGraph.clusters`. `process_tick` updated. `_bfs_order` iterates clusters.
- `app/factory/world.py` — remove speed upgrade code, remove throughput ramp, add `buy_throughput_upgrade`. `hire_worker` itself is unchanged (it just creates a `FactoryWorker`); the UI's `_action_add_router` / `_action_add_specialist` are what change: instead of `graph.add_node(...)` + `assign_worker`, they call `graph.add_cluster(cluster_id, initial_member_worker=worker)`. `get_stats` reports `objects_per_tick` not `speed_level`.
- `app/factory/economy.py` — no changes (rewards/penalties unchanged).
- `app/ui/factory_floor.py` — shop button rewrite, cluster rendering, radial menu entry, flow animation rework, schema sync helper, remove all `speed_level` / `speed_mult` references, rename `_node_rects` → `_cluster_rects`, update training flows to go through `_sync_cluster_schema`.
- `app/ui/graph_layout.py` — rename `BIN_W/BIN_H` constants are fine as-is; add `CLUSTER_W` (replaces NODE_W) and accept variable-height rects per cluster (already does). `ForceLayout` signature stays rect-based.
- `app/factory/save_load.py` (if present) — bump version, serialize clusters with members, class_names, edges, queue_capacity, and each member's worker state. Migration from v1 (node-based) saves: each old node becomes a size-1 cluster; old `speed_level` becomes max of 1 and `objects_per_tick` rounded up to nearest valid value.
- `app/simulation/headless.py` — update construction code to use the cluster API.
- `tests/` — empty today, stays empty (manual test plan below).

## Testing (manual)

1. Boot game. Verify shop button shows `Shapes/tick 1→2 ($200)`. Verify stats show `Shapes/tick: 1`. Verify `Speed` row is gone.
2. Buy upgrade. Verify 2 shapes spawn per tick. Button now shows `2→3 ($500)`.
3. Buy to max. Verify `Shapes/tick: 4 (MAX)`, button disabled.
4. Add a router, name it, give it two output labels. Verify it renders at the full 50px height, visually identical to today.
5. Train it to a reasonable accuracy. Verify flow animations still anchor to midleft/midright of the node.
6. Open radial menu on the trained router. Verify the new "Add to Cluster" button appears at the bottom (90°). Click it.
7. Dialog prompts for a name. Enter `"member2"`. Verify:
   - A second slim member row appears; the cluster rect is now ~58px tall.
   - Both member rows show a name and `0%` accuracy (the new one is untrained).
   - Class names on `member2.worker` match the original member.
   - Graph layout shifts cleanly to accommodate the new height.
8. Unpause. Verify:
   - Each tick, one object is routed to one of the two members at random.
   - The member that processes flashes aqua briefly.
   - The untrained member makes more wrong calls; more red flow animations come out of its row specifically.
9. Select the cluster → Train Worker. Verify the training UI defaults to the first member (or lets you pick which to train — decide during impl: start with "first member only" for simplicity).
10. Rename a class during training. Verify both members' `class_names` update and both heads rebuild.
11. Save. Reload. Verify the cluster comes back with both members, correct class names, both workers restored.
12. Delete the second member (via radial menu? — decide during impl: reuse "Remove Node" to mean "remove selected member if multi-member, else remove cluster"). Verify cluster shrinks back to solo and reverts to 50px height.
13. Regression: start a fresh game, build a single-worker factory exactly like yesterday, never touch clusters. Verify it looks and plays identical to pre-change behavior.

## Open decisions (deferred to implementation)

- **Which member does "Train Worker" target in a multi-member cluster?** Start with "first member only" — keeps the UI simple. If playtesting reveals a need, add per-member selection later.
- **How does "Remove Node" behave on a multi-member cluster?** Start with: removes the cluster entirely (with confirmation if `len(members) > 1`). Adding a per-member removal is a follow-up if needed.
- **Save/load version bump details** — handled in implementation plan.

## Vision alignment

This change reinforces the game's core thesis: workers are real neural networks the player teaches, and the gameplay reward loop is about *investing in teaching*. Clusters make that investment visible — you can literally see which member is weak (aqua flash + red flow), and your coins go toward training each member to pull its weight. The shop upgrade gives the player a clean pressure lever that forces cluster investment at exactly the right moments.
