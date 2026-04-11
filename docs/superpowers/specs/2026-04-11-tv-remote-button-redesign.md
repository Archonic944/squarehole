# TV-Remote-Style Button Redesign

## Summary

Reshape the factory floor UI so buttons are visually distinct — different shapes, sizes, and locations — rather than a uniform column of identical 225x32 rectangles. Inspired by TV remotes where each button has a unique form factor you can find without reading the label.

## Current State

The side panel (255px, right edge) contains every button as an identical `_draw_btn()` call: same width (225), same height (32), same border-radius-5 rounded rect. The only differentiation is background color. Node-specific and global actions are stacked vertically with no spatial grouping.

## Design

### 1. Radial Node Context Menu

When a node is clicked/selected, a ring of circular buttons fans out around it in the graph area. Node-specific actions are **removed from the side panel entirely**.

#### Buttons

Arranged clockwise from top. Each is a filled circle with a procedurally-drawn icon. Larger radius = more prominent action.

| Button | Circle radius | Icon | Clock position |
|---|---|---|---|
| Train Worker | 22px | Mallet (handle + rectangular head) | 12 o'clock (top) |
| Connect To Node | 16px | Right-pointing arrow → small square | ~2 o'clock |
| Connect To Bin | 16px | Downward arrow into trapezoid/bucket | ~4 o'clock |
| Remove Node | 13px | X mark (two crossing lines) | ~8 o'clock |
| Set as Root | 14px | 5-point star outline | ~10 o'clock |

- **Set as Root** is hidden when the node is already the root. The remaining buttons redistribute evenly.
- **Ring radius:** 65px from node center.

#### Adaptive Arc (Edge Handling)

The radial menu must stay within the graph area bounds (`GRAPH_X, GRAPH_Y, GRAPH_W, GRAPH_H`).

Algorithm:
1. Compute the node center (`cx, cy`).
2. For each cardinal direction, compute available space: `space_right = (GRAPH_X + GRAPH_W) - cx`, `space_left = cx - GRAPH_X`, `space_up = cy - GRAPH_Y`, `space_down = (GRAPH_Y + GRAPH_H) - cy`.
3. The ring radius is 65px. Any direction with less than `65 + max_button_radius + 4` pixels of space is "blocked."
4. Compute the valid angular range by excluding blocked arcs. Distribute buttons evenly across the remaining arc.
5. If the node is fully interior (all directions have space), use the full 360° circle with the default clock positions above.

#### Tooltips

On hover over a radial button:
- Draw a small rounded-rect label (dark background `(50,50,50)`, white text, 4px padding, border-radius 4).
- Position: offset 8px outward from the button center (away from the node). If that would go off-screen, flip to the opposite side.
- Font: `font_sm` (13px Arial).

#### Interaction

- Clicking a radial button triggers the corresponding action (same as the old side panel buttons).
- Clicking anywhere else in the graph area dismisses the radial menu and deselects the node.
- The radial menu draws **last** in the graph area (highest z-order), on top of edges, nodes, flow shapes.
- When a node is selected but the radial is showing, clicking a *different* node should select that node and show its radial instead.

#### Visual Style

- Button fill: each button has its own base color (Train: `(80,160,80)`, Connect Node: `(130,140,170)`, Connect Bin: `(130,160,130)`, Remove: `(190,100,100)`, Set Root: `(150,150,100)`).
- On hover: lighten fill by 25 (same approach as current `_draw_btn_raw`).
- Border: 2px dark outline `(40,40,40)`.
- Drop shadow: subtle 2px offset shadow `(0,0,0,60)` behind each circle for depth.
- Icons: white `(255,255,255)` strokes/fills, drawn with `pygame.draw.line`, `pygame.draw.polygon`, `pygame.draw.circle`, `pygame.draw.arc`. Line width 2px.

### 2. Side Panel (Revised Layout)

The side panel (255px right) becomes info display + factory-building controls. No node action buttons.

#### Top: Build Buttons (side-by-side)

Two buttons placed side by side at the top of the panel:

- **Add Router**: left button, width 108px, height 38px, x = `SIDE_X + 15`, color `(100,150,200)`.
- **Add Specialist**: right button, width 108px, height 38px, x = `SIDE_X + 15 + 108 + 9`, color `(80,160,160)`.
- Border-radius: 6px.
- Icons inside (left of text): Router gets a small branching-lines icon (one line splitting into two). Specialist gets a single focused line with a dot.

#### Speed Upgrade (pill shape)

Below the build buttons, after 12px gap:

- Full-width pill: width 225px, height 34px, border-radius 17px (half of height = fully rounded ends).
- x = `SIDE_X + 15`.
- Label: `"Speed Lv.X → X+1  ($cost)"` centered.
- Color: `(100,170,100)` if affordable, `(140,140,140)` if not.
- Border: 2px, darker shade of fill color.

#### Separator

Horizontal line, 10px below speed button.

#### Node Info (when selected)

Same text content as current `_draw_node_info` but **without any buttons** (those are now in the radial menu):
- Node ID
- Worker name
- Accuracy bar (unchanged)
- Classes list
- Support count
- Speed / Queue

#### Separator

#### Factory Stats

Same as current: Workers, Nodes, Speed, Categories, Tick — rendered as `font_sm` text lines.

### 3. Bottom Bar (Revised)

Left side unchanged (active categories, objects/tick).

Right side:

- **Pause/Resume**: Circular button, diameter 36px (radius 18px). Center at `(W - 55, bottom_bar_y + 25)` — vertically centered in the 50px bar. 
  - Icon: play triangle (when paused) or two vertical pause bars (when running).
  - Fill: `(60,170,60)` when running, `(200,60,60)` when paused.
  - Border: 2px dark.
- **Status label**: "Paused"/"Running" text, right-aligned to the left of the circle, vertically centered.
- **Quit**: Small rounded-square, 55x30px, border-radius 6px. Positioned to the left of the status label with 15px gap. Color `(150,120,100)`.

### 4. Icon Drawing Reference

All icons drawn with pygame primitives. Colors: white `(255,255,255)` for icons on colored backgrounds, `TEXT_DARK` for icons on light backgrounds. Stroke width: 2px unless noted.

**Mallet (Train Worker, r=22):**
- Handle: diagonal line from center-bottom-left to center, width 3.
- Head: filled rectangle rotated ~45°, drawn as a polygon. Roughly 12x6px.

**Arrow → Square (Connect To Node, r=16):**
- Horizontal arrow: line with arrowhead (small triangle at tip), pointing right.
- Small 5x5 filled square at arrow tip.

**Arrow → Bucket (Connect To Bin, r=16):**
- Downward arrow with arrowhead.
- Trapezoid below (wider at top, narrower at bottom) — 3 lines forming an open-top bucket.

**X Mark (Remove Node, r=13):**
- Two diagonal lines crossing at center, each ~14px long.

**Star (Set as Root, r=14):**
- 5-point star outline, computed with inner/outer radius formula. Outer radius ~9px, inner ~4px.

**Play Triangle (Pause/Resume when paused):**
- Right-pointing equilateral triangle, filled.

**Pause Bars (Pause/Resume when running):**
- Two vertical filled rectangles, 4px wide, 14px tall, 5px apart.

**Router Icon (Add Router button):**
- One line on the left splitting into two lines diverging to the right (Y-shape, rotated 90°).

**Specialist Icon (Add Specialist button):**
- Single horizontal line with a circle/dot at the end (focused beam).

## Files Modified

- `app/ui/factory_floor.py` — all changes are in this single file.

## Pixel Math Constraints

- Graph area: `(0, 45)` to `(769, 718)` — width 769, height 673.
- Side panel: `(769, 45)` to `(1024, 718)` — width 255, height 673.
- Bottom bar: `(0, 718)` to `(1024, 768)` — height 50.
- Top bar: `(0, 0)` to `(1024, 45)` — height 45.
- All radial button positions must be clamped to keep the full button circle (center ± radius ± border) within the graph area bounds.
- Side panel buttons must maintain 15px left margin from panel edge (`SIDE_X + 15 = 784`).
