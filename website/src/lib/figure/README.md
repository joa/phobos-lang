# Figure library

The base for every figure on the site: the visualizer, the `ctx` figures, the front page and inference figures.
Figures are drawn as a star chart: a dotted graticule plate with corner registration ticks and a mono plate label,
1-unit hairlines that end in dots, square corners, point-cloud packets, Noto Sans Mono labels. The look lives in
`src/styles/chart.css`; this folder holds the mechanics.

## Rules

- JS never writes a color. Every visual state is a class; `chart.css` decides the color.
- Hue says where data lives. Put a space class on an element or a group and the state classes read it:
  `.sp-global` (`--brand`), `.sp-shared` (`--nebula`), `.sp-reg` (`--star`, registers and a thread's own values),
  `.sp-cost` (`--mars`). A figure that sets no space draws in `--brand`.
- States: `.is-on` (active, in its space's hue), `.is-hot` (look here), `.is-warn` (cost: `--mars`, always dashed),
  `.is-idle` (dimmed), `.is-hidden`. No state class is the resting look: `--sky-line` hairlines.
- Hue never carries meaning alone: each space has its own place in the layout and a label; cost is also dashed.
- Labels are mono, sentence case, at least 12.5 units (`.chart-label` is 13.5, `.chart-label.is-small` 12.5), so the
  smallest renders at 11 px in a 640 px docs column. Only plate labels are uppercase.
- Anything illustrative is marked: `plate(..., { schematic: true })` prints "schematic" on the plate. Never invent a
  number; a figure that states one reads it from data.
- Motion: only while at least 30% of the figure is on screen in a visible tab; under `prefers-reduced-motion` nothing
  moves and the figure shows its settled end state, with no tour and no loop.
- Figures loop. Once a run (and its tour) has finished, the end state holds for 1.5 s, then cross-fades into the next
  run. Off screen or in a hidden tab the hold pauses with the frame loop.

## Files

| File | What it gives you |
| --- | --- |
| `chart-figure.ts` | `ChartFigure`, the base class: lifecycle, layouts, frame loop, state from controls, tours, readout, static rendering |
| `draw.ts` | Star chart primitives: `plate`, `box`, `cellGrid`, `wire`, `node`, `label`, `glow`, the space classes |
| `packet.ts` | `StarPacket`: a point cloud of particles on three depth layers behind a glow |
| `motion.ts` | `Motion` (draw-in, entrances, FLIP glide, arrival pulse), `viewTransition`, `setBreathing`, `sweepWire`, the motion tokens |
| `timeline.ts` | `Timeline`: steps run back to back, each a pure function of its progress; `LoopClock`, the loop's hold |
| `polyline.ts` | `Polyline` (constant-speed paths), `elbow` (right-angled routes) |
| `layout.ts` | `layoutFor` (600 px breakpoint), `VIEW_WIDTH_UNITS`, graticule and tick geometry |
| `math.ts` | `mulberry32`, `clamp`, `lerp`, easings, `cubicBezier`, `easeOut` / `easeInOut` matching the CSS tokens |
| `svg.ts` | `svgEl`, `svgText`, `setAttributes` |
| `static.ts` | The minimal element static rendering draws into |
| `index.ts` | Everything above, for one import |

Nothing here touches the DOM at module level, so pure parts (geometry, a timeline's logic) import in Node and in
vitest.

## A figure in four parts

1. **A module** named `<name>.figure.ts` anywhere under `src/`, whose default export extends `ChartFigure`.
   `src/components/figures/boot.ts` finds it by that name with `import.meta.glob` and loads it lazily, one chunk per
   figure, only on pages that have one.
2. **A geometry function**, pure, in its own file, so tests can check that nothing overlaps or leaves the plate in
   either layout. See `src/components/figures/cta-copy/geometry.ts` and `tests/figure-cta-copy.test.ts`.
3. **An Astro component** that renders `src/components/figures/ChartFigure.astro`:

   ```astro
   <ChartFigure
     id="fig-example"
     name="example"
     figure={ExampleFigure}
     state={{ mode: "a" }}
     title="What the figure shows"
     desc="The same in a sentence or two, for screen readers."
     legend={["global", "shared"]}
   >
     <div class="chart-group" role="group" aria-label="Mode">
       <button type="button" class="chart-btn" data-group="mode" data-value="a" aria-pressed="true">A</button>
       <button type="button" class="chart-btn" data-group="mode" data-value="b" aria-pressed="false">B</button>
     </div>
     <Fragment slot="caption">Schematic. A caption.</Fragment>
   </ChartFigure>
   ```

   `ChartFigure.astro` runs the module's `build()` and `settle()` at build time in both layouts
   (`ChartFigure.renderStatic`), so without JavaScript the page shows the figure's end state; a container query picks
   the layout. `state` must match the pressed buttons. Ids must be unique on the page.
4. **Tests** for the geometry and, where it has logic, the figure's static render (`renderStatic` works in Node).

`src/components/figures/cta-copy/` is the worked example; `token-loop/` is a second, simpler one.

## `ChartFigure`

Subclass hooks:

- `build(layout)`: draw everything for `"wide"` (viewBox 720 units wide) or `"narrow"` (400 wide) into
  `this.drawing`, call `this.setViewBox(width, height)`, and keep references to what later steps change. A rebuild
  replaces every element, so steps look elements up through fields at apply time, never through closures. Keep wide
  at about 340 units high or less and narrow at about 540 or less. Narrow is a real rearrangement, not the wide drawing
  shrunk.
- `settle()`: jump to the end state of whatever is playing. Runs after every build, under reduced motion and when a
  timeline is cut short. With a `Timeline`, this is `timeline.seekEnd()`; call `this.motion.finishAll()` first.
- `update(dtMs)`: advance animations, return `true` to keep the frame loop going. Usually `timeline.advance(dtMs)`.
- `onChange(group, value)`, `onAction(action)`, `onInput(param, value)`: control events.
- `onFirstVisible()`: runs once, the first time the figure is in view (under reduced motion too). The default runs the
  tour. Override it to add a draw-in, then call `super.onFirstVisible()`.
- `get tour()`: state patches such as `[{ phase: "tile" }, { phase: "load" }]`, applied as if clicked, each held
  1.4 s after its animation ends. Any reader input cancels it; reduced motion skips it.
- `replay()`: starts a loop's next run. The default plays the tour again while the reader has not touched the figure,
  else fires the `replay` action. A figure with neither stops after one run; override it to loop anyway. If it starts
  nothing, the loop rests until the next change.
- `get loops()` and `get loopFades()` (both `true` by default): turn the loop, or its cross-fade, off. A figure whose
  end and start look the same (only packets move) can skip the fade.

Helpers and getters: `play()` (starts the frame loop, or settles under reduced motion), `pause()`, `crossfade()`
(lays a copy of the drawing over it that fades out while the drawing fades in; call it before changing state),
`setReadout(text)` (writes the `aria-live` readout; call it on a state change, never per frame), `setViewBox`,
`setState(patch)` (changes state and controls without calling hooks), `root`, `id`, `svg`, `drawing`, `layout`,
`state`, `motion`, `reducedMotion`, `isVisible`, `hasInteracted`.

Static rendering: `build()` and `settle()` may use attributes, `classList`, `style.setProperty`, `textContent`,
`append` and `remove`. Measuring (`getBBox`, `getTotalLength`) and `querySelector` inside the drawing belong in the
motion paths (`onFirstVisible`, `onChange`), which never run at build time.

## Drawing

```ts
plate(this.drawing, this.id, { widthUnits, heightUnits, label: "FIG / GEMM / SHARED", schematic: true })
const cells = cellGrid(group, { x, y, rows: 8, cols: 8, cellUnits: 11, space: "shared" })
const link = wire(this.drawing, elbow(from, to), { space: "global" }) // { group, path, nodes }
label(this.drawing, "shared  a", { x, y })
const halo = glow(this.drawing, x, y, radiusUnits, "reg")
```

`cellGrid` cells take `.is-on` / `.is-hot`. Wires and their end and vertex dots take state classes on `wire.group`.
`Polyline.simplified()` drops vertices on a straight run, so an elbow that came out straight gets no extra dots.

## Motion

Tokens in `tokens.css`: `--dur-1` 120 ms (hover), `--dur-2` 240 ms (a state change), `--dur-3` 480 ms (a layout
move), `--dur-4` 900 ms (an entrance), `--ease-out`, `--ease-in-out`, `--stagger` 30 ms. `readMotionTokens()` reads
them at run time; under reduced motion they shrink to 120 ms or 0. In a frame loop, use `easeOut` / `easeInOut` from
`math.ts`, which match the CSS curves.

- **State changes cross-fade.** Toggle classes; fill, stroke and opacity transition over `--dur-2` in CSS.
- **Draw-in.** `this.motion.drawIn(paths, { staggerMs })`: hairlines draw from their start (dash offset over the
  measured length), `--dur-4`, staggered.
- **Entrances.** `this.motion.enter(elements, { delayMs })`: from opacity 0, 6 px down and `blur(6px)` to rest.
- **Glides.** `this.motion.glide(elements, () => moveThem())`: FLIP. Measures, lets the callback move the elements
  (attributes or classes), then glides each from where it was over `--dur-3`. Works for SVG and HTML; a glide started
  mid-glide continues from where the element is. For an HTML layout swap, `viewTransition(() => swap())` uses
  `document.startViewTransition` where the browser has it (Chrome does not capture SVG children, so SVG uses FLIP).
- **Packets.** `new StarPacket(parent, { space, count })`, then in a timeline step
  `packet.placeAtFraction(line, progress)` (hidden outside 0 to 1) or `packet.place(line, distanceUnits)`. Constant
  speed: give the step a duration of `line.lengthUnits / speed`. A packet is a cloud of `count` particles (14 by
  default, at most 24), each with a fixed lag and offset from a seeded generator, so the static render matches. It
  gathers to a point at both ends of its line. Keep `count` small where many packets move at once (the visualizer's
  32 lanes use 6).
- **Loop.** Nothing to do in a figure with a tour or a `replay` action: the base holds the end state, cross-fades
  and calls `replay()`. Figures carry no Replay button; the loop is the replay.
- **Arrival glow.** Either `this.motion.pulse(glowElement)` (fire and forget, 300 ms), or, inside a seekable timeline,
  a `glow` with class `is-pulse` whose `--pulse` property (0 to 1) you set per frame, as `cta-copy` does.
- **Breathing.** `setBreathing(glowElement, true)`: opacity between 0.85 and 1 over 2.4 s, paused while the figure is
  off screen (the base sets `data-visible` on the root).
- **Light sweep.** `sweepWire(parent, line.toPathData(), space)` returns an overlay; toggle `is-on` on it while the
  wire is active. For an HTML hairline, the class `sweep-line` (with `is-on`) runs the same sweep as a gradient whose
  offset is the typed `--sweep` property.
- **Glow** is one pre-blurred radial gradient per space, defined once per document (`ensureGlowDefs()`, called by
  `boot.ts`), never an SVG filter per element.

- **Depth.** The plate has a faint glow at its heart (`.chart-vignette`) and its graticule drifts a few units against
  the scroll (a scroll-driven CSS animation where the browser has one), so the dots read as a layer behind the drawing.
  Labels never get a 3D transform.

Budget: animate only transforms, opacity and dash or gradient offsets; one frame loop per playing figure, stopped once
the figure has finished and its loop's hold has run out; no element creation per frame.

## Static plates

For a drawing with no script at all (a front page scene), `src/components/figures/ChartPlate.astro` draws the plate
(graticule, ticks, label, "schematic") around SVG children that use the same classes.
