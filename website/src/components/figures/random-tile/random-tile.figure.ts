// The standard library's counter-based random numbers on a tile: each element's counter is its flat
// index, and the cells fill in counter order with what `uniform`, `randn` or `dropout` gives it
// under the seed, computed from the library's source. Nothing but the counter and the seed decides
// a value, so the same seed draws the same tile again.

import { ChartFigure } from "../../../lib/figure/chart-figure"
import { label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline } from "../../../lib/figure/timeline"
import { callOf, draw, FUNCTIONS, randomTileGeometry, SEEDS, type RandomFunction, type RandomTileGeometry } from "./geometry"

const FILL_MS = 1400

export default class RandomTileFigure extends ChartFigure {
  #geometry: RandomTileGeometry | null = null
  #cells: SVGRectElement[] = []
  #call: SVGTextElement | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour() {
    return FUNCTIONS.map(fn => ({ fn, seed: String(this.#seed) }))
  }

  get #fn(): RandomFunction {
    const fn = String(this.state.fn)
    return (FUNCTIONS as readonly string[]).includes(fn) ? (fn as RandomFunction) : FUNCTIONS[0]
  }

  get #seed(): number {
    const seed = Number(this.state.seed)
    return (SEEDS as readonly number[]).includes(seed) ? seed : SEEDS[0]
  }

  override build(layout: Layout): void {
    const g = randomTileGeometry(layout)
    this.#geometry = g
    this.setViewBox(g.widthUnits, g.heightUnits)
    const root = this.drawing
    plate(root, this.id, { widthUnits: g.widthUnits, heightUnits: g.heightUnits, label: g.plateLabel })
    const reg = svgEl("g", { class: "sp-reg" }, root)
    this.#cells = g.cells.flat().map(rect => svgEl("rect", { class: "chart-cell", ...rect }, reg))
    const { labels } = g
    this.#labels = [
      label(root, labels.tile.text, { x: labels.tile.x, y: labels.tile.y, anchor: "end" }),
      label(root, labels.counter.text, { x: labels.counter.x, y: labels.counter.y, anchor: "middle", className: "is-small" }),
    ]
    this.#call = label(root, "", { x: labels.call.x, y: labels.call.y, anchor: "middle", className: "is-hot" })
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    const timeline = this.#timeline ?? this.#makeTimeline()
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) this.motion.enter(this.#labels, { delayMs: 120 })
    super.onFirstVisible()
  }

  override onChange(group: string, _value: string): void {
    if (group === "fn" || group === "seed") this.#run()
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run()
  }

  #run(): void {
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  #makeTimeline(): Timeline {
    const g = this.#geometry
    if (!g) return new Timeline([])
    const fn = this.#fn
    const count = g.rows * g.cols
    const drawn = draw(fn, this.#seed, count)
    return new Timeline([
      {
        durationMs: FILL_MS,
        apply: progress => {
          const call = callOf(fn)
          if (this.#call && this.#call.textContent !== call) this.#call.textContent = call
          const reached = progress >= 1 ? count : Math.floor(progress * count)
          // A dropped element stays an empty cell.
          this.#cells.forEach((cell, c) => {
            const lit = c < reached && drawn.shades[c] > 0
            cell.classList.toggle("is-hot", lit)
            if (lit) cell.setAttribute("fill-opacity", drawn.shades[c].toFixed(3))
            else cell.removeAttribute("fill-opacity")
          })
          this.setReadout(
            progress >= 1 ? drawn.readout : `${fn} fills the tile in counter order, row by row, from the seed alone.`
          )
        },
      },
    ])
  }
}
