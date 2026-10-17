// A row-wise library function, statement by statement, on a schematic `[R, C]` tile in shared
// memory: a row reduction folds each row, left to right, into a `[R, 1]` column; an elementwise
// statement sweeps the tile with the column broadcast along its row and, for a norm, the weight row
// down its column. The statements shown are the function's own source.

import { ChartFigure } from "../../../lib/figure/chart-figure"
import { label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import { COLS, FUNCTIONS, programOf, ROWS, rowSweepGeometry, type RowFunction, type Step } from "./geometry"

const FOLD_MS = 1100
const SWEEP_MS = 1100

export default class RowSweepFigure extends ChartFigure {
  #cells: SVGRectElement[][] = []
  #column: SVGRectElement[] = []
  #weight: SVGRectElement[] = []
  #weightGroup: SVGGElement | null = null
  #tileLabel: SVGTextElement | null = null
  #columnLabel: SVGTextElement | null = null
  #statement: SVGTextElement | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour() {
    const first = this.#fn
    return [first, ...FUNCTIONS.filter(fn => fn !== first)].map(fn => ({ fn }))
  }

  get #fn(): RowFunction {
    const fn = String(this.state.fn)
    return (FUNCTIONS as readonly string[]).includes(fn) ? (fn as RowFunction) : FUNCTIONS[0]
  }

  override build(layout: Layout): void {
    const g = rowSweepGeometry(layout)
    this.setViewBox(g.widthUnits, g.heightUnits)
    const root = this.drawing
    plate(root, this.id, { widthUnits: g.widthUnits, heightUnits: g.heightUnits, label: g.plateLabel, schematic: true })
    const shared = svgEl("g", { class: "sp-shared" }, root)
    this.#cells = g.cells.map(row => row.map(rect => svgEl("rect", { class: "chart-cell", ...rect }, shared)))
    this.#column = g.column.map(rect => svgEl("rect", { class: "chart-cell", ...rect }, shared))
    this.#weightGroup = svgEl("g", { class: "sp-global" }, root)
    this.#weight = g.weight.map(rect => svgEl("rect", { class: "chart-cell", ...rect }, this.#weightGroup!))
    const { labels } = g
    this.#tileLabel = label(root, labels.tile.text, { x: labels.tile.x, y: labels.tile.y, anchor: "end" })
    this.#columnLabel = label(root, labels.column.text, { x: labels.column.x, y: labels.column.y, anchor: "middle" })
    const weightLabel = label(this.#weightGroup, labels.weight.text, {
      x: labels.weight.x,
      y: labels.weight.y,
      anchor: "end",
    })
    this.#statement = label(root, "", { x: labels.statement.x, y: labels.statement.y, anchor: "middle", className: "is-hot" })
    this.#labels = [this.#tileLabel, this.#columnLabel, weightLabel]
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
    if (group === "fn") this.#run()
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run()
  }

  #run(): void {
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  #makeTimeline(): Timeline {
    const name = this.#fn
    const { tile, steps } = programOf(name)
    const usesWeight = steps.some(step => step.readsWeight)
    const timelineSteps: TimelineStep[] = [
      {
        durationMs: 0,
        apply: () => {
          this.#weightGroup?.classList.toggle("is-hidden", !usesWeight)
          if (this.#tileLabel) this.#tileLabel.textContent = tile
          this.#paint(() => "is-on", () => null, () => null)
        },
      },
    ]
    steps.forEach((step, index) => {
      timelineSteps.push({
        durationMs: step.kind === "fold" ? FOLD_MS : SWEEP_MS,
        apply: progress => this.#show(step, progress, index === steps.length - 1, name),
      })
    })
    return new Timeline(timelineSteps)
  }

  #show(step: Step, progress: number, last: boolean, name: RowFunction): void {
    if (this.#statement && this.#statement.textContent !== step.statement) this.#statement.textContent = step.statement
    const done = progress >= 1
    if (step.kind === "fold") {
      if (this.#columnLabel && step.column) this.#columnLabel.textContent = step.column
      // Every row at once, one column of the fold at a time.
      const reached = Math.floor(progress * COLS)
      this.#paint(
        (_, col) => (done ? "is-on" : col === reached ? "is-hot" : col < reached ? "is-on" : null),
        () => (done ? "is-on" : null),
        () => null
      )
      this.setReadout(`${step.statement}: each row folds into one element of the [R, 1] column ${step.column}.`)
      return
    }
    // A sweep in row-major order, the column broadcast along each row.
    const reached = Math.floor(progress * ROWS * COLS)
    this.#paint(
      (row, col) => {
        const i = row * COLS + col
        return done ? "is-on" : i === reached ? "is-hot" : "is-on"
      },
      row => (done ? "is-on" : Math.floor(reached / COLS) === row ? "is-hot" : "is-on"),
      col => (!step.readsWeight ? null : done ? "is-on" : reached % COLS === col ? "is-hot" : "is-on")
    )
    this.setReadout(
      done && last
        ? `${name}: ${programOf(name).steps.length} statements, each one pass over the tile or its rows.`
        : `${step.statement}: one sweep over the tile, with the column broadcast along its row.`
    )
  }

  #paint(
    cell: (row: number, col: number) => "is-on" | "is-hot" | null,
    column: (row: number) => "is-on" | "is-hot" | null,
    weight: (col: number) => "is-on" | "is-hot" | null
  ): void {
    const set = (el: SVGRectElement, state: "is-on" | "is-hot" | null) => {
      el.classList.toggle("is-on", state === "is-on")
      el.classList.toggle("is-hot", state === "is-hot")
    }
    this.#cells.forEach((row, r) => row.forEach((el, c) => set(el, cell(r, c))))
    this.#column.forEach((el, r) => set(el, column(r)))
    this.#weight.forEach((el, c) => set(el, weight(c)))
  }
}
