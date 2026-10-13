// ctx.thread, ctx.warp and ctx.lane: one CTA of 256 threads as 8 rows of 32 lanes. Hovering or
// focusing a thread names its indices and the relation thread = warp * lanes + lane. The loop mode
// runs `for k in range(ctx.lane, K, ctx.lanes)` over one row of W: each iteration the warp reads 32
// neighboring elements, and the picked lane's elements light in its column. Schematic.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { clamp } from "../../../lib/figure/math"
import { StarPacket } from "../../../lib/figure/packet"
import { Polyline } from "../../../lib/figure/polyline"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import { LANES } from "../ctx-hierarchy/rules"
import { enablePicking, makePickable } from "../ctx-hierarchy/pick"
import {
  DEFAULT_THREAD,
  formulaOf,
  indicesGeometry,
  indicesOf,
  ITERATIONS,
  THREADS,
  type IndicesGeometry,
  type Rect,
  type TextAt,
  type ThreadIndices,
} from "./geometry"

const PACKET_SPEED_UNITS_PER_MS = 0.3
const ITERATION_HOLD_MS = 180
// A static lane tick this close to the picked lane's column gives way to the picked lane's number.
const TICK_CLEARANCE_LANES = 2

type Mode = "indices" | "loop"

export default class CtxIndicesFigure extends ChartFigure {
  #geometry: IndicesGeometry | null = null
  #cells: SVGRectElement[][] = []
  #strip: SVGRectElement[][] = []
  #stripGroup: SVGGElement | null = null
  #warpLabels: SVGTextElement[] = []
  #laneTicks: SVGTextElement[] = []
  #laneHot: SVGTextElement | null = null
  #formula: SVGTextElement | null = null
  #numbers: SVGTextElement | null = null
  #fields: SVGTextElement[] = []
  #loopLines: SVGTextElement[] = []
  #packet: StarPacket | null = null
  #entering: SVGElement[] = []
  #pressed: SVGRectElement | null = null
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ thread: 160 }, { thread: 113 }, { mode: "loop" }, { thread: 17 }]
  }

  get #indices(): ThreadIndices {
    return indicesOf(clamp(Math.round(Number(this.state.thread ?? DEFAULT_THREAD)), 0, THREADS - 1))
  }

  get #mode(): Mode {
    return this.state.mode === "loop" ? "loop" : "indices"
  }

  override start(): void {
    super.start()
    enablePicking(this.svg, { pickOnMove: true })
  }

  override build(layout: Layout): void {
    const geometry = indicesGeometry(layout)
    this.#geometry = geometry
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    const root = this.drawing
    plate(root, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: geometry.plateLabel,
      schematic: true,
    })
    const text = (at: TextAt, className?: string) =>
      label(root, at.text, { x: at.x, y: at.y, anchor: at.anchor ?? "start", className })

    const selected = this.#indices.thread
    const ctaGroup = svgEl("g", { class: "chart-cells" }, root)
    this.#cells = geometry.cells.map((row, warp) =>
      row.map((at, lane) => {
        const cell = rect(ctaGroup, at)
        const thread = warp * LANES + lane
        makePickable(cell, {
          group: "thread",
          value: String(thread),
          label: `Thread ${thread}, warp ${warp}, lane ${lane}`,
          row: warp,
          col: lane,
          pressed: thread === selected,
        })
        return cell
      })
    )
    this.#warpLabels = geometry.warpLabels.map(at => text(at, "is-small"))
    this.#laneTicks = geometry.laneTicks.map(at => text(at, "is-small"))
    this.#laneHot = text({ ...geometry.laneTicks[0], text: "" }, "is-small")
    const laneAxis = text(geometry.laneAxis, "is-small")

    this.#stripGroup = svgEl("g", { class: "sp-global" }, root)
    const stripCells = svgEl("g", { class: "chart-cells" }, this.#stripGroup)
    this.#strip = geometry.strip.map(row => row.map(at => rect(stripCells, at)))
    for (const at of geometry.stripLabels)
      label(this.#stripGroup, at.text, { x: at.x, y: at.y, anchor: at.anchor, className: "is-small" })
    label(this.#stripGroup, geometry.stripTitle.text, {
      x: geometry.stripTitle.x,
      y: geometry.stripTitle.y,
      className: layout === "narrow" ? "is-small" : undefined,
    })

    const { panel } = geometry
    this.#formula = label(root, "thread = warp * lanes + lane", { x: panel.x, y: panel.formulaY })
    this.#numbers = label(root, "", { x: panel.x, y: panel.numbersY, className: "is-hot" })
    this.#fields = panel.fieldYs.map(y => label(root, "", { x: panel.x, y, className: "is-small" }))
    this.#loopLines = panel.loopYs.map(y => label(root, "", { x: panel.x, y, className: "is-small" }))
    this.#packet = new StarPacket(root, { space: "global" })
    this.#entering = [...this.#warpLabels, laneAxis, this.#formula, this.#numbers, ...this.#fields]
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    this.motion.finishAll()
    const timeline = this.#timeline ?? this.#makeTimeline(false)
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) {
      this.motion.drawIn(this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks"))
      this.motion.enter(this.#entering, { delayMs: 120 })
    }
    super.onFirstVisible()
  }

  override onChange(group: string): void {
    if (group === "mode") this.#run(this.#mode === "loop")
    if (group === "thread") this.#showNow()
  }

  override onInput(param: string): void {
    if (param === "thread") this.#showNow()
  }

  override onAction(action: string): void {
    if (action !== "replay") return
    this.setState({ mode: "loop" })
    this.#run(true)
  }

  // A new thread shows at once: hovering across the grid should not queue animations.
  #showNow(): void {
    const output = this.root.querySelector("output[data-for='thread']")
    if (output) output.textContent = String(this.#indices.thread)
    this.#timeline = null
    this.#makeTimeline(false).seekEnd()
  }

  #run(animated: boolean): void {
    this.#timeline = this.#makeTimeline(animated)
    if (animated) this.play()
    else this.settle()
  }

  #makeTimeline(animated: boolean): Timeline {
    const indices = this.#indices
    const isLoop = this.#mode === "loop"
    const steps: TimelineStep[] = [{ durationMs: 0, apply: () => this.#applyThread(indices, isLoop) }]
    if (isLoop) {
      for (let iteration = 0; iteration < ITERATIONS; iteration += 1)
        steps.push(...this.#iteration(indices, iteration, animated))
    }
    return new Timeline(steps)
  }

  #iteration(indices: ThreadIndices, iteration: number, animated: boolean): TimelineStep[] {
    const from = this.#geometry?.strip[iteration][indices.lane]
    const to = this.#geometry?.cells[indices.warp][indices.lane]
    if (!from || !to) return []
    const x = from.x + from.width / 2
    const line = new Polyline([
      { x, y: from.y },
      { x, y: to.y + to.height },
    ])
    return [
      {
        durationMs: animated ? line.lengthUnits / PACKET_SPEED_UNITS_PER_MS : 0,
        apply: progress => {
          if (animated) this.#packet?.placeAtFraction(line, progress)
          const arrived = progress >= 1
          for (const cell of this.#strip[iteration]) cell.classList.toggle("is-on", arrived)
          this.#strip[iteration][indices.lane].classList.toggle("is-hot", arrived)
          this.#strip[iteration][indices.lane].classList.toggle("sp-reg", arrived)
          const readsLine = this.#loopLines[1]
          if (readsLine) readsLine.textContent = indices.reads.slice(0, arrived ? iteration + 1 : iteration).join(", ")
        },
      },
      { durationMs: animated ? ITERATION_HOLD_MS : 0, apply: () => {} },
    ]
  }

  #applyThread(indices: ThreadIndices, isLoop: boolean): void {
    const { thread, warp, lane } = indices
    this.#packet?.hide()
    this.#cells.forEach((row, rowWarp) =>
      row.forEach((cell, rowLane) => {
        const isThread = rowWarp === warp && rowLane === lane
        cell.classList.toggle("is-on", rowWarp === warp && !isThread)
        cell.classList.toggle("is-hot", isThread)
        cell.classList.toggle("sp-reg", isThread)
      })
    )
    const pressed = this.#cells[warp]?.[lane] ?? null
    if (this.#pressed !== pressed) {
      this.#pressed?.setAttribute("aria-pressed", "false")
      pressed?.setAttribute("aria-pressed", "true")
      this.#pressed = pressed
    }
    this.#warpLabels.forEach((warpLabel, index) => warpLabel.classList.toggle("is-hot", index === warp))
    this.#laneTicks.forEach((tick, index) =>
      tick.classList.toggle("is-hidden", Math.abs(index * 4 - lane) <= TICK_CLEARANCE_LANES)
    )
    const cell = this.#geometry?.cells[0][lane]
    if (this.#laneHot && cell) {
      this.#laneHot.textContent = String(lane)
      this.#laneHot.setAttribute("x", String(cell.x + cell.width / 2))
      this.#laneHot.classList.add("is-hot")
    }

    if (this.#numbers) this.#numbers.textContent = formulaOf(indices)
    const values = [
      `ctx.thread = ${thread}`,
      `ctx.warp = ${thread} / ${LANES} = ${warp}`,
      `ctx.lane = ${thread} - ${warp} * ${LANES} = ${lane}`,
    ]
    this.#fields.forEach((field, index) => {
      field.textContent = values[index]
    })

    this.#stripGroup?.classList.toggle("is-idle", !isLoop)
    for (const row of this.#strip) {
      for (const stripCell of row) stripCell.classList.remove("is-on", "is-hot", "sp-reg")
    }
    const loopText = isLoop ? [`lane ${lane} reads k =`, "", "32 neighbors per iteration"] : ["", "", ""]
    this.#loopLines.forEach((loopLine, index) => {
      loopLine.textContent = loopText[index]
      loopLine.classList.toggle("is-hot", index === 1)
    })

    this.setReadout(
      isLoop
        ? `Lane ${lane} of warp ${warp} reads k = ${indices.reads.join(", ")}. On each iteration the warp's 32 lanes read 32 neighboring elements.`
        : `Thread ${thread} is lane ${lane} of warp ${warp}: ${formulaOf(indices)}.`
    )
  }
}

function rect(parent: Element, at: Rect): SVGRectElement {
  return svgEl("rect", { class: "chart-cell", x: at.x, y: at.y, width: at.width, height: at.height }, parent)
}
