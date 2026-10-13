// One warp per output. A decode matvec mapped onto a warp two ways: a thread per output walks its own
// row, so the warp's 32 reads land in 32 rows at once; a warp per output, as the lanes matvecs do it,
// reads one contiguous 512-byte run of one row a step and folds the 32 partial sums with
// sum<over ctx.lane>, five xor-shuffle stages. Schematic: sizes and timing are illustrative.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { box, label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import type { Rect, TextAt } from "../../inference/figure-geometry"
import { BLOCKS, LANES, ROWS, SHUFFLE_MASKS, STEPS, STEP_BYTES, qdotGeometry, type QdotGeometry } from "./geometry"

type Mapping = "thread" | "warp"

const BLOCK_MS = 90
const STEP_MS = 520
const STAGE_MS = 260
const FILL_MS = 240

export const READOUTS: Record<Mapping, string> = {
  thread:
    "A thread per output: each thread walks its own row and applies a scale every block, so the warp's 32 reads land in 32 rows at once.",
  warp: `A warp per output: 32 lanes read 16 bytes each, one contiguous ${STEP_BYTES}-byte run a step, then sum<over ctx.lane> adds their partial sums in ${SHUFFLE_MASKS.length} xor-shuffle stages.`,
}

export default class QdotMappingFigure extends ChartFigure {
  #geometry: QdotGeometry | null = null
  #rows: SVGRectElement[] = []
  #outputs: SVGRectElement[] = []
  #lanes: SVGRectElement[] = []
  #band: SVGRectElement | null = null
  #dots: SVGCircleElement[] = []
  #ticks: SVGPathElement[] = []
  #arcs: SVGPathElement[][] = []
  #stageLabel: SVGTextElement | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ mapping: "thread" }, { mapping: "warp" }]
  }

  get #mapping(): Mapping {
    return this.state.mapping === "thread" ? "thread" : "warp"
  }

  override build(layout: Layout): void {
    const geometry = qdotGeometry(layout)
    this.#geometry = geometry
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    const root = this.drawing
    plate(root, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: geometry.plateLabel,
      schematic: true,
    })
    this.#labels = []

    const global = svgEl("g", { class: "sp-global" }, root)
    this.#addLabel(global, geometry.activation.label)
    for (const block of geometry.activation.blocks) rect(global, block, "chart-cell is-on")
    this.#addLabel(global, geometry.matrix.label, "is-small")
    this.#rows = geometry.matrix.rows.map(row => rect(global, row, "chart-cell"))
    svgEl("path", { class: "chart-wire qm-boundaries", d: geometry.matrix.boundaries }, global)
    this.#band = box(global, { ...geometry.blockRect(0), className: "is-outline" })
    this.#ticks = Array.from({ length: LANES }, () => svgEl("path", { class: "chart-wire sp-reg" }, global))

    const reg = svgEl("g", { class: "sp-reg" }, root)
    this.#addLabel(reg, geometry.output.label, "is-small")
    this.#outputs = geometry.output.cells.map(cell => rect(reg, cell, "chart-cell"))
    this.#addLabel(reg, geometry.lanes.label, "is-small")
    this.#lanes = geometry.lanes.cells.map(cell => rect(reg, cell, "chart-cell"))
    this.#arcs = geometry.arcs.map(stage =>
      stage.map(arc => svgEl("path", { class: "chart-wire qm-arc", d: arc.d, pathLength: 1 }, reg))
    )
    this.#dots = Array.from({ length: ROWS }, () => svgEl("circle", { class: "chart-node is-on", r: 1.6 }, reg))
    this.#stageLabel = label(root, "", {
      x: geometry.stageLabel.x,
      y: geometry.stageLabel.y,
      anchor: geometry.stageLabel.anchor ?? "start",
      className: "is-small",
    })
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    this.motion.finishAll()
    const timeline = this.#timeline ?? this.#makeTimeline(this.#mapping)
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) {
      const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-box")
      this.motion.drawIn(lines)
      this.motion.enter(this.#labels, { delayMs: 240 })
    }
    super.onFirstVisible()
  }

  override onChange(group: string, value: string): void {
    if (group === "mapping") this.#run(value === "thread" ? "thread" : "warp")
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run(this.#mapping)
  }

  #run(mapping: Mapping): void {
    this.#timeline = this.#makeTimeline(mapping)
    this.play()
  }

  #makeTimeline(mapping: Mapping): Timeline {
    const steps: TimelineStep[] = [{ durationMs: 0, apply: () => this.#reset(mapping) }]
    if (mapping === "thread") {
      for (let block = 0; block < BLOCKS; block += 1) {
        steps.push({ durationMs: BLOCK_MS, apply: progress => progress > 0 && this.#threadAt(block) })
      }
      steps.push({ durationMs: FILL_MS, apply: progress => this.#finishThread(progress) })
    } else {
      for (let step = 0; step < STEPS; step += 1) {
        steps.push({ durationMs: STEP_MS, apply: progress => progress > 0 && this.#warpAt(step) })
      }
      SHUFFLE_MASKS.forEach((_, stage) => {
        steps.push({ durationMs: STAGE_MS, apply: progress => this.#stageAt(stage, progress) })
      })
      steps.push({ durationMs: FILL_MS, apply: progress => this.#finishWarp(progress) })
    }
    return new Timeline(steps)
  }

  #reset(mapping: Mapping): void {
    for (const row of this.#rows) row.classList.remove("is-on", "is-hot", "is-idle")
    for (const cell of this.#outputs) cell.classList.remove("is-on")
    for (const tick of this.#ticks) tick.classList.add("is-hidden")
    for (const dot of this.#dots) dot.classList.add("is-hidden")
    for (const stage of this.#arcs) for (const arc of stage) setArc(arc, 0, false)
    for (const lane of this.#lanes) {
      lane.classList.toggle("is-on", mapping === "thread")
      lane.classList.remove("is-hot")
    }
    this.#band?.classList.remove("is-warn", "is-hot")
    this.#band?.classList.add("is-hidden")
    if (this.#stageLabel) this.#stageLabel.textContent = ""
    this.setReadout(READOUTS[mapping])
  }

  // Each of the 32 threads sits in its own row, all in the same block column.
  #threadAt(block: number): void {
    const geometry = this.#geometry
    if (!geometry || !this.#band) return
    const column = geometry.blockRect(block)
    placeRect(this.#band, column)
    this.#band.classList.remove("is-hidden")
    this.#band.classList.add("is-warn")
    const x = column.x + column.width / 2
    geometry.matrix.rows.forEach((row, index) => {
      const dot = this.#dots[index]
      dot.setAttribute("cx", String(x))
      dot.setAttribute("cy", String(row.y + row.height / 2))
      dot.classList.remove("is-hidden")
      this.#rows[index].classList.add("is-on")
    })
    if (this.#stageLabel) this.#stageLabel.textContent = `block ${block + 1} of ${BLOCKS}, scale`
  }

  #finishThread(progress: number): void {
    if (progress < 1) return
    this.#band?.classList.add("is-hidden")
    for (const dot of this.#dots) dot.classList.add("is-hidden")
    for (const cell of this.#outputs) cell.classList.add("is-on")
    if (this.#stageLabel) this.#stageLabel.textContent = "32 rows, 32 outputs"
  }

  // Row 0 belongs to this warp; the other rows go to other warps at the same time.
  #warpAt(step: number): void {
    const geometry = this.#geometry
    if (!geometry || !this.#band) return
    placeRect(this.#band, geometry.stepRect(step))
    this.#band.classList.remove("is-hidden", "is-warn")
    this.#band.classList.add("is-hot")
    this.#rows.forEach((row, index) => {
      row.classList.toggle("is-hot", index === 0)
      row.classList.toggle("is-idle", index !== 0)
      row.classList.toggle("is-on", index !== 0)
    })
    this.#ticks.forEach((tick, lane) => {
      const at = geometry.laneTick(step, lane)
      tick.setAttribute("d", `M${at.x} ${at.y1}V${at.y2}`)
      tick.classList.remove("is-hidden")
      tick.classList.add("is-on")
    })
    for (const lane of this.#lanes) lane.classList.add("is-on")
    if (this.#stageLabel) this.#stageLabel.textContent = `step ${step + 1} of ${STEPS}, ${STEP_BYTES} B`
  }

  #stageAt(stage: number, progress: number): void {
    if (progress <= 0) return
    this.#band?.classList.add("is-hidden")
    for (const tick of this.#ticks) tick.classList.add("is-hidden")
    this.#arcs.forEach((arcs, index) => {
      for (const arc of arcs) {
        if (index < stage) setArc(arc, 1, false)
        else if (index === stage) setArc(arc, progress, true)
        else setArc(arc, 0, false)
      }
    })
    if (this.#stageLabel) this.#stageLabel.textContent = `xor ${SHUFFLE_MASKS[stage]}`
  }

  #finishWarp(progress: number): void {
    if (progress < 1) return
    this.#lanes.forEach((lane, index) => lane.classList.toggle("is-hot", index === 0))
    for (const cell of this.#outputs) cell.classList.add("is-on")
    this.#outputs[0]?.classList.add("is-hot")
    if (this.#stageLabel) this.#stageLabel.textContent = `lane 0 writes the sum`
  }

  #addLabel(parent: Element, at: TextAt, className?: string): SVGTextElement {
    const text = label(parent, at.text, { x: at.x, y: at.y, anchor: at.anchor ?? "start", className })
    this.#labels.push(text)
    return text
  }
}

function rect(parent: Element, at: Rect, className: string): SVGRectElement {
  return svgEl("rect", { class: className, x: at.x, y: at.y, width: at.width, height: at.height }, parent)
}

function placeRect(element: SVGRectElement, at: Rect): void {
  element.setAttribute("x", String(at.x))
  element.setAttribute("y", String(at.y))
  element.setAttribute("width", String(at.width))
  element.setAttribute("height", String(at.height))
}

// An arc draws itself from its start: a dash as long as the path, offset by what is left to draw.
// Finished arcs of earlier stages stay, dimmed.
function setArc(arc: SVGPathElement, progress: number, isCurrent: boolean): void {
  arc.classList.toggle("is-hidden", progress <= 0)
  arc.classList.toggle("is-on", progress > 0)
  arc.classList.toggle("is-idle", progress >= 1 && !isCurrent)
  arc.style.setProperty("stroke-dasharray", "1 1")
  arc.style.setProperty("stroke-dashoffset", String(Math.round((1 - Math.min(1, progress)) * 1000) / 1000))
}
