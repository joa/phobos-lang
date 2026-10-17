// The online softmax of Flash attention, one block of keys at a time: the block's scores are
// computed, the running row maximum `m` rises to cover them, the scores become exponentials, and
// the denominator `l` and the output `acc` are rescaled by `exp(m - mnew)` before the block's share
// is added. After the last block, `acc / l` is the softmax-weighted sum of every value. Schematic:
// sizes are illustrative, and no score is shown, only which state each statement touches.

import { ChartFigure } from "../../../lib/figure/chart-figure"
import { box, label, plate, wire, type Wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { StarPacket } from "../../../lib/figure/packet"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import type { Rect } from "../../inference/figure-geometry"
import { BLOCKS, onlineSoftmaxGeometry, type OnlineSoftmaxGeometry } from "./geometry"

const TRAVEL_MS = 700
const STEP_MS = 650

// The statements of one step, as the library writes them, and what each touches.
export const STATEMENTS = [
  { text: "s = dot_t(q, k) * scale", touches: ["scores"] },
  { text: "mnew = tmax(m, rowmax(s))", touches: ["m"] },
  { text: "s = exp(s - mnew)", touches: ["scores"] },
  { text: "l = l * corr + rowsum(s)", touches: ["l"] },
  { text: "acc = acc * corr + dot(s, v)", touches: ["acc"] },
] as const

type Part = "scores" | "m" | "l" | "acc"

export default class OnlineSoftmaxFigure extends ChartFigure {
  #geometry: OnlineSoftmaxGeometry | null = null
  #blocks: SVGRectElement[] = []
  #cells: Record<Part, SVGRectElement[]> = { scores: [], m: [], l: [], acc: [] }
  #keyWires: Wire[] = []
  #valueWires: Wire[] = []
  #keyPacket: StarPacket | null = null
  #valuePacket: StarPacket | null = null
  #step: SVGTextElement | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override build(layout: Layout): void {
    const geometry = onlineSoftmaxGeometry(layout)
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

    this.#keyWires = []
    this.#valueWires = []
    for (let i = 0; i < BLOCKS; i += 1) {
      this.#keyWires.push(wire(root, geometry.keyLine(i), { space: "global", className: "is-hidden" }))
      this.#valueWires.push(wire(root, geometry.valueLine(i), { space: "global", className: "is-hidden" }))
    }

    const global = svgEl("g", { class: "sp-global" }, root)
    this.#blocks = geometry.blocks.map(rect => box(global, { ...rect }))
    for (const at of geometry.blockLabels)
      this.#labels.push(label(global, at.text, { x: at.x, y: at.y, anchor: "middle", className: "is-small" }))

    const shared = svgEl("g", { class: "sp-shared" }, root)
    const cells = (rects: Rect[]) => rects.map(rect => svgEl("rect", { class: "chart-cell", ...rect }, shared))
    this.#cells = {
      scores: cells(geometry.scores.flat()),
      m: cells(geometry.m),
      l: cells(geometry.l),
      acc: cells(geometry.acc.flat()),
    }
    const { labels } = geometry
    for (const at of [labels.kv, labels.scores, labels.m, labels.l, labels.acc])
      this.#labels.push(label(root, at.text, { x: at.x, y: at.y }))
    this.#step = label(root, "", { x: labels.step.x, y: labels.step.y, anchor: "middle", className: "is-hot" })

    this.#keyPacket = new StarPacket(root, { space: "global", count: 10 })
    this.#valuePacket = new StarPacket(root, { space: "global", count: 10 })
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    this.motion.finishAll()
    const timeline = this.#timeline ?? this.#makeTimeline()
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (this.reducedMotion) return
    const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-box")
    this.motion.drawIn(lines)
    this.motion.enter(this.#labels, { delayMs: 240 })
    this.#run()
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run()
  }

  #run(): void {
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  #makeTimeline(): Timeline {
    const geometry = this.#geometry
    if (!geometry) return new Timeline([])
    const steps: TimelineStep[] = [{ durationMs: 0, apply: () => this.#reset() }]
    for (let block = 0; block < BLOCKS; block += 1) {
      steps.push({
        durationMs: TRAVEL_MS,
        apply: progress => {
          this.#showBlock(block)
          this.#keyPacket?.placeAtFraction(geometry.keyLine(block), progress < 1 ? progress : -1)
          this.#say(STATEMENTS[0].text, `Block ${block}: its keys meet the queries, and the scores land in s.`)
          this.#light(["scores"], progress >= 1 ? "is-on" : null)
        },
      })
      STATEMENTS.slice(1).forEach((statement, index) => {
        const isValue = statement.touches[0] === "acc"
        steps.push({
          durationMs: isValue ? TRAVEL_MS : STEP_MS,
          apply: progress => {
            this.#say(statement.text, READOUTS[index + 1](block))
            this.#light(statement.touches as readonly Part[], progress < 1 ? "is-hot" : "is-on")
            if (isValue) this.#valuePacket?.placeAtFraction(geometry.valueLine(block), progress < 1 ? progress : -1)
          },
        })
      })
      steps.push({
        durationMs: 0,
        apply: () => {
          this.#blocks[block]?.classList.remove("is-hot")
          this.#blocks[block]?.classList.add("is-on")
          this.#keyWires[block]?.group.classList.add("is-hidden")
          this.#valueWires[block]?.group.classList.add("is-hidden")
        },
      })
    }
    steps.push({
      durationMs: STEP_MS,
      apply: progress => {
        this.#say("acc = acc / l", "After the last block, acc / l is the softmax-weighted sum of every value.")
        this.#light(["acc", "l"], progress < 1 ? "is-hot" : "is-on")
      },
    })
    return new Timeline(steps)
  }

  #reset(): void {
    for (const block of this.#blocks) block.classList.remove("is-on", "is-hot")
    for (const part of Object.values(this.#cells)) for (const cell of part) cell.classList.remove("is-on", "is-hot")
    for (const w of [...this.#keyWires, ...this.#valueWires]) w.group.classList.add("is-hidden")
    this.#keyPacket?.hide()
    this.#valuePacket?.hide()
  }

  #showBlock(block: number): void {
    this.#blocks[block]?.classList.add("is-hot")
    this.#keyWires[block]?.group.classList.remove("is-hidden")
    this.#keyWires[block]?.group.classList.add("is-on")
    this.#valueWires[block]?.group.classList.remove("is-hidden")
  }

  // `parts` take `state`; the parts of the state already written keep `is-on`.
  #light(parts: readonly Part[], state: "is-on" | "is-hot" | null): void {
    for (const part of parts) {
      for (const cell of this.#cells[part]) {
        cell.classList.toggle("is-hot", state === "is-hot")
        cell.classList.toggle("is-on", state === "is-on")
      }
    }
  }

  #say(statement: string, readout: string): void {
    if (this.#step && this.#step.textContent !== statement) this.#step.textContent = statement
    this.setReadout(readout)
  }
}

const READOUTS: ((block: number) => string)[] = [
  () => "",
  block =>
    block === 0
      ? "The row maximum m starts at -1e30, so the first block's maximum becomes it."
      : `Block ${block}: m rises wherever the block holds a larger score; corr = exp(m - mnew) is 1 where it does not.`,
  () => "The scores become exponentials against the new maximum, so none overflows.",
  () => "The denominator l is rescaled by corr, then gains this block's row sums.",
  () => "acc is rescaled by corr, then gains the block's values weighted by s.",
]
