// Rotate before rounding. A 16-wide toy row with one outlier is rounded to ternary with the best scale,
// once as it is and once after a randomized Hadamard rotation. Every number in the readout is computed
// from the drawn row (model.ts). Schematic: the row is a toy, the model's blocks are 1,024 wide.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { clamp, easeOut } from "../../../lib/figure/math"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import type { Rect, TextAt } from "../../inference/figure-geometry"
import { hadamardGeometry, type HadamardGeometry } from "./geometry"
import { FITS, LANE_COUNT, OUTLIER_LANE, ROW, STAGE_COUNT, readoutFor, type Rotation } from "./model"

const SIGNS_MS = 350
const STAGE_MS = 300
const ARRIVE_MS = 450
const ROUND_MS = 600

export default class HadamardRotationFigure extends ChartFigure {
  #geometry: HadamardGeometry | null = null
  #signs: SVGTextElement[] = []
  #stages: SVGPathElement[] = []
  #rightBars: SVGRectElement[] = []
  #ticks: SVGPathElement[] = []
  #errors: SVGPathElement[] = []
  #levels: SVGPathElement[] = []
  #codeLabels: SVGTextElement[] = []
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ rotation: "off" }, { rotation: "on" }]
  }

  get #rotation(): Rotation {
    return this.state.rotation === "off" ? "off" : "on"
  }

  override build(layout: Layout): void {
    const geometry = hadamardGeometry(layout)
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
    const { labels, network, laneY } = geometry
    const axisTop = geometry.top - 6
    const axisBottom = geometry.bottom + 6

    this.#addLabel(root, labels.left)
    this.#addLabel(root, labels.network)
    this.#addLabel(root, labels.right)
    this.#addLabel(root, labels.signs, "is-small")
    this.#addLabel(root, labels.stages, "is-small")
    for (const note of labels.note) this.#addLabel(root, note, "is-small")

    const left = svgEl("g", { class: "sp-global" }, root)
    svgEl("path", { class: "chart-wire", d: `M${geometry.leftAxis} ${axisTop}V${axisBottom}` }, left)
    ROW.weights.forEach((weight, lane) => {
      const isOutlier = lane === OUTLIER_LANE
      rect(
        left,
        geometry.bar(geometry.leftAxis, lane, weight),
        isOutlier ? "chart-box sp-cost is-warn" : "chart-box is-on"
      )
    })

    const net = svgEl("g", { class: "sp-reg" }, root)
    for (const y of laneY) svgEl("path", { class: "chart-wire", d: `M${network.left} ${y}H${network.right}` }, net)
    this.#signs = ROW.signs.map((sign, lane) =>
      label(net, sign < 0 ? "-" : "+", { x: network.signX, y: laneY[lane], anchor: "middle", className: "is-small" })
    )
    this.#stages = Array.from({ length: STAGE_COUNT }, (_, stage) =>
      svgEl("path", { class: "chart-wire", d: geometry.butterfly(stage) }, net)
    )

    const right = svgEl("g", { class: "sp-global" }, root)
    this.#levels = [-1, 0, 1].map(code =>
      svgEl("path", { class: code === 0 ? "chart-wire" : "chart-wire hr-level" }, right)
    )
    this.#codeLabels = ["-1", "0", "+1"].map(text =>
      label(right, text, { x: 0, y: geometry.codeLabelY, anchor: "middle", className: "is-small" })
    )
    this.#rightBars = ROW.weights.map(() => rect(right, { x: 0, y: 0, width: 0, height: 0 }, "chart-box is-on"))
    this.#errors = ROW.weights.map(() => svgEl("path", { class: "chart-wire is-warn" }, right))
    this.#ticks = ROW.weights.map(() => svgEl("path", { class: "chart-wire hr-tick sp-reg" }, right))
    // The level lines span the whole column, so they are placed now and moved per fit.
    for (const level of this.#levels) level.setAttribute("d", `M0 ${axisTop}V${axisBottom}`)
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    this.motion.finishAll()
    const timeline = this.#timeline ?? this.#makeTimeline(this.#rotation)
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) {
      const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire")
      this.motion.drawIn(lines)
      this.motion.enter(this.#labels, { delayMs: 240 })
    }
    super.onFirstVisible()
  }

  override onChange(group: string, value: string): void {
    if (group === "rotation") this.#run(value === "off" ? "off" : "on")
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run(this.#rotation)
  }

  #run(rotation: Rotation): void {
    this.#timeline = this.#makeTimeline(rotation)
    this.play()
  }

  #makeTimeline(rotation: Rotation): Timeline {
    const steps: TimelineStep[] = [{ durationMs: 0, apply: () => this.#reset(rotation) }]
    if (rotation === "on") {
      steps.push({
        durationMs: SIGNS_MS,
        apply: progress => {
          for (const sign of this.#signs) {
            sign.classList.toggle("is-hot", progress > 0 && progress < 1)
            sign.classList.toggle("is-on", progress >= 1)
          }
        },
      })
      this.#stages.forEach((stage, index) => {
        steps.push({
          durationMs: STAGE_MS,
          apply: progress => {
            stage.classList.toggle("is-hidden", progress <= 0)
            stage.classList.toggle("is-on", progress > 0)
            this.#stages[index - 1]?.classList.toggle("is-idle", progress > 0)
          },
        })
      })
      steps.push({ durationMs: ARRIVE_MS, apply: progress => this.#placeBars("on", easeOut(progress)) })
    }
    steps.push({ durationMs: ROUND_MS, apply: progress => this.#showRounding(rotation, progress) })
    steps.push({
      durationMs: 0,
      apply: () => this.#stages[this.#stages.length - 1]?.classList.toggle("is-idle", rotation === "on"),
    })
    return new Timeline(steps)
  }

  #reset(rotation: Rotation): void {
    for (const sign of this.#signs) {
      sign.classList.remove("is-hot", "is-on")
      sign.classList.toggle("is-idle", rotation === "off")
    }
    for (const stage of this.#stages) {
      stage.classList.remove("is-on", "is-idle")
      stage.classList.toggle("is-hidden", true)
    }
    // As is, the right column shows the raw row at once; rotated, the bars grow from the axis.
    this.#placeBars(rotation, rotation === "off" ? 1 : 0)
    this.#showRounding(rotation, 0)
    this.setReadout(readoutFor(rotation))
  }

  #placeBars(rotation: Rotation, progress: number): void {
    const geometry = this.#geometry
    if (!geometry) return
    const { values } = FITS[rotation]
    this.#rightBars.forEach((bar, lane) => {
      placeRect(bar, geometry.bar(geometry.rightAxis, lane, values[lane] * progress))
      bar.classList.toggle("sp-cost", rotation === "off" && lane === OUTLIER_LANE)
    })
  }

  // The levels at -scale, 0 and +scale, a tick at each value's rounded position and a cost wire for
  // the error: ticks from 40% of the step, errors from 60%.
  #showRounding(rotation: Rotation, progress: number): void {
    const geometry = this.#geometry
    if (!geometry) return
    const { values, fit } = FITS[rotation]
    const xOf = (value: number) => geometry.rightAxis + value * geometry.unitsPerValue
    const levelOpacity = clamp(progress / 0.4, 0, 1)
    ;[-1, 0, 1].forEach((code, index) => {
      const x = round(xOf(code * fit.scale))
      this.#levels[index].setAttribute("transform", `translate(${x} 0)`)
      this.#levels[index].classList.toggle("is-hidden", code !== 0 && levelOpacity === 0)
      this.#levels[index].classList.toggle("is-on", code === 0 || levelOpacity > 0)
      this.#codeLabels[index].setAttribute("x", String(x))
      this.#codeLabels[index].classList.toggle("is-hidden", levelOpacity === 0)
    })
    for (let lane = 0; lane < LANE_COUNT; lane += 1) {
      const y = geometry.laneY[lane]
      const rounded = xOf(fit.codes[lane] * fit.scale)
      const value = xOf(values[lane])
      this.#ticks[lane].setAttribute("d", `M${round(rounded)} ${y - 6}V${y + 6}`)
      this.#ticks[lane].classList.toggle("is-hidden", progress < 0.4)
      this.#ticks[lane].classList.toggle("is-on", progress >= 0.4)
      this.#errors[lane].setAttribute("d", `M${round(value)} ${y}H${round(rounded)}`)
      this.#errors[lane].classList.toggle("is-hidden", progress < 0.6 || Math.abs(value - rounded) < 0.5)
    }
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
  element.setAttribute("x", String(round(at.x)))
  element.setAttribute("y", String(round(at.y)))
  element.setAttribute("width", String(round(at.width)))
  element.setAttribute("height", String(round(at.height)))
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
