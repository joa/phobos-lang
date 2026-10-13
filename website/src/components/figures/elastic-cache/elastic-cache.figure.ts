// The elastic expert cache, from the replay under memory pressure: another process takes 2 GiB of the
// card, the expert cache gives it back in the steps the run logged, more misses go to the CPU threads,
// and when the process exits the cache grows again. Beside it, both engines' measured decode rates
// before, during and after. One run is a pure function of its elapsed time, so the static render and
// reduced motion show the end state, which keeps all three phases on the chart.

import data from "../../../data/elastic-cache.json"
import type { ElasticCacheData } from "../../../lib/bench/agent"
import { ChartFigure } from "../../../lib/figure/chart-figure"
import { box, label, plate, wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { clamp, easeInOut } from "../../../lib/figure/math"
import { StarPacket } from "../../../lib/figure/packet"
import type { Polyline } from "../../../lib/figure/polyline"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline } from "../../../lib/figure/timeline"
import type { Rect, TextAt } from "../../inference/figure-geometry"
import {
  cacheEndText,
  cacheText,
  elasticCacheGeometry,
  formatGiB,
  formatRate,
  type ElasticCacheGeometry,
} from "./geometry"

export const RUN = data as ElasticCacheData

const BEFORE_MS = 1800
const ARRIVE_MS = 700
const SHRINK_MS = 520
const SHRINK_GAP_MS = 260
const DURING_HOLD_MS = 1500
const LEAVE_MS = 600
const GROW_MS = 700
const AFTER_MS = 1600
const FLIGHT_MS = 900
const PACKETS = 10
const QUIET_EVERY_MS = 640
const BUSY_EVERY_MS = 170

export interface Schedule {
  arriveMs: number
  shrinks: { startMs: number; from: number; to: number }[]
  leaveMs: number
  growMs: number
  afterMs: number
  totalMs: number
}

export function schedule(run: ElasticCacheData = RUN): Schedule {
  const arriveMs = BEFORE_MS
  let atMs = arriveMs + ARRIVE_MS
  let from = run.cache.before
  const shrinks = run.cache.shrinks.map(to => {
    const step = { startMs: atMs, from, to }
    atMs += SHRINK_MS + SHRINK_GAP_MS
    from = to
    return step
  })
  const leaveMs = atMs + DURING_HOLD_MS
  const growMs = leaveMs + LEAVE_MS
  const afterMs = growMs + GROW_MS
  return { arriveMs, shrinks, leaveMs, growMs, afterMs, totalMs: afterMs + AFTER_MS }
}

// Experts a block at a moment of the run, eased between the sizes the run logged.
export function perBlockAt(timeMs: number, run: ElasticCacheData = RUN, plan = schedule(run)): number {
  let perBlock = run.cache.before
  for (const { startMs, from, to } of plan.shrinks) {
    if (timeMs >= startMs) perBlock = from + (to - from) * easeInOut(clamp((timeMs - startMs) / SHRINK_MS, 0, 1))
  }
  if (timeMs >= plan.growMs) {
    const from = run.cache.shrinks.at(-1) ?? run.cache.before
    perBlock = from + (run.cache.after - from) * easeInOut(clamp((timeMs - plan.growMs) / GROW_MS, 0, 1))
  }
  return perBlock
}

// When each miss leaves for the CPU and on which lane: sparse while the cache is large, dense while it
// is small. Deterministic, and every flight lands before the run ends.
export function emissions(plan = schedule()): { atMs: number; lane: number }[] {
  const out: { atMs: number; lane: number }[] = []
  const busyFrom = plan.shrinks[0]?.startMs ?? plan.arriveMs
  let atMs = 120
  while (atMs + FLIGHT_MS <= plan.totalMs) {
    out.push({ atMs, lane: out.length % 3 })
    atMs += atMs >= busyFrom && atMs < plan.growMs ? BUSY_EVERY_MS : QUIET_EVERY_MS
  }
  return out
}

export type Phase = "before" | "during" | "after"

export function phaseAt(timeMs: number, plan = schedule()): Phase {
  if (timeMs < plan.arriveMs) return "before"
  if (timeMs < plan.growMs) return "during"
  return "after"
}

export function readoutFor(phase: Phase, run: ElasticCacheData = RUN): string {
  const [phobos, other] = run.engines
  const rates = (index: number) =>
    `${phobos.label} decodes at ${formatRate(phobos.decode[index])} tokens/s, ${other.label} at ${formatRate(other.decode[index])}`
  const smallest = run.cache.shrinks.at(-1) ?? run.cache.before
  if (phase === "before")
    return `Before: the expert cache holds ${run.cache.before} experts a block. ${rates(0)}.`
  if (phase === "during")
    return `Another process takes ${formatGiB(run.squeezeMiB / 1024)}. The cache shrinks from ${run.cache.before} to ${smallest} experts a block and the CPU threads compute more of the misses. ${rates(1)}.`
  return `The process exits and the cache grows back to ${run.cache.after} experts a block. ${rates(2)}. The replay took ${phobos.wallSecs.toFixed(1)} s on ${phobos.label} and ${other.wallSecs.toFixed(1)} s on ${other.label}.`
}

export default class ElasticCacheFigure extends ChartFigure {
  #geometry: ElasticCacheGeometry | null = null
  #plan = schedule()
  #emissions = emissions(this.#plan)
  #cells: SVGRectElement[][] = []
  #cacheLabel: SVGTextElement | null = null
  #other: SVGRectElement | null = null
  #otherLabel: SVGTextElement | null = null
  #cpu: SVGRectElement | null = null
  #series: { path: SVGPathElement; lengthUnits: number; vertexUnits: number[] }[] = []
  #values: SVGTextElement[][] = []
  #packets: StarPacket[] = []
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override build(layout: Layout): void {
    const geometry = elasticCacheGeometry(layout, RUN)
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

    for (const lane of geometry.lanes) wire(root, lane, { space: "reg" })

    const card = svgEl("g", { class: "sp-global" }, root)
    box(card, { ...geometry.bar, className: "is-outline" })
    this.#addLabel(card, geometry.vramLabel, "is-small")
    box(card, { ...geometry.rest, className: "is-idle" })
    this.#addLabel(card, geometry.restLabel, "is-small")
    this.#cells = geometry.columns.map(column => column.map(cell => rect(card, cell, "chart-cell is-on")))
    this.#cacheLabel = this.#addLabel(card, geometry.cacheLabel, "is-small is-on")

    const other = svgEl("g", { class: "sp-cost" }, root)
    this.#other = box(other, { ...geometry.other, className: "is-warn" })
    this.#otherLabel = this.#addLabel(other, geometry.otherLabel, "is-small is-warn")

    const cpu = svgEl("g", { class: "sp-reg" }, root)
    this.#cpu = box(cpu, { ...geometry.cpu.box, className: "is-on" })
    this.#addLabel(cpu, geometry.cpu.label)
    this.#addLabel(cpu, geometry.cpu.note, "is-small")

    this.#addLabel(root, geometry.wall.title, "is-small")
    geometry.wall.rows.forEach((row, index) => this.#addLabel(root, row, `is-small ec-text ${seriesClass(index)}`))

    const spark = geometry.spark
    this.#addLabel(root, spark.title, "is-small")
    spark.legend.forEach(({ swatch, label: at }, index) => {
      svgEl("path", { class: `ec-series ${seriesClass(index)}`, d: swatch.toPathData() }, root)
      this.#addLabel(root, at, `is-small ec-text ${seriesClass(index)}`)
    })
    const band = svgEl("g", { class: "sp-cost" }, root)
    box(band, { ...spark.band, className: "is-warn ec-band" })
    for (const at of spark.phases) this.#addLabel(root, at, "is-small")
    svgEl("path", { class: "chart-wire", d: spark.baseline.toPathData() }, root)
    this.#series = spark.series.map((series, index) => {
      const lengthUnits = series.line.lengthUnits
      const path = svgEl(
        "path",
        {
          class: `ec-series is-line ${seriesClass(index)}`,
          d: series.line.toPathData(),
          "stroke-dasharray": `${round(lengthUnits)} ${round(lengthUnits)}`,
        },
        root
      )
      return { path, lengthUnits, vertexUnits: vertexDistances(series.line) }
    })
    this.#values = spark.series.map((series, index) =>
      series.values.map(at => this.#addLabel(root, at, `is-small ec-text ${seriesClass(index)}`))
    )

    this.#packets = Array.from({ length: PACKETS }, () => new StarPacket(root, { space: "reg", count: 6 }))
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
    const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-box:not(.is-warn)")
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
    const totalMs = this.#plan.totalMs
    return new Timeline([{ durationMs: totalMs, apply: progress => this.#show(progress * totalMs) }])
  }

  // Everything the figure shows at a moment of the run.
  #show(timeMs: number): void {
    const plan = this.#plan
    const isEnd = timeMs >= plan.totalMs

    const perBlock = perBlockAt(timeMs, RUN, plan)
    this.#cells.forEach((column, index) => {
      const shown = clamp(perBlock - index, 0, 1)
      const cell = this.#geometry?.columns[index][0]
      for (const element of column) {
        element.setAttribute("width", String(round((cell?.width ?? 0) * shown)))
        element.classList.toggle("is-hidden", shown <= 0)
      }
    })
    if (this.#cacheLabel) this.#cacheLabel.textContent = isEnd ? cacheEndText(RUN) : cacheText(Math.round(perBlock))

    this.#showOther(timeMs)

    const phase = phaseAt(timeMs, plan)
    this.#cpu?.classList.toggle("is-hot", phase === "during" && !isEnd)
    this.#showSeries(timeMs)
    this.#showPackets(timeMs)
    this.setReadout(readoutFor(phase))
  }

  // Grows in from the card's end, holds, then fades to a dashed outline once the process has exited.
  #showOther(timeMs: number): void {
    const geometry = this.#geometry
    const other = this.#other
    if (!geometry || !other) return
    const plan = this.#plan
    const arrived = easeInOut(clamp((timeMs - plan.arriveMs) / ARRIVE_MS, 0, 1))
    const gone = timeMs >= plan.leaveMs
    const width = gone ? geometry.other.width : geometry.other.width * arrived
    other.setAttribute("x", String(round(geometry.other.x + geometry.other.width - width)))
    other.setAttribute("width", String(round(width)))
    other.classList.toggle("is-hidden", !gone && arrived <= 0)
    other.classList.toggle("is-idle", gone)
    this.#otherLabel?.classList.toggle("is-hidden", !gone && arrived <= 0)
    this.#otherLabel?.classList.toggle("is-idle", gone)
  }

  // Each engine's line draws along with the run: the before level, the drop while the cache shrinks,
  // the during level, the rise while it grows back, the after level.
  #showSeries(timeMs: number): void {
    const plan = this.#plan
    const lastShrinkEndMs = (plan.shrinks.at(-1)?.startMs ?? plan.arriveMs) + SHRINK_MS
    // Time at which the line reaches each vertex.
    const vertexMs = [0, plan.arriveMs, lastShrinkEndMs, plan.growMs, plan.afterMs, plan.totalMs]
    this.#series.forEach(({ path, lengthUnits, vertexUnits }, index) => {
      const drawnUnits = interpolate(timeMs, vertexMs, vertexUnits)
      path.setAttribute("stroke-dashoffset", String(round(lengthUnits - drawnUnits)))
      this.#values[index]?.forEach((value, phase) => {
        value.classList.toggle("is-hidden", timeMs < vertexMs[phase * 2])
      })
    })
  }

  #showPackets(timeMs: number): void {
    const geometry = this.#geometry
    if (!geometry) return
    this.#packets.forEach((packet, slot) => {
      let placed = false
      for (let index = slot; index < this.#emissions.length; index += PACKETS) {
        const { atMs, lane } = this.#emissions[index]
        const fraction = (timeMs - atMs) / FLIGHT_MS
        if (fraction > 0 && fraction < 1) {
          packet.placeAtFraction(geometry.lanes[lane], fraction)
          placed = true
          break
        }
      }
      if (!placed) packet.hide()
    })
  }

  #addLabel(parent: Element, at: TextAt, className?: string): SVGTextElement {
    const text = label(parent, at.text, { x: at.x, y: at.y, anchor: at.anchor ?? "start", className })
    this.#labels.push(text)
    return text
  }
}

function seriesClass(index: number): string {
  return index === 0 ? "is-phobos" : "is-other"
}

function rect(parent: Element, at: Rect, className: string): SVGRectElement {
  return svgEl("rect", { class: className, x: at.x, y: at.y, width: at.width, height: at.height }, parent)
}

function vertexDistances(line: Polyline): number[] {
  const distances = [0]
  for (let index = 1; index < line.points.length; index += 1) {
    const [a, b] = [line.points[index - 1], line.points[index]]
    distances.push(distances[index - 1] + Math.hypot(b.x - a.x, b.y - a.y))
  }
  return distances
}

// Piecewise linear: times[i] maps to values[i], clamped at both ends.
export function interpolate(time: number, times: number[], values: number[]): number {
  if (time <= times[0]) return values[0]
  for (let index = 1; index < times.length; index += 1) {
    if (time <= times[index]) {
      const span = times[index] - times[index - 1]
      const fraction = span > 0 ? (time - times[index - 1]) / span : 1
      return values[index - 1] + (values[index] - values[index - 1]) * fraction
    }
  }
  return values[values.length - 1]
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
