// The races page's kernel: 64 threads in two warps each store their own element of a shared tile t
// and load their mirror's. Warps run at their own pace, so without a barrier warp 0 can load what
// warp 1 has not stored yet; with one, warp 0 waits. Shared memory is drawn in --nebula, the threads'
// registers in --star, a stale load and the wait at the barrier as cost. Schematic: which warp runs
// ahead, and by how much, is illustrative; the pair it ends on is the one the race check reports.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { StarPacket } from "../../../lib/figure/packet"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import { mirrorOf, raceMirrorGeometry, THREADS, WARP, type RaceMirrorGeometry, type Rect, type TextAt } from "./geometry"

const PACKET_MS = 520
const ARRIVE_MS = 160
const WAIT_MS = 620
const PAIR_MS = 400

type Order = "none" | "barrier"

const ORDERS: readonly Order[] = ["none", "barrier"]

export default class RaceMirrorFigure extends ChartFigure {
  #geometry: RaceMirrorGeometry | null = null
  #threads: SVGRectElement[] = []
  #tile: SVGRectElement[] = []
  #reads: SVGRectElement[] = []
  #writes: SVGPathElement[] = []
  #loads: SVGPathElement[] = []
  #barrier: SVGPathElement[] = []
  #barrierLabel: SVGTextElement | null = null
  #writerLabel: SVGTextElement | null = null
  #readerLabel: SVGTextElement | null = null
  #writePackets: StarPacket[] = []
  #loadPackets: StarPacket[] = []
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ order: "none" }, { order: "barrier" }]
  }

  get #order(): Order {
    const order = String(this.state.order ?? "none")
    return (ORDERS as readonly string[]).includes(order) ? (order as Order) : "none"
  }

  override build(layout: Layout): void {
    const geometry = raceMirrorGeometry(layout)
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
    for (const at of geometry.rowLabels) this.#addLabel(root, at)
    for (const at of geometry.warpLabels) this.#addLabel(root, at, "is-small")

    const writeGroup = svgEl("g", { class: "sp-reg" }, root)
    this.#writes = geometry.writes.map(line => svgEl("path", { class: "chart-wire", d: line.toPathData() }, writeGroup))
    const loadGroup = svgEl("g", { class: "sp-shared" }, root)
    this.#loads = geometry.loads.map(line => svgEl("path", { class: "chart-wire", d: line.toPathData() }, loadGroup))

    const barrierGroup = svgEl("g", { class: "sp-cost" }, root)
    this.#barrier = geometry.barrier.map(({ from, to }) =>
      svgEl("path", { class: "chart-wire", d: `M${from.x} ${from.y}L${to.x} ${to.y}` }, barrierGroup)
    )
    this.#barrierLabel = this.#addLabel(root, geometry.barrierLabel, "is-small")

    const threadGroup = svgEl("g", { class: "sp-reg" }, root)
    this.#threads = geometry.threads.map(at => rect(threadGroup, at, "chart-cell is-on"))
    const tileGroup = svgEl("g", { class: "sp-shared" }, root)
    this.#tile = geometry.tile.map(at => rect(tileGroup, at, "chart-cell"))
    const readGroup = svgEl("g", { class: "sp-reg" }, root)
    this.#reads = geometry.reads.map(at => rect(readGroup, at, "chart-box"))

    this.#writerLabel = this.#addLabel(root, geometry.writerLabel, "is-small")
    this.#readerLabel = this.#addLabel(root, geometry.readerLabel, "is-small")

    const packet = (space: "reg" | "shared", seed: number) =>
      new StarPacket(root, { space, count: 3, trailGapUnits: 2, seed })
    this.#writePackets = Array.from({ length: THREADS }, (_, thread) => packet("reg", 7000 + thread))
    this.#loadPackets = Array.from({ length: THREADS }, (_, thread) => packet("shared", 8000 + thread))
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
    if (!this.reducedMotion) {
      const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-box")
      this.motion.drawIn(lines)
      this.motion.enter(this.#labels, { delayMs: 240 })
    }
    super.onFirstVisible()
  }

  override onChange(group: string, _value: string): void {
    if (group === "order") this.#run()
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run()
  }

  #run(): void {
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  // Warp 0 runs ahead of warp 1. Without a barrier it loads the upper half before warp 1 stores it,
  // and the figure ends on the pair the race check names; with one, warp 0 waits for warp 1.
  #makeTimeline(): Timeline {
    const steps: TimelineStep[] = [{ durationMs: 0, apply: () => this.#reset() }]
    const warp0 = range(0, WARP)
    const warp1 = range(WARP, THREADS)
    steps.push(
      this.#say("Each thread stores its own element: t[0, ctx.thread] = X[0, ctx.thread]."),
      this.#writeStep(warp0)
    )
    if (this.#order === "none") {
      steps.push(
        this.#say("Warp 0 runs ahead and loads t[0, 63 - ctx.thread]: elements 63 down to 32, not stored yet."),
        this.#loadStep(warp0),
        this.#say("Warp 1 stores its elements, too late for warp 0."),
        this.#writeStep(warp1),
        this.#loadStep(warp1),
        {
          durationMs: PAIR_MS,
          apply: progress => {
            if (progress < 1) return
            this.#showPair(true)
            this.setReadout(
              "Nothing orders thread 63's store before thread 0's load. The race check reports that pair."
            )
          },
        }
      )
    } else {
      steps.push(
        {
          durationMs: WAIT_MS,
          apply: progress => {
            this.#barrier[0]?.classList.toggle("is-warn", progress > 0)
            this.#barrierLabel?.classList.toggle("is-warn", progress > 0)
            if (progress === 0) this.setReadout("Warp 0 reaches barrier() and waits there.")
          },
        },
        this.#say("Warp 1 stores its elements and reaches the barrier too."),
        this.#writeStep(warp1),
        {
          durationMs: 0,
          apply: () => {
            for (const segment of this.#barrier) segment.classList.remove("is-warn")
            this.#barrierLabel?.classList.remove("is-warn")
          },
        },
        this.#say("Every element is stored before any thread loads one: no race."),
        this.#loadStep(range(0, THREADS))
      )
    }
    return new Timeline(steps)
  }

  #say(text: string): TimelineStep {
    return { durationMs: 0, apply: () => this.setReadout(text) }
  }

  // The threads store at once, one instruction per warp: each packet runs down its own wire and lights
  // the element it reaches.
  #writeStep(threads: number[]): TimelineStep {
    return {
      durationMs: PACKET_MS + ARRIVE_MS,
      apply: progress => {
        const fraction = (progress * (PACKET_MS + ARRIVE_MS)) / PACKET_MS
        for (const thread of threads) {
          const line = this.#geometry?.writes[thread]
          if (line) this.#writePackets[thread]?.placeAtFraction(line, fraction)
          this.#writes[thread]?.classList.toggle("is-on", fraction > 0 && fraction < 1)
          this.#tile[thread]?.classList.toggle("is-on", fraction >= 1)
        }
      },
    }
  }

  // Each thread loads its mirror's element. One stored already arrives lit; one not stored yet
  // arrives as cost, a stale value.
  #loadStep(threads: number[]): TimelineStep {
    return {
      durationMs: PACKET_MS + ARRIVE_MS,
      apply: progress => {
        const fraction = (progress * (PACKET_MS + ARRIVE_MS)) / PACKET_MS
        for (const thread of threads) {
          const stale = !this.#tile[mirrorOf(thread)]?.classList.contains("is-on")
          const line = this.#geometry?.loads[thread]
          if (line && !stale) this.#loadPackets[thread]?.placeAtFraction(line, fraction)
          // A wire lights only while its value travels, so 64 of them never stand lit at once.
          const inFlight = fraction > 0 && fraction < 1
          const wire = this.#loads[thread]
          wire?.classList.toggle("is-on", !stale && inFlight)
          wire?.classList.toggle("is-warn", stale && inFlight)
          const read = this.#reads[thread]
          read?.classList.toggle("is-on", !stale && fraction >= 1)
          read?.classList.toggle("is-warn", stale && fraction >= 1)
        }
      },
    }
  }

  #showPair(shown: boolean): void {
    const writer = THREADS - 1
    this.#tile[writer]?.classList.toggle("is-hot", shown)
    this.#threads[writer]?.classList.toggle("is-hot", shown)
    this.#reads[0]?.classList.toggle("is-hot", shown)
    this.#writerLabel?.classList.toggle("is-hidden", !shown)
    this.#readerLabel?.classList.toggle("is-hidden", !shown)
    this.#writerLabel?.classList.toggle("is-warn", shown)
    this.#readerLabel?.classList.toggle("is-warn", shown)
    this.#loads[0]?.classList.toggle("is-warn", shown)
    this.#writes[writer]?.classList.toggle("is-on", shown)
  }

  #reset(): void {
    const barrierShown = this.#order === "barrier"
    for (const cell of this.#tile) cell.classList.remove("is-on", "is-hot")
    for (const cell of this.#threads) cell.classList.remove("is-hot")
    for (const read of this.#reads) read.classList.remove("is-on", "is-warn", "is-hot")
    for (const wire of [...this.#writes, ...this.#loads]) wire.classList.remove("is-on", "is-warn")
    for (const packet of [...this.#writePackets, ...this.#loadPackets]) packet.hide()
    for (const segment of this.#barrier) {
      segment.classList.remove("is-warn")
      segment.classList.toggle("is-hidden", !barrierShown)
    }
    this.#barrierLabel?.classList.remove("is-warn")
    this.#barrierLabel?.classList.toggle("is-hidden", !barrierShown)
    this.#showPair(false)
  }

  #addLabel(parent: Element, at: TextAt, className?: string): SVGTextElement {
    const text = label(parent, at.text, { x: at.x, y: at.y, anchor: at.anchor ?? "start", className })
    this.#labels.push(text)
    return text
  }
}

function range(from: number, to: number): number[] {
  return Array.from({ length: to - from }, (_, index) => from + index)
}

function rect(parent: Element, at: Rect, className: string): SVGRectElement {
  return svgEl("rect", { class: className, x: at.x, y: at.y, width: at.width, height: at.height }, parent)
}
