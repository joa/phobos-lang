// Sample figure: one CTA copies a tile of a global tensor into shared memory, spread over its threads.
// Global memory is drawn in --brand, registers in --star, shared memory in --nebula, the barrier wait
// as cost. Schematic: the sizes are illustrative (see geometry.ts), not a trace of the compiler.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { box, glow, label, plate, wire, type Space, type Wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { sweepWire } from "../../../lib/figure/motion"
import { StarPacket } from "../../../lib/figure/packet"
import type { Polyline } from "../../../lib/figure/polyline"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import { ctaCopyGeometry, sliceOf, TILE, TILE_BYTES, type CtaCopyGeometry, type Rect, type TextAt } from "./geometry"

const PACKET_SPEED_UNITS_PER_MS = 0.26
const PACKET_STAGGER_MS = 120
const TILE_LIGHT_MS = 560
const BARRIER_HOLD_MS = 260
const BARRIER_RELEASE_MS = 360
const ARRIVAL_GLOW_MS = 300
const MOVE_MS = 480

type Phase = "tile" | "load" | "store"
type Segment = Phase | "move"

const PHASES: readonly Phase[] = ["tile", "load", "store"]

export default class CtaCopyFigure extends ChartFigure {
  #geometry: CtaCopyGeometry | null = null
  #globalCells: SVGRectElement[][] = []
  #tileOutline: SVGRectElement | null = null
  #lead: SVGPathElement | null = null
  #leadNode: SVGCircleElement | null = null
  #wireLoad: Wire | null = null
  #wireStore: Wire | null = null
  #sweepLoad: SVGPathElement | null = null
  #sweepStore: SVGPathElement | null = null
  #regCells: SVGRectElement[][] = []
  #regGlows: SVGCircleElement[] = []
  #sharedCells: SVGRectElement[][] = []
  #sharedGlows: SVGCircleElement[] = []
  #sharedLabel: SVGTextElement | null = null
  #barrier: SVGPathElement | null = null
  #barrierLabel: SVGTextElement | null = null
  #loadPackets: StarPacket[] = []
  #storePackets: StarPacket[] = []
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ phase: "tile" }, { phase: "load" }, { phase: "store" }, { kt: "1" }]
  }

  get #phase(): Phase {
    const phase = String(this.state.phase ?? "store")
    return (PHASES as readonly string[]).includes(phase) ? (phase as Phase) : "store"
  }

  get #kStep(): number {
    return Number(this.state.kt ?? 0)
  }

  override build(layout: Layout): void {
    const geometry = ctaCopyGeometry(layout)
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

    const globalGroup = svgEl("g", { class: "sp-global" }, root)
    this.#addLabel(globalGroup, geometry.global.label)
    this.#globalCells = geometry.global.cells.map(row => row.map(cell => rect(globalGroup, cell, "chart-cell")))

    const lead = geometry.lead(this.#kStep)
    this.#lead = svgEl("path", { class: "chart-wire is-on sp-global", d: lead.toPathData() }, root)
    this.#leadNode = svgEl(
      "circle",
      { class: "chart-node is-on sp-global", cx: lead.start.x, cy: lead.start.y, r: 2 },
      root
    )
    const tile = geometry.tileRect(this.#kStep)
    this.#tileOutline = box(root, { ...tile, space: "global", className: "is-outline" })

    this.#wireLoad = wire(root, geometry.wireLoad, { space: "global" })
    this.#sweepLoad = sweepWire(root, geometry.wireLoad.toPathData(), "global")
    this.#wireStore = wire(root, geometry.wireStore, { space: "shared" })
    this.#sweepStore = sweepWire(root, geometry.wireStore.toPathData(), "reg")

    const regGroup = svgEl("g", { class: "sp-reg" }, root)
    this.#addLabel(regGroup, geometry.registers.label)
    for (const threadLabel of geometry.registers.threadLabels) this.#addLabel(regGroup, threadLabel, "is-small")
    this.#addLabel(regGroup, geometry.registers.note, "is-small")
    this.#regGlows = geometry.registers.cells.map(row => rowGlow(regGroup, row, "reg"))
    for (const thread of geometry.registers.threads) rect(regGroup, thread, "chart-box")
    this.#regCells = geometry.registers.cells.map(row => row.map(cell => rect(regGroup, cell, "chart-cell")))

    const sharedGroup = svgEl("g", { class: "sp-shared" }, root)
    this.#sharedLabel = this.#addLabel(sharedGroup, geometry.shared.label)
    this.#sharedGlows = geometry.shared.cells.map(row => rowGlow(sharedGroup, row, "shared"))
    this.#sharedCells = geometry.shared.cells.map(row => row.map(cell => rect(sharedGroup, cell, "chart-cell")))

    const { from, to } = geometry.barrier
    this.#barrier = svgEl("path", { class: "chart-wire", d: `M${from.x} ${from.y}H${to.x}` }, root)
    this.#barrierLabel = this.#addLabel(root, geometry.barrier.label, "is-small")

    this.#loadPackets = Array.from({ length: TILE }, () => new StarPacket(root, { space: "global" }))
    this.#storePackets = Array.from({ length: TILE }, () => new StarPacket(root, { space: "reg" }))
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    this.motion.finishAll()
    const timeline = this.#timeline ?? this.#makeTimeline(this.#segmentsUpTo(this.#phase))
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) {
      const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire, .chart-box")
      this.motion.drawIn(lines)
      this.motion.enter(this.#labels, { delayMs: 240 })
    }
    super.onFirstVisible()
  }

  override onChange(group: string, value: string): void {
    if (group === "phase") this.#run(this.#segmentsUpTo(value as Phase))
    if (group === "kt") {
      const kStep = Number(value)
      this.setState({ phase: "store" })
      // The tile outline glides to its new place; the cells and the lead follow in the timeline.
      this.motion.glide(this.#tileOutline ? [this.#tileOutline] : [], () => this.#placeOutline(kStep))
      this.#run(["move", "tile", "load", "store"])
    }
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run(this.#segmentsUpTo(this.#phase))
  }

  #run(segments: Segment[]): void {
    this.#timeline = this.#makeTimeline(segments)
    this.play()
  }

  #segmentsUpTo(phase: Phase): Segment[] {
    return PHASES.slice(0, PHASES.indexOf(phase) + 1)
  }

  // A timeline replays every segment before the last at once (0 ms), then animates the last, unless
  // the reader moved the tile: then each segment plays in turn.
  #makeTimeline(segments: Segment[]): Timeline {
    const playsAll = segments[0] === "move"
    const steps: TimelineStep[] = [{ durationMs: 0, apply: () => this.#reset() }]
    segments.forEach((segment, index) => {
      const animated = playsAll || index === segments.length - 1
      steps.push(...this.#stepsFor(segment, animated))
    })
    return new Timeline(steps)
  }

  #stepsFor(segment: Segment, animated: boolean): TimelineStep[] {
    const geometry = this.#geometry
    if (!geometry) return []
    const scale = (durationMs: number) => (animated ? durationMs : 0)
    const { kt, rows, cols } = sliceOf(this.#kStep)
    switch (segment) {
      case "move":
        return [
          {
            durationMs: scale(MOVE_MS),
            apply: progress => {
              this.#lead?.classList.toggle("is-hidden", progress < 1)
              this.#leadNode?.classList.toggle("is-hidden", progress < 1)
              if (progress === 1) this.#placeLead()
            },
          },
        ]
      case "tile":
        return [
          {
            durationMs: 0,
            apply: () => {
              this.#placeOutline(this.#kStep)
              this.#placeLead()
              this.setReadout(
                `Block (1, 0) at kt = ${kt} owns A[${rows[0]}:${rows[1]}, ${cols[0]}:${cols[1]}], an 8 x 8 tile of global memory.`
              )
            },
          },
          { durationMs: scale(TILE_LIGHT_MS), apply: progress => this.#lightTile(progress) },
        ]
      case "load":
        return [
          {
            durationMs: 0,
            apply: () => {
              this.#sweepLoad?.classList.add("is-on")
              this.#wireLoad?.group.classList.add("is-on")
              this.setReadout("32 threads load the tile into registers, 2 adjacent f32 each: thread t takes row t / 4.")
            },
          },
          this.#packetStep(geometry.wireLoad, this.#loadPackets, this.#regCells, this.#regGlows, scale),
          { durationMs: 0, apply: () => this.#sweepLoad?.classList.remove("is-on") },
        ]
      case "store":
        return [
          {
            durationMs: 0,
            apply: () => {
              this.#sweepStore?.classList.add("is-on")
              this.#wireStore?.group.classList.add("is-on")
              this.#setBarrier("wait")
              this.setReadout("Each thread stores its 2 values into shared tile a. The barrier waits for all 32.")
            },
          },
          this.#packetStep(geometry.wireStore, this.#storePackets, this.#sharedCells, this.#sharedGlows, scale),
          { durationMs: scale(BARRIER_HOLD_MS), apply: () => {} },
          {
            durationMs: scale(BARRIER_RELEASE_MS),
            apply: progress => {
              if (progress < 1) return
              this.#sweepStore?.classList.remove("is-on")
              this.#setBarrier("pass")
              this.#sharedLabel?.classList.add("is-on")
              this.setReadout(
                `After the barrier every thread can read all 64 values of a: ${TILE_BYTES} bytes of shared memory.`
              )
            },
          },
        ]
    }
  }

  // Eight packets, one per tile row, leave one after another at constant speed. Each lights its row
  // on arrival with a glow that fades back over 300 ms.
  #packetStep(
    line: Polyline,
    packets: StarPacket[],
    cells: SVGRectElement[][],
    glows: SVGCircleElement[],
    scale: (durationMs: number) => number
  ): TimelineStep {
    const travelMs = line.lengthUnits / PACKET_SPEED_UNITS_PER_MS
    const totalMs = travelMs + (packets.length - 1) * PACKET_STAGGER_MS
    return {
      durationMs: scale(totalMs + ARRIVAL_GLOW_MS),
      apply: progress => {
        const elapsedMs = progress * (totalMs + ARRIVAL_GLOW_MS)
        packets.forEach((packet, row) => {
          const sinceSendMs = elapsedMs - row * PACKET_STAGGER_MS
          packet.placeAtFraction(line, sinceSendMs / travelMs)
          const sinceArrivalMs = sinceSendMs - travelMs
          const arrived = sinceArrivalMs >= 0
          for (const cell of cells[row]) cell.classList.toggle("is-on", arrived)
          const glowLevel = arrived ? Math.max(0, 1 - sinceArrivalMs / ARRIVAL_GLOW_MS) : 0
          glows[row].style.setProperty("--pulse", String(Math.round(glowLevel * 100) / 100))
        })
      },
    }
  }

  #reset(): void {
    for (const row of this.#globalCells) for (const cell of row) cell.classList.remove("is-on")
    for (const row of [...this.#regCells, ...this.#sharedCells]) for (const cell of row) cell.classList.remove("is-on")
    for (const glowElement of [...this.#regGlows, ...this.#sharedGlows]) glowElement.style.setProperty("--pulse", "0")
    for (const packet of [...this.#loadPackets, ...this.#storePackets]) packet.hide()
    this.#tileOutline?.classList.remove("is-hot")
    this.#wireLoad?.group.classList.remove("is-on")
    this.#wireStore?.group.classList.remove("is-on")
    this.#sweepLoad?.classList.remove("is-on")
    this.#sweepStore?.classList.remove("is-on")
    this.#sharedLabel?.classList.remove("is-on")
    this.#setBarrier("rest")
  }

  #lightTile(progress: number): void {
    const { rows, cols } = sliceOf(this.#kStep)
    const litRows = Math.ceil(progress * TILE)
    for (let row = rows[0]; row < rows[1]; row += 1) {
      for (let col = cols[0]; col < cols[1]; col += 1) {
        this.#globalCells[row][col].classList.toggle("is-on", row - rows[0] < litRows)
      }
    }
    this.#tileOutline?.classList.toggle("is-hot", progress > 0)
  }

  #placeOutline(kStep: number): void {
    const tile = this.#geometry?.tileRect(kStep)
    if (!tile || !this.#tileOutline) return
    this.#tileOutline.setAttribute("x", String(tile.x))
    this.#tileOutline.setAttribute("y", String(tile.y))
  }

  #placeLead(): void {
    const lead = this.#geometry?.lead(this.#kStep)
    if (!lead) return
    this.#lead?.setAttribute("d", lead.toPathData())
    this.#leadNode?.setAttribute("cx", String(lead.start.x))
    this.#leadNode?.setAttribute("cy", String(lead.start.y))
  }

  #setBarrier(mode: "rest" | "wait" | "pass"): void {
    this.#barrier?.classList.toggle("is-warn", mode === "wait")
    this.#barrierLabel?.classList.toggle("is-warn", mode === "wait")
    this.#barrier?.classList.toggle("is-idle", mode === "pass")
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

// A glow behind one row of cells, for the arrival pulse. Its opacity follows the --pulse property.
function rowGlow(parent: Element, row: Rect[], space: Space): SVGCircleElement {
  const first = row[0]
  const last = row[row.length - 1]
  const element = glow(
    parent,
    (first.x + last.x + last.width) / 2,
    first.y + first.height / 2,
    (last.x + last.width - first.x) * 0.6,
    space
  )
  element.classList.add("is-pulse")
  return element
}
