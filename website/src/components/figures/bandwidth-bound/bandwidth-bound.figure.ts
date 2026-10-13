// Decode waits on memory. Weights stream from VRAM over the memory bus to the SMs. In decode every
// weight crosses the bus once a token for about two operations, so the bus is always busy and only a few
// SMs have work; a prompt uses each weight for every token, so the SMs are busy and the bus mostly idle.
// Runs continuously while in view. Schematic: the timing is illustrative, the ratios are derived (model.ts).

import { ChartFigure } from "../../../lib/figure/chart-figure"
import { box, glow, label, plate, wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { clamp, easeInOut, mulberry32 } from "../../../lib/figure/math"
import { setBreathing, sweepWire } from "../../../lib/figure/motion"
import { StarPacket } from "../../../lib/figure/packet"
import { svgEl } from "../../../lib/figure/svg"
import { knuthHash, type Rect, type TextAt } from "../../inference/figure-geometry"
import { bandwidthGeometry, notesFor, WEIGHT_ROWS, type BandwidthGeometry } from "./geometry"
import { CARD, DECODE_BUSY_SMS, DECODE_MATH_BUSY, PROMPT_BUS_BUSY, READOUTS, percent, type Phase } from "./model"

const PACKET_TRAVEL_MS = 1250
const PACKET_POOL = 6
const POINT_MOVE_MS = 700
const HOT_COLUMN_MS = 160
const INTRO_SWITCH_MS = [6000, 12000]

interface PhaseTiming {
  packetIntervalMs: number
  passMs: number
  // A frame with the bus full, for the settled end state.
  stillMs: number
}

const TIMING: Record<Phase, PhaseTiming> = {
  decode: { packetIntervalMs: PACKET_TRAVEL_MS / 5, passMs: 2400, stillMs: 1300 },
  prompt: { packetIntervalMs: 4000, passMs: 60000, stillMs: 600 },
}

// The SMs one delivery keeps busy: DECODE_BUSY_SMS of them, picked by a hash of the delivery's index so
// they wander over the grid the same way on every load.
export function smsForDelivery(delivery: number): number[] {
  const random = mulberry32(knuthHash(delivery))
  const picks = new Set<number>()
  while (picks.size < DECODE_BUSY_SMS) picks.add(Math.floor(random() * CARD.smCount))
  return [...picks]
}

export default class BandwidthBoundFigure extends ChartFigure {
  #geometry: BandwidthGeometry | null = null
  #rows: SVGRectElement[] = []
  #cells: SVGRectElement[] = []
  #packets: StarPacket[] = []
  #busGroups: SVGGElement[] = []
  #busSweep: SVGPathElement | null = null
  #weightsBox: SVGRectElement | null = null
  #smBox: SVGRectElement | null = null
  #notes: { weights: SVGTextElement; bus: SVGTextElement; sms: SVGTextElement } | null = null
  #point: SVGCircleElement | null = null
  #pointGlow: SVGCircleElement | null = null
  #pointLabel: SVGTextElement | null = null
  #memorySegment: SVGPathElement | null = null
  #computeSegment: SVGPathElement | null = null
  #labels: SVGTextElement[] = []
  #elapsedMs = 0
  #introMs: number | null = null
  #paused = false
  #pointFromX = 0
  #pointMoveMs = POINT_MOVE_MS

  get #phase(): Phase {
    return this.state.phase === "prompt" ? "prompt" : "decode"
  }

  override build(layout: Layout): void {
    const geometry = bandwidthGeometry(layout)
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

    const weights = svgEl("g", { class: "sp-global" }, root)
    this.#weightsBox = box(weights, { ...geometry.weights.box, className: "is-on" })
    this.#rows = geometry.weights.rows.map(row => rect(weights, row, "chart-cell"))
    this.#addLabel(weights, geometry.weights.label, "is-on")

    const bus = svgEl("g", { class: "sp-global" }, root)
    this.#busGroups = [geometry.bus.upper, geometry.bus.lower].map(line => wire(bus, line, { space: "global" }).group)
    this.#busSweep = sweepWire(bus, geometry.bus.path.toPathData(), "global")
    this.#addLabel(bus, geometry.bus.label)

    const sms = svgEl("g", { class: "sp-reg" }, root)
    this.#smBox = box(sms, { ...geometry.sms.box })
    this.#cells = geometry.sms.cells.map(cell => rect(sms, cell, "chart-cell"))
    this.#addLabel(sms, geometry.sms.label)

    const noteAt = (x: number, y: number, anchor: "start" | "middle") =>
      label(root, "", { x, y, anchor, className: "is-small" })
    this.#notes = {
      weights: noteAt(geometry.weights.note.x, geometry.weights.note.y, "start"),
      bus: noteAt(geometry.bus.note.x, geometry.bus.note.y, "middle"),
      sms: noteAt(geometry.sms.note.x, geometry.sms.note.y, "start"),
    }

    const { axis } = geometry
    this.#memorySegment = svgEl(
      "path",
      { class: "chart-wire sp-cost", d: `M${axis.x0} ${axis.y}H${axis.balanceX}` },
      root
    )
    this.#computeSegment = svgEl(
      "path",
      { class: "chart-wire sp-reg", d: `M${axis.balanceX} ${axis.y}H${axis.x1}` },
      root
    )
    for (const tick of axis.ticks) {
      svgEl("path", { class: "chart-ticks", d: `M${tick.x} ${axis.y - 4}V${axis.y + 4}` }, root)
      this.#addLabel(root, tick.label, "is-small")
    }
    svgEl("path", { class: "chart-wire is-warn", d: `M${axis.balanceX} ${axis.y - 10}V${axis.y + 8}` }, root)
    this.#addLabel(root, axis.balanceLabel, "is-small")
    this.#addLabel(root, axis.memoryLabel, "is-small")
    this.#addLabel(root, axis.computeLabel, "is-small")
    this.#addLabel(root, axis.title, "is-small")

    const pointX = geometry.pointX(this.#phase)
    this.#pointGlow = glow(root, pointX, axis.y, 14, "global")
    this.#pointGlow.classList.add("is-on")
    this.#point = svgEl("circle", { class: "chart-node is-on sp-global", cx: pointX, cy: axis.y, r: 3.5 }, root)
    this.#pointLabel = label(root, "", { x: 0, y: axis.pointLabelY, className: "is-hot" })

    this.#packets = Array.from({ length: PACKET_POOL }, () => new StarPacket(root, { space: "global" }))
    this.#pointFromX = pointX
    this.#pointMoveMs = POINT_MOVE_MS
  }

  override update(dtMs: number): boolean {
    if (this.#paused) return false
    this.#elapsedMs += dtMs
    if (this.#introMs !== null) {
      const before = this.#introMs
      this.#introMs += dtMs
      if (this.hasInteracted) this.#introMs = null
      else {
        const crossed = INTRO_SWITCH_MS.findIndex(atMs => before < atMs && atMs <= this.#introMs!)
        if (crossed >= 0) this.#switchTo(crossed === 0 ? "prompt" : "decode")
        if (this.#introMs >= INTRO_SWITCH_MS[INTRO_SWITCH_MS.length - 1]) this.#introMs = null
      }
    }
    this.#pointMoveMs = Math.min(POINT_MOVE_MS, this.#pointMoveMs + dtMs)
    this.#render()
    return true
  }

  override settle(): void {
    this.#elapsedMs = TIMING[this.#phase].stillMs
    this.#pointMoveMs = POINT_MOVE_MS
    this.#render()
    this.setReadout(READOUTS[this.#phase])
  }

  override onFirstVisible(): void {
    if (this.reducedMotion) return
    const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire, .chart-box")
    this.motion.drawIn(lines)
    this.motion.enter(this.#labels, { delayMs: 240 })
    this.#introMs = 0
    this.#elapsedMs = 0
    if (this.#pointGlow) setBreathing(this.#pointGlow, true)
    this.play()
  }

  override onChange(group: string, value: string): void {
    if (group !== "phase") return
    this.#introMs = null
    this.#switchTo(value === "prompt" ? "prompt" : "decode")
  }

  override onAction(action: string): void {
    if (action !== "toggle") return
    this.#introMs = null
    this.#paused = !this.#paused
    const button = this.root.querySelector<HTMLButtonElement>('[data-action="toggle"]')
    if (button) {
      button.textContent = this.#paused ? "Play" : "Pause"
      button.setAttribute("aria-pressed", String(this.#paused))
    }
    if (!this.#paused) this.play()
  }

  #switchTo(phase: Phase): void {
    const geometry = this.#geometry
    if (!geometry) return
    this.#pointFromX = this.#currentPointX()
    this.setState({ phase })
    this.#elapsedMs = 0
    this.#pointMoveMs = 0
    this.setReadout(READOUTS[phase])
    if (this.reducedMotion || this.#paused) {
      this.#pointMoveMs = POINT_MOVE_MS
      this.settle()
      return
    }
    this.play()
  }

  #currentPointX(): number {
    const geometry = this.#geometry
    if (!geometry) return 0
    const t = easeInOut(clamp(this.#pointMoveMs / POINT_MOVE_MS, 0, 1))
    return this.#pointFromX + (geometry.pointX(this.#phase) - this.#pointFromX) * t
  }

  // Everything the frame shows is a function of the phase, the time in it and the point's glide.
  #render(): void {
    const geometry = this.#geometry
    if (!geometry) return
    const phase = this.#phase
    const timing = TIMING[phase]
    const t = this.#elapsedMs

    // Packets leave every interval and cross the bus at constant speed.
    const newest = Math.floor(t / timing.packetIntervalMs)
    this.#packets.forEach((packet, slot) => {
      const index = newest - slot
      const fraction = index < 0 ? -1 : (t - index * timing.packetIntervalMs) / PACKET_TRAVEL_MS
      packet.placeAtFraction(geometry.bus.path, fraction)
    })
    const delivered = Math.floor((t - PACKET_TRAVEL_MS) / timing.packetIntervalMs)

    // The read cursor walks the weight rows once a pass.
    const passFraction = (t % timing.passMs) / timing.passMs
    const current = Math.floor(passFraction * WEIGHT_ROWS)
    this.#rows.forEach((row, index) => {
      row.classList.toggle("is-hot", index === current)
      row.classList.toggle("is-on", index < current)
    })

    if (phase === "decode") {
      const busy = new Set(delivered >= 0 ? smsForDelivery(delivered) : [])
      this.#cells.forEach((cell, index) => {
        cell.classList.toggle("is-hot", busy.has(index))
        cell.classList.remove("is-on")
      })
    } else {
      const hotColumn = Math.floor(t / HOT_COLUMN_MS) % geometry.sms.columns
      this.#cells.forEach((cell, index) => {
        cell.classList.add("is-on")
        cell.classList.toggle("is-hot", index % geometry.sms.columns === hotColumn)
      })
    }

    const isDecode = phase === "decode"
    for (const group of this.#busGroups) group.classList.toggle("is-hot", isDecode)
    for (const group of this.#busGroups) group.classList.toggle("is-on", true)
    this.#busSweep?.classList.toggle("is-on", isDecode)
    this.#smBox?.classList.toggle("is-on", true)
    this.#smBox?.classList.toggle("is-hot", !isDecode)
    this.#weightsBox?.classList.toggle("is-hot", isDecode)
    this.#memorySegment?.classList.toggle("is-warn", isDecode)
    this.#memorySegment?.classList.toggle("is-idle", !isDecode)
    this.#computeSegment?.classList.toggle("is-on", !isDecode)
    this.#computeSegment?.classList.toggle("is-idle", isDecode)

    const notes = notesFor(phase, percent(PROMPT_BUS_BUSY), percent(DECODE_MATH_BUSY))
    if (this.#notes) {
      setText(this.#notes.weights, notes.weights)
      setText(this.#notes.bus, notes.bus)
      setText(this.#notes.sms, notes.sms)
      this.#notes.bus.classList.toggle("is-hot", isDecode)
      this.#notes.sms.classList.toggle("is-warn", isDecode)
      this.#notes.sms.classList.toggle("is-hot", !isDecode)
    }

    const pointX = this.#currentPointX()
    const landed = this.#pointMoveMs >= POINT_MOVE_MS
    this.#point?.setAttribute("cx", String(round(pointX)))
    this.#pointGlow?.setAttribute("cx", String(round(pointX)))
    if (this.#pointLabel) {
      const at = geometry.pointLabel(phase)
      this.#pointLabel.setAttribute("x", String(round(at.x)))
      setText(this.#pointLabel, at.text)
      this.#pointLabel.classList.toggle("is-hidden", !landed)
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

function setText(element: SVGTextElement, text: string): void {
  if (element.textContent !== text) element.textContent = text
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
