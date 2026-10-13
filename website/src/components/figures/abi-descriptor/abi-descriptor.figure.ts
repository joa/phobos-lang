// How one tensor parameter becomes kernel arguments. `A: tensor<f32>[M, K]` lowers to seven
// arguments, the flattened memref descriptor: two pointers, the offset, two sizes, two strides.
// A packet carries the parameter out to each slot in turn; the 32/64 control swaps the index
// width, which widens the five index fields and leaves the pointers as they are.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { box, label, plate, wire, type Space } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { StarPacket } from "../../../lib/figure/packet"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import { abiGeometry, totalBytes, type AbiGeometry, type IndexBits, type Slot, type TextAt } from "./geometry"

const PACKET_SPEED_UNITS_PER_MS = 0.34
const STAGGER_MS = 150
const PACKET_PARTICLES = 8

export function readoutFor(bits: IndexBits): string {
  const pointers = "two 8-byte pointers"
  const fields = bits === 32 ? "five 4-byte i32 fields" : "five 8-byte i64 fields"
  return `At ${bits}-bit index, A is ${totalBytes(bits)} bytes of arguments: ${pointers}, then ${fields} for the offset, the sizes and the strides.`
}

function spaceOf(slot: Slot): Space {
  return slot.kind === "pointer" ? "global" : "reg"
}

export default class AbiDescriptorFigure extends ChartFigure {
  #geometry: AbiGeometry | null = null
  #drawnBits: IndexBits | null = null
  #layer: SVGGElement | null = null
  #slotBoxes: SVGRectElement[] = []
  #packets: StarPacket[] = []
  #totals: SVGTextElement[] = []
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ bits: "32" }, { bits: "64" }]
  }

  get #bits(): IndexBits {
    return this.state.bits === "64" ? 64 : 32
  }

  override build(layout: Layout): void {
    const geometry = abiGeometry(layout, this.#bits)
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    plate(this.drawing, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: geometry.plateLabel,
      schematic: true,
    })
    this.#layer = null
    this.#draw(layout, this.#bits)
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

  override onChange(group: string, value: string): void {
    if (group !== "bits") return
    const bits: IndexBits = value === "64" ? 64 : 32
    if (bits !== this.#drawnBits) {
      this.crossfade()
      this.#draw(this.layout, bits)
    }
    this.#run()
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run()
  }

  #run(): void {
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  // Everything that depends on the index width lives in one layer, redrawn when it changes.
  #draw(layout: Layout, bits: IndexBits): void {
    const geometry = abiGeometry(layout, bits)
    this.#geometry = geometry
    this.#drawnBits = bits
    this.#layer?.remove()
    const layer = svgEl("g", { class: "abi-layer" }, this.drawing)
    this.#layer = layer
    this.#labels = []

    for (const slot of geometry.slots) wire(layer, slot.wire, { space: spaceOf(slot) })
    for (const group of geometry.groups) {
      svgEl("path", { class: "chart-wire", d: group.path }, layer)
      this.#addLabel(layer, group.label, "is-small")
    }

    box(layer, { ...geometry.param.rect, space: "global", className: "is-on" })
    this.#addLabel(layer, geometry.param.label, "is-on sp-global")
    this.#addLabel(layer, geometry.widthNote, "is-small")

    this.#slotBoxes = geometry.slots.map(slot => box(layer, { ...slot.rect, space: spaceOf(slot) }))
    for (const slot of geometry.slots) {
      this.#addLabel(layer, slot.valueLabel, `sp-${spaceOf(slot)}`)
      this.#addLabel(layer, slot.bytesLabel, "is-small")
      if (slot.roleLabel) this.#addLabel(layer, slot.roleLabel, "is-small")
    }
    this.#totals = geometry.total.map(text => this.#addLabel(layer, text, "is-small"))

    this.#packets = geometry.slots.map(
      slot => new StarPacket(layer, { space: spaceOf(slot), count: PACKET_PARTICLES })
    )
  }

  #makeTimeline(): Timeline {
    const geometry = this.#geometry
    if (!geometry) return new Timeline([])
    const travelMs = geometry.slots.map(slot => slot.wire.lengthUnits / PACKET_SPEED_UNITS_PER_MS)
    const sendMs = Math.max(...travelMs.map((ms, index) => ms + index * STAGGER_MS))
    const steps: TimelineStep[] = [
      { durationMs: 0, apply: () => this.#reset() },
      {
        durationMs: sendMs,
        apply: progress => {
          const elapsedMs = progress * sendMs
          geometry.slots.forEach((slot, index) => {
            const fraction = (elapsedMs - index * STAGGER_MS) / travelMs[index]
            this.#packets[index]?.placeAtFraction(slot.wire, fraction)
            this.#slotBoxes[index]?.classList.toggle("is-on", fraction >= 1)
          })
        },
      },
      {
        durationMs: 0,
        apply: () => {
          for (const total of this.#totals) total.classList.add("is-on")
        },
      },
    ]
    return new Timeline(steps)
  }

  #reset(): void {
    for (const slotBox of this.#slotBoxes) slotBox.classList.remove("is-on")
    for (const packet of this.#packets) packet.hide()
    for (const total of this.#totals) total.classList.remove("is-on")
    this.setReadout(readoutFor(this.#drawnBits ?? this.#bits))
  }

  #addLabel(parent: Element, at: TextAt, className?: string): SVGTextElement {
    const text = label(parent, at.text, { x: at.x, y: at.y, anchor: at.anchor ?? "start", className })
    this.#labels.push(text)
    return text
  }
}
