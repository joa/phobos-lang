// A model that does not fit. One layer of a mixture of experts: the router picks 8 of 256 experts a
// token; picks already in the cache on the card run there, misses are computed by the CPU from host
// memory (decode) or copied over PCIe into the least recently used slot (prompt). A live simulation
// that runs while in view; under reduced motion it rests on a settled token and Step advances it.

import { ChartFigure } from "../../../lib/figure/chart-figure"
import { box, label, plate, wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { StarPacket } from "../../../lib/figure/packet"
import { Polyline } from "../../../lib/figure/polyline"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import type { Rect, TextAt } from "../../inference/figure-geometry"
import { expertCacheGeometry, type ExpertCacheGeometry } from "./geometry"
import { PICKS_PER_TOKEN, RouterSimulation, type MissPath, type TokenResult } from "./simulation"

const ROUTE_MS = 220
const HITS_MS = 200
const MISSES_MS = 720
const HOLD_MS = 160
const COPY_FLIGHT_MS = 460
const COPY_STAGGER_MS = 70
const RESULT_FLIGHT_MS = 200
const PACKETS = 8

export function readoutFor(token: TokenResult): string {
  const misses = PICKS_PER_TOKEN - token.hitCount
  const rate = Math.round(token.hitRate * 100)
  const what =
    token.path === "decode"
      ? misses === 0
        ? "every pick runs on the card"
        : `the CPU computes ${misses === 1 ? "the miss" : `the ${misses} misses`} from host memory`
      : misses === 0
        ? "every pick runs on the card"
        : `${misses === 1 ? "the miss crosses" : `${misses} misses cross`} PCIe into the cache`
  return `Token ${token.number}: ${token.hitCount} of ${PICKS_PER_TOKEN} picks already on the card, ${what}. Simulated hit rate so far ${rate}%.`
}

export default class ExpertCacheFigure extends ChartFigure {
  #geometry: ExpertCacheGeometry | null = null
  #simulation = new RouterSimulation()
  #token: TokenResult | null = null
  #cells: SVGRectElement[] = []
  #slots: SVGRectElement[] = []
  #missWires: SVGPathElement[] = []
  #linkGroups: SVGGElement[] = []
  #cpuBox: SVGRectElement | null = null
  #copyPackets: StarPacket[] = []
  #resultPacket: StarPacket | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null
  #running = false

  get #path(): MissPath {
    return this.state.path === "prompt" ? "prompt" : "decode"
  }

  override build(layout: Layout): void {
    const geometry = expertCacheGeometry(layout)
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

    const host = svgEl("g", { class: "sp-reg" }, root)
    this.#addLabel(host, geometry.experts.label)
    this.#cells = geometry.experts.cells.map(cell => rect(host, cell, "chart-cell"))
    this.#missWires = Array.from({ length: PICKS_PER_TOKEN }, () =>
      svgEl("path", { class: "chart-wire is-warn is-hidden" }, root)
    )

    this.#linkGroups = [geometry.link.upper, geometry.link.lower].map(line => wire(root, line, { space: "cost" }).group)
    this.#addLabel(root, geometry.link.label)

    const cpu = svgEl("g", { class: "sp-reg" }, root)
    this.#cpuBox = box(cpu, { ...geometry.cpu.box })
    this.#addLabel(cpu, geometry.cpu.label)
    this.#addLabel(cpu, geometry.cpu.note, "is-small")
    wire(cpu, geometry.cpu.result, { space: "reg" })

    const gpu = svgEl("g", { class: "sp-global" }, root)
    box(gpu, { ...geometry.gpu.frame, className: "is-on is-outline" })
    this.#addLabel(gpu, geometry.gpu.label, "is-on")
    this.#addLabel(gpu, geometry.cache.label, "is-small")
    this.#slots = geometry.cache.slots.map(slot => rect(gpu, slot, "chart-cell"))
    box(gpu, { ...geometry.rest.box, className: "is-idle" })
    this.#addLabel(gpu, geometry.rest.label, "is-small")

    this.#copyPackets = Array.from({ length: PACKETS }, () => new StarPacket(root, { space: "cost" }))
    this.#resultPacket = new StarPacket(root, { space: "reg" })
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (running) return true
    this.#timeline = null
    if (!this.#running) return false
    this.#startToken(true)
    return true
  }

  override settle(): void {
    this.motion.finishAll()
    if (!this.#token) this.#token = this.#stepSimulation()
    const timeline = this.#timeline ?? this.#makeTimeline(this.#token)
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (this.reducedMotion) {
      this.#setToggle(false)
      return
    }
    const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-box")
    this.motion.drawIn(lines)
    this.motion.enter(this.#labels, { delayMs: 240 })
    this.#running = true
    this.#setToggle(true)
    this.#startToken(true)
  }

  override onChange(group: string, value: string): void {
    if (group !== "path") return
    this.#simulation.setPath(value === "prompt" ? "prompt" : "decode")
    this.#startToken(this.#running && !this.reducedMotion)
  }

  override onAction(action: string): void {
    if (action === "toggle") {
      this.#running = !this.#running
      this.#setToggle(this.#running)
      if (this.#running) this.#startToken(true)
    } else if (action === "step") {
      this.#running = false
      this.#setToggle(false)
      this.#startToken(!this.reducedMotion)
    }
  }

  #stepSimulation(): TokenResult {
    if (this.#simulation.path !== this.#path) this.#simulation.setPath(this.#path)
    return this.#simulation.step()
  }

  #startToken(animated: boolean): void {
    this.#token = this.#stepSimulation()
    this.#timeline = this.#makeTimeline(this.#token)
    // While it plays on its own the readout stays quiet; a stepped token is announced.
    this.root.querySelector(".chart-readout")?.setAttribute("aria-live", this.#running ? "off" : "polite")
    if (animated) this.play()
    else this.settle()
  }

  #setToggle(running: boolean): void {
    const button = this.root.querySelector<HTMLButtonElement>('[data-action="toggle"]')
    if (!button) return
    button.textContent = running ? "Pause" : "Play"
    button.setAttribute("aria-pressed", String(!running))
  }

  #makeTimeline(token: TokenResult): Timeline {
    const geometry = this.#geometry
    if (!geometry) return new Timeline([])
    const misses = token.picks.filter(pick => !pick.isHit)
    const steps: TimelineStep[] = [
      { durationMs: 0, apply: () => this.#reset(token) },
      {
        durationMs: ROUTE_MS,
        apply: progress => {
          for (const pick of token.picks) this.#cells[pick.expert].classList.toggle("is-hot", progress > 0)
        },
      },
      {
        durationMs: HITS_MS,
        apply: progress => {
          for (const pick of token.picks) {
            if (!pick.isHit) continue
            this.#cells[pick.expert].classList.add("is-on")
            this.#slots[pick.slot!].classList.toggle("is-hot", progress > 0)
          }
        },
      },
      {
        durationMs: MISSES_MS,
        apply: progress => {
          misses.forEach((pick, index) => {
            const cell = this.#cells[pick.expert]
            cell.classList.remove("is-hot")
            cell.classList.add("is-warn", "is-on", "sp-cost")
            if (token.path === "decode") this.#showMissWire(index, pick.expert, progress)
          })
          if (token.path === "decode") this.#cpuBox?.classList.toggle("is-hot", misses.length > 0 && progress < 1)
          this.#placeCopies(token, progress)
          for (const group of this.#linkGroups)
            group.classList.toggle("is-warn", token.copies.length > 0 && progress < 1)
          if (progress >= 1) this.setReadout(readoutFor(token))
        },
      },
      { durationMs: HOLD_MS, apply: () => {} },
    ]
    return new Timeline(steps)
  }

  #reset(token: TokenResult): void {
    for (const cell of this.#cells) cell.classList.remove("is-hot", "is-on", "is-warn", "sp-cost")
    const slots = this.#simulation.slots
    const copied = new Set(token.copies.map(copy => copy.slot))
    this.#slots.forEach((slot, index) => {
      slot.classList.remove("is-hot", "is-warn")
      slot.classList.toggle("is-on", slots[index] !== null && !copied.has(index))
      slot.classList.toggle("is-idle", copied.has(index))
    })
    for (const wireElement of this.#missWires) wireElement.classList.add("is-hidden")
    for (const packet of this.#copyPackets) packet.hide()
    this.#resultPacket?.hide()
    this.#cpuBox?.classList.remove("is-hot")
  }

  // A dashed wire from a missed expert to the CPU, then the CPU's result to the card.
  #showMissWire(index: number, expert: number, progress: number): void {
    const geometry = this.#geometry
    const wireElement = this.#missWires[index]
    if (!geometry || !wireElement) return
    const cell = geometry.experts.cells[expert]
    const from = { x: cell.x + cell.width / 2, y: cell.y + cell.height / 2 }
    wireElement.setAttribute("d", new Polyline([from, geometry.cpuInlet]).toPathData())
    wireElement.classList.toggle("is-hidden", progress <= 0 || progress >= 1)
    const resultFraction = (progress * MISSES_MS - (MISSES_MS - RESULT_FLIGHT_MS - HOLD_MS)) / RESULT_FLIGHT_MS
    this.#resultPacket?.placeAtFraction(geometry.cpu.result, resultFraction)
  }

  // Copies cross the link one after another and light their slot on arrival.
  #placeCopies(token: TokenResult, progress: number): void {
    const geometry = this.#geometry
    if (!geometry) return
    const elapsedMs = progress * MISSES_MS
    token.copies.forEach((copy, index) => {
      const fraction = (elapsedMs - index * COPY_STAGGER_MS) / COPY_FLIGHT_MS
      this.#copyPackets[index % PACKETS]?.placeAtFraction(geometry.link.path, fraction)
      const slot = this.#slots[copy.slot]
      const arrived = fraction >= 1
      slot.classList.toggle("is-idle", !arrived)
      slot.classList.toggle("is-on", arrived)
      slot.classList.toggle("is-hot", arrived && !copy.isRefill)
    })
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
