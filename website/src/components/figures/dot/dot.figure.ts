// dot and tensor cores: one CTA through gemm's k loop. At each k step panel kt of A and of B travels
// from global memory into shared a and b, and acc += dot(a, b) lands their product in acc, whose
// ruler counts the k steps added. The path toggle switches acc's overlay between the FMA path, a
// block of acc per thread in registers, and the tensor-core path, 16 x 16 fragments tiled by the
// CTA's 8 warps. Schematic: the sizes are illustrative (see geometry.ts).

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { box, glow, label, plate, wire, type Space, type Wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { sweepWire } from "../../../lib/figure/motion"
import { StarPacket } from "../../../lib/figure/packet"
import type { Polyline } from "../../../lib/figure/polyline"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import {
  COPY_LINES,
  DOT_LINE,
  dotGeometry,
  FRAGMENTS,
  K_STEPS,
  NOTES,
  slicesOf,
  TILE_K,
  type DotGeometry,
  type Path,
  type Rect,
  type TextAt,
} from "./geometry"

const PACKET_SPEED_UNITS_PER_MS = 0.26
const PACKET_STAGGER_MS = 110
const ARRIVAL_GLOW_MS = 300
const PANEL_LIGHT_MS = 360
const DOT_MS = 640

export default class DotFigure extends ChartFigure {
  #geometry: DotGeometry | null = null
  #panelsA: SVGRectElement[] = []
  #panelsB: SVGRectElement[] = []
  #outlineA: SVGRectElement | null = null
  #outlineB: SVGRectElement | null = null
  #partsA: SVGRectElement[] = []
  #partsB: SVGRectElement[] = []
  #acc: SVGRectElement | null = null
  #glowA: SVGCircleElement | null = null
  #glowB: SVGCircleElement | null = null
  #glowAcc: SVGCircleElement | null = null
  #wireA: Wire | null = null
  #wireB: Wire | null = null
  #sweepA: SVGPathElement | null = null
  #sweepB: SVGPathElement | null = null
  #joinA: Wire | null = null
  #joinB: Wire | null = null
  #lineA: Polyline | null = null
  #lineB: Polyline | null = null
  #packetsA: StarPacket[] = []
  #packetsB: StarPacket[] = []
  #segments: SVGRectElement[] = []
  #rulerLabel: SVGTextElement | null = null
  #code: SVGTextElement[] = []
  #tensorcore: SVGTextElement | null = null
  #notes: SVGTextElement[] = []
  #tcGroup: SVGGElement | null = null
  #fmaGroup: SVGGElement | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ kt: "0" }, { kt: "1" }, { path: "tc" }, { kt: "2" }, { kt: "3" }, { path: "fma" }]
  }

  get #kStep(): number {
    const kStep = Number(this.state.kt ?? K_STEPS - 1)
    return Number.isInteger(kStep) && kStep >= 0 && kStep < K_STEPS ? kStep : K_STEPS - 1
  }

  get #path(): Path {
    return this.state.path === "tc" ? "tc" : "fma"
  }

  override build(layout: Layout): void {
    const geometry = dotGeometry(layout)
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
    this.#addLabel(globalGroup, geometry.globalA.label)
    this.#addLabel(globalGroup, geometry.globalB.label)
    this.#panelsA = geometry.globalA.panels.map(panel => rect(globalGroup, panel, "chart-cell"))
    this.#panelsB = geometry.globalB.panels.map(panel => rect(globalGroup, panel, "chart-cell"))
    this.#outlineA = box(root, {
      ...pad(geometry.globalA.panels[this.#kStep]),
      space: "global",
      className: "is-outline",
    })
    this.#outlineB = box(root, {
      ...pad(geometry.globalB.panels[this.#kStep]),
      space: "global",
      className: "is-outline",
    })

    this.#lineA = geometry.wireA(this.#kStep)
    this.#lineB = geometry.wireB(this.#kStep)
    this.#wireA = wire(root, this.#lineA, { space: "global" })
    this.#wireB = wire(root, this.#lineB, { space: "global" })
    this.#sweepA = sweepWire(root, this.#lineA.toPathData(), "global")
    this.#sweepB = sweepWire(root, this.#lineB.toPathData(), "global")

    const sharedGroup = svgEl("g", { class: "sp-shared" }, root)
    this.#addLabel(sharedGroup, geometry.sharedA.label)
    this.#addLabel(sharedGroup, geometry.sharedB.label)
    this.#glowA = pulseGlow(sharedGroup, geometry.sharedA.bounds, "shared")
    this.#glowB = pulseGlow(sharedGroup, geometry.sharedB.bounds, "shared")
    this.#glowAcc = pulseGlow(sharedGroup, geometry.acc.bounds, "shared")
    this.#partsA = geometry.sharedA.parts.map(part => rect(sharedGroup, part, "chart-cell"))
    this.#partsB = geometry.sharedB.parts.map(part => rect(sharedGroup, part, "chart-cell"))
    this.#acc = rect(sharedGroup, geometry.acc.bounds, "chart-box")
    this.#addLabel(sharedGroup, geometry.acc.label)
    this.#joinA = wire(root, geometry.joinA, { space: "shared" })
    this.#joinB = wire(root, geometry.joinB, { space: "shared" })

    this.#fmaGroup = svgEl("g", { class: "sp-reg" }, root)
    svgEl("path", { class: "chart-wire is-on is-idle", d: geometry.acc.threadPath }, this.#fmaGroup)
    box(this.#fmaGroup, { ...geometry.acc.threadBlock, className: "is-hot" })
    this.#tcGroup = svgEl("g", { class: "sp-shared" }, root)
    svgEl("path", { class: "chart-wire is-on", d: geometry.acc.fragmentPath }, this.#tcGroup)
    for (const at of geometry.warpLabels) label(this.#tcGroup, at.text, { ...at, className: "is-small" })

    this.#segments = geometry.ruler.segments.map(segment => rect(sharedGroup, segment, "chart-cell"))
    this.#rulerLabel = this.#addLabel(root, geometry.ruler.label, "is-small")

    this.#tensorcore = this.#addLabel(root, geometry.tensorcore)
    this.#code = geometry.code.map(at => this.#addLabel(root, at))
    this.#notes = geometry.notes.map(at => this.#addLabel(root, at, "is-small"))

    this.#packetsA = Array.from({ length: FRAGMENTS }, () => new StarPacket(root, { space: "global" }))
    this.#packetsB = Array.from({ length: FRAGMENTS }, () => new StarPacket(root, { space: "global" }))
    this.#applyPath()
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
    this.#updateReadout()
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) {
      const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire, .chart-box")
      this.motion.drawIn(lines)
      // A hidden label would show for the length of its entrance.
      this.motion.enter(
        this.#labels.filter(text => !text.classList.contains("is-hidden")),
        { delayMs: 240 }
      )
    }
    super.onFirstVisible()
  }

  override onChange(group: string): void {
    if (group === "path") {
      this.#applyPath()
      this.#updateReadout()
    }
    if (group === "kt") {
      const kStep = this.#kStep
      const outlines = [this.#outlineA, this.#outlineB].filter((outline): outline is SVGRectElement => !!outline)
      this.motion.glide(outlines, () => this.#placeOutlines(kStep))
      this.#run()
    }
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run()
  }

  #run(): void {
    this.#updateReadout()
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  #updateReadout(): void {
    const kStep = this.#kStep
    const { a, b, kt } = slicesOf(kStep)
    const path =
      this.#path === "tc"
        ? "Under @tensorcore the f32 operands are rounded to f16, and each of the 8 warps multiplies 2 of acc's 16 x 16 fragments."
        : "On the FMA path each thread adds into its own block of acc, in registers."
    this.setReadout(
      `kt = ${kt}: ${a} and ${b} are copied into shared a and b, and acc += dot(a, b) adds k ${kt} to ${kt + TILE_K - 1}. ${path}`
    )
  }

  #applyPath(): void {
    const path = this.#path
    this.#fmaGroup?.classList.toggle("is-hidden", path !== "fma")
    this.#tcGroup?.classList.toggle("is-hidden", path !== "tc")
    this.#tensorcore?.classList.toggle("is-hidden", path !== "tc")
    this.#tensorcore?.classList.toggle("is-on", path === "tc")
    this.#notes.forEach((note, index) => {
      note.textContent = NOTES[path][index]
      note.classList.toggle("is-on", index === 0)
    })
  }

  // Every k step before the selected one replays at once; the selected one animates.
  #makeTimeline(): Timeline {
    const last = this.#kStep
    const steps: TimelineStep[] = [{ durationMs: 0, apply: () => this.#reset() }]
    for (let kStep = 0; kStep <= last; kStep += 1) steps.push(...this.#stepsFor(kStep, kStep === last))
    return new Timeline(steps)
  }

  #stepsFor(kStep: number, animated: boolean): TimelineStep[] {
    const geometry = this.#geometry
    if (!geometry) return []
    const scale = (durationMs: number) => (animated ? durationMs : 0)
    return [
      {
        durationMs: 0,
        apply: () => {
          this.#placeWires(kStep)
          this.#placeOutlines(kStep)
          for (const part of [...this.#partsA, ...this.#partsB]) part.classList.remove("is-on")
          this.#panelsA.forEach((panel, index) => panel.classList.toggle("is-hot", index === kStep))
          this.#panelsB.forEach((panel, index) => panel.classList.toggle("is-hot", index === kStep))
          this.#setCode(COPY_LINES)
          this.#wireA?.group.classList.add("is-on")
          this.#wireB?.group.classList.add("is-on")
          this.#sweepA?.classList.add("is-on")
          this.#sweepB?.classList.add("is-on")
        },
      },
      {
        durationMs: scale(PANEL_LIGHT_MS),
        apply: progress => {
          this.#outlineA?.classList.toggle("is-hot", progress > 0)
          this.#outlineB?.classList.toggle("is-hot", progress > 0)
        },
      },
      this.#packetStep(scale),
      {
        durationMs: 0,
        apply: () => {
          this.#sweepA?.classList.remove("is-on")
          this.#sweepB?.classList.remove("is-on")
          this.#wireA?.group.classList.remove("is-on")
          this.#wireB?.group.classList.remove("is-on")
          this.#setCode([DOT_LINE])
          this.#joinA?.group.classList.add("is-on")
          this.#joinB?.group.classList.add("is-on")
        },
      },
      {
        durationMs: scale(DOT_MS),
        apply: progress => {
          const landed = progress >= 0.35
          this.#acc?.classList.toggle("is-hot", landed && progress < 1)
          this.#acc?.classList.toggle("is-on", landed)
          const level = landed ? Math.max(0, 1 - (progress - 0.35) / 0.65) : 0
          this.#glowAcc?.style.setProperty("--pulse", String(Math.round(level * 100) / 100))
          this.#segments[kStep].classList.toggle("is-on", landed)
          if (this.#rulerLabel) this.#rulerLabel.textContent = rulerText(landed ? kStep + 1 : kStep)
          if (progress === 1) {
            this.#joinA?.group.classList.remove("is-on")
            this.#joinB?.group.classList.remove("is-on")
          }
        },
      },
    ]
  }

  // Four packets per operand, one per 16-element part, leave one after another at constant speed and
  // light their part of a or b on arrival.
  #packetStep(scale: (durationMs: number) => number): TimelineStep {
    const travelMs = (line: Polyline | null) => (line ? line.lengthUnits / PACKET_SPEED_UNITS_PER_MS : 0)
    const longestMs = Math.max(travelMs(this.#lineA), travelMs(this.#lineB))
    const totalMs = longestMs + (FRAGMENTS - 1) * PACKET_STAGGER_MS + ARRIVAL_GLOW_MS
    return {
      durationMs: scale(totalMs),
      apply: progress => {
        const elapsedMs = progress * totalMs
        const run = (
          line: Polyline | null,
          packets: StarPacket[],
          parts: SVGRectElement[],
          halo: SVGCircleElement | null
        ) => {
          if (!line) return
          const lineMs = travelMs(line)
          let glowLevel = 0
          packets.forEach((packet, index) => {
            const sinceSendMs = elapsedMs - index * PACKET_STAGGER_MS
            packet.placeAtFraction(line, sinceSendMs / lineMs)
            const sinceArrivalMs = sinceSendMs - lineMs
            parts[index].classList.toggle("is-on", sinceArrivalMs >= 0)
            if (sinceArrivalMs >= 0) glowLevel = Math.max(glowLevel, 1 - sinceArrivalMs / ARRIVAL_GLOW_MS)
          })
          halo?.style.setProperty("--pulse", String(Math.round(Math.max(0, glowLevel) * 100) / 100))
        }
        run(this.#lineA, this.#packetsA, this.#partsA, this.#glowA)
        run(this.#lineB, this.#packetsB, this.#partsB, this.#glowB)
      },
    }
  }

  #reset(): void {
    for (const element of [...this.#panelsA, ...this.#panelsB]) element.classList.remove("is-hot")
    for (const element of [...this.#partsA, ...this.#partsB, ...this.#segments]) element.classList.remove("is-on")
    for (const packet of [...this.#packetsA, ...this.#packetsB]) packet.hide()
    for (const halo of [this.#glowA, this.#glowB, this.#glowAcc]) halo?.style.setProperty("--pulse", "0")
    this.#acc?.classList.remove("is-on", "is-hot")
    this.#outlineA?.classList.remove("is-hot")
    this.#outlineB?.classList.remove("is-hot")
    for (const join of [this.#joinA, this.#joinB, this.#wireA, this.#wireB]) join?.group.classList.remove("is-on")
    this.#sweepA?.classList.remove("is-on")
    this.#sweepB?.classList.remove("is-on")
    if (this.#rulerLabel) this.#rulerLabel.textContent = rulerText(0)
    this.#setCode([])
  }

  #setCode(lines: number[]): void {
    this.#code.forEach((line, index) => line.classList.toggle("is-on", lines.includes(index)))
  }

  #placeOutlines(kStep: number): void {
    const geometry = this.#geometry
    if (!geometry) return
    place(this.#outlineA, pad(geometry.globalA.panels[kStep]))
    place(this.#outlineB, pad(geometry.globalB.panels[kStep]))
  }

  #placeWires(kStep: number): void {
    const geometry = this.#geometry
    if (!geometry) return
    this.#lineA = geometry.wireA(kStep)
    this.#lineB = geometry.wireB(kStep)
    for (const [target, sweep, line] of [
      [this.#wireA, this.#sweepA, this.#lineA],
      [this.#wireB, this.#sweepB, this.#lineB],
    ] as const) {
      if (!target) continue
      const d = line.toPathData()
      target.path.setAttribute("d", d)
      sweep?.setAttribute("d", d)
      target.nodes.forEach((node, index) => {
        const point = line.points[index]
        if (!point) return
        node.setAttribute("cx", String(point.x))
        node.setAttribute("cy", String(point.y))
      })
    }
  }

  #addLabel(parent: Element, at: TextAt, className?: string): SVGTextElement {
    const text = label(parent, at.text, { x: at.x, y: at.y, anchor: at.anchor ?? "start", className })
    this.#labels.push(text)
    return text
  }
}

function rulerText(added: number): string {
  return `k steps added: ${added} of ${K_STEPS}`
}

function rect(parent: Element, at: Rect, className: string): SVGRectElement {
  return svgEl("rect", { class: className, x: at.x, y: at.y, width: at.width, height: at.height }, parent)
}

function place(element: SVGRectElement | null, at: Rect): void {
  element?.setAttribute("x", String(at.x))
  element?.setAttribute("y", String(at.y))
}

function pad(at: Rect, padUnits = 2): Rect {
  return { x: at.x - padUnits, y: at.y - padUnits, width: at.width + 2 * padUnits, height: at.height + 2 * padUnits }
}

function pulseGlow(parent: Element, at: Rect, space: Space): SVGCircleElement {
  const element = glow(parent, at.x + at.width / 2, at.y + at.height / 2, Math.max(at.width, at.height) * 0.75, space)
  element.classList.add("is-pulse")
  return element
}
