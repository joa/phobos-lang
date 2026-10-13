// The decode token loop: a GGUF file's weights feed the layers, the tokenizer turns the prompt into
// tokens, the layers (the fused decode kernel) produce logits, the sampler picks a token, and that
// token both streams out and goes back in as the next input. Schematic: the tokens are illustrative.

import { ChartFigure } from "../../../lib/figure/chart-figure"
import { box, glow, label, plate, wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { setBreathing, sweepWire } from "../../../lib/figure/motion"
import { StarPacket } from "../../../lib/figure/packet"
import { Polyline, type Point } from "../../../lib/figure/polyline"
import { svgText } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"

export const TOKENS = ["Tiles", " move", " through", " shared", " memory", "."]

const PACKET_SPEED_UNITS_PER_MS = 0.32
const COMPUTE_MS = 420
const SAMPLE_MS = 160

interface Box {
  x: number
  y: number
  width: number
  height: number
}

export interface TokenLoopGeometry {
  widthUnits: number
  heightUnits: number
  gguf: Box
  tokenizer: Box
  layers: Box
  sampler: Box
  weights: Polyline
  prompt: Polyline
  logits: Polyline
  feedback: Polyline
  emit: Polyline
  stream: Point
  streamLabel: Point
}

const mid = (b: Box): Point => ({ x: b.x + b.width / 2, y: b.y + b.height / 2 })

export function tokenLoopGeometry(layout: Layout): TokenLoopGeometry {
  if (layout === "wide") {
    const gguf = { x: 36, y: 66, width: 132, height: 40 }
    const tokenizer = { x: 36, y: 164, width: 132, height: 40 }
    const layers = { x: 262, y: 96, width: 196, height: 96 }
    const sampler = { x: 552, y: 124, width: 132, height: 40 }
    return {
      widthUnits: 720,
      heightUnits: 300,
      gguf,
      tokenizer,
      layers,
      sampler,
      weights: new Polyline([
        { x: 168, y: 86 },
        { x: 215, y: 86 },
        { x: 215, y: 120 },
        { x: 262, y: 120 },
      ]),
      prompt: new Polyline([
        { x: 168, y: 184 },
        { x: 215, y: 184 },
        { x: 215, y: 168 },
        { x: 262, y: 168 },
      ]),
      logits: new Polyline([
        { x: 458, y: 144 },
        { x: 552, y: 144 },
      ]),
      feedback: new Polyline([
        { x: 618, y: 124 },
        { x: 618, y: 62 },
        { x: 360, y: 62 },
        { x: 360, y: 96 },
      ]),
      emit: new Polyline([
        { x: 618, y: 164 },
        { x: 618, y: 238 },
      ]),
      stream: { x: 262, y: 258 },
      streamLabel: { x: 36, y: 258 },
    }
  }
  const gguf = { x: 24, y: 56, width: 150, height: 38 }
  const tokenizer = { x: 226, y: 56, width: 150, height: 38 }
  const layers = { x: 100, y: 146, width: 200, height: 84 }
  const sampler = { x: 125, y: 276, width: 150, height: 38 }
  return {
    widthUnits: 400,
    heightUnits: 400,
    gguf,
    tokenizer,
    layers,
    sampler,
    weights: new Polyline([
      { x: 99, y: 94 },
      { x: 99, y: 120 },
      { x: 150, y: 120 },
      { x: 150, y: 146 },
    ]),
    prompt: new Polyline([
      { x: 301, y: 94 },
      { x: 301, y: 120 },
      { x: 250, y: 120 },
      { x: 250, y: 146 },
    ]),
    logits: new Polyline([
      { x: 200, y: 230 },
      { x: 200, y: 276 },
    ]),
    feedback: new Polyline([
      { x: 275, y: 295 },
      { x: 350, y: 295 },
      { x: 350, y: 188 },
      { x: 300, y: 188 },
    ]),
    emit: new Polyline([
      { x: 125, y: 295 },
      { x: 50, y: 295 },
      { x: 50, y: 340 },
    ]),
    stream: { x: 64, y: 362 },
    streamLabel: { x: 24, y: 362 },
  }
}

export default class TokenLoopFigure extends ChartFigure {
  #geometry: TokenLoopGeometry | null = null
  #layersBox: SVGRectElement | null = null
  #layersGlow: SVGCircleElement | null = null
  #samplerBox: SVGRectElement | null = null
  #weightsSweep: SVGPathElement | null = null
  #promptPacket: StarPacket | null = null
  #logitsPacket: StarPacket | null = null
  #emitPacket: StarPacket | null = null
  #feedbackPacket: StarPacket | null = null
  #streamText: SVGTextElement | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override build(layout: Layout): void {
    const geometry = tokenLoopGeometry(layout)
    this.#geometry = geometry
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    const root = this.drawing
    plate(root, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: "FIG / DECODE / TOKEN LOOP",
      schematic: true,
    })
    this.#labels = []

    wire(root, geometry.weights, { space: "global", className: "is-on" })
    this.#weightsSweep = sweepWire(root, geometry.weights.toPathData(), "global")
    wire(root, geometry.prompt, { space: "reg" })
    wire(root, geometry.logits, { space: "reg" })
    wire(root, geometry.feedback, { space: "reg" })
    wire(root, geometry.emit, { space: "reg" })

    const center = mid(geometry.layers)
    this.#layersGlow = glow(root, center.x, center.y, geometry.layers.width * 0.42, "global")
    box(root, { ...geometry.gguf, space: "global", className: "is-on" })
    box(root, { ...geometry.tokenizer })
    this.#layersBox = box(root, { ...geometry.layers, space: "global" })
    this.#samplerBox = box(root, { ...geometry.sampler })

    this.#boxLabel(geometry.gguf, "GGUF file", "is-on sp-global")
    this.#boxLabel(geometry.tokenizer, "tokenizer")
    this.#boxLabel(geometry.layers, "layers", undefined, -9)
    this.#boxLabel(geometry.layers, "fused decode kernel", "is-small", 11)
    this.#boxLabel(geometry.sampler, "sampler")
    this.#labels.push(
      label(root, "out", { x: geometry.streamLabel.x, y: geometry.streamLabel.y, className: "is-small" })
    )

    this.#streamText = svgText(
      "",
      {
        class: "chart-label is-hot",
        x: geometry.stream.x,
        y: geometry.stream.y,
        "dominant-baseline": "middle",
      },
      root
    )

    this.#promptPacket = new StarPacket(root, { space: "reg" })
    this.#logitsPacket = new StarPacket(root, { space: "reg" })
    this.#emitPacket = new StarPacket(root, { space: "reg" })
    this.#feedbackPacket = new StarPacket(root, { space: "reg" })
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    const timeline = this.#timeline ?? this.#makeTimeline()
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (this.reducedMotion) return
    const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire, .chart-box")
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
    const travel = (line: Polyline) => line.lengthUnits / PACKET_SPEED_UNITS_PER_MS
    const steps: TimelineStep[] = [
      { durationMs: 0, apply: () => this.#reset() },
      this.#move(this.#promptPacket, geometry.prompt, travel(geometry.prompt)),
    ]
    TOKENS.forEach((_, index) => {
      steps.push({
        durationMs: COMPUTE_MS,
        apply: progress => this.#layersBox?.classList.toggle("is-hot", progress < 1),
      })
      steps.push(this.#move(this.#logitsPacket, geometry.logits, travel(geometry.logits)))
      steps.push({
        durationMs: SAMPLE_MS,
        apply: progress => this.#samplerBox?.classList.toggle("is-hot", progress < 1),
      })
      const emitMs = travel(geometry.emit)
      const feedbackMs = travel(geometry.feedback)
      const bothMs = Math.max(emitMs, feedbackMs)
      steps.push({
        durationMs: bothMs,
        apply: progress => {
          const elapsedMs = progress * bothMs
          this.#emitPacket?.placeAtFraction(geometry.emit, elapsedMs / emitMs)
          this.#feedbackPacket?.placeAtFraction(geometry.feedback, elapsedMs / feedbackMs)
          if (elapsedMs >= emitMs) this.#showTokens(index + 1)
        },
      })
    })
    steps.push({
      durationMs: 0,
      apply: () => {
        this.#weightsSweep?.classList.remove("is-on")
        if (this.#layersGlow) setBreathing(this.#layersGlow, false)
        this.setReadout("Each sampled token streams out and goes back into the layers as the next input.")
      },
    })
    return new Timeline(steps)
  }

  #move(packet: StarPacket | null, line: Polyline, durationMs: number): TimelineStep {
    return { durationMs, apply: progress => packet?.placeAtFraction(line, progress) }
  }

  #reset(): void {
    this.#showTokens(0)
    this.#weightsSweep?.classList.add("is-on")
    if (this.#layersGlow) setBreathing(this.#layersGlow, true)
    this.#layersBox?.classList.remove("is-hot")
    this.#samplerBox?.classList.remove("is-hot")
    for (const packet of [this.#promptPacket, this.#logitsPacket, this.#emitPacket, this.#feedbackPacket])
      packet?.hide()
    this.setReadout("The layers read the model's weights from global memory for every token they produce.")
  }

  #showTokens(count: number): void {
    if (this.#streamText) this.#streamText.textContent = TOKENS.slice(0, count).join("")
  }

  #boxLabel(at: Box, text: string, className?: string, dyUnits = 0): void {
    const center = mid(at)
    this.#labels.push(label(this.drawing, text, { x: center.x, y: center.y + dyUnits, anchor: "middle", className }))
  }
}
