// Draws the backdrop on two canvases while the scene plays: the bodies on one, the blue line that
// rings the lit ones on another above it, at an opacity of its own. The look comes from the
// stylesheet: the backdrop element names a hue per body, the metal they turn to, the sky they fade
// into, the shine a polished facet goes toward, how faint the bodies are drawn, the line's color and
// whether to draw in ink, read again after a theme switch.
//
// Facets are drawn opaque, each mixed toward the sky by how dim it is and by how faint the bodies
// sit; drawn with alpha, the hairline seams between neighbouring facets would show the sky through.
// The backdrop lies on the plain sky, so mixing toward it looks the same as an opacity would, and
// lets a highlight (a facet's sheen) shine past it.

import {
  BACKDROP_WIDTH_UNITS,
  facets,
  makeMeshes,
  outlines,
  silvering,
  tone,
  type Facet,
  type Motion,
  type Pose,
} from "./backdrop"

type Rgb = [number, number, number]

const MAX_DPR = 2
const BODIES = 3
// The line's width, in CSS px.
const OUTLINE_WIDTH_PX = 1.25

export class BackdropCanvas {
  #root: HTMLElement
  #bodies: HTMLCanvasElement
  #outline: HTMLCanvasElement
  #meshes = makeMeshes()
  #hues: Rgb[] = []
  #sky: Rgb = [0, 0, 0]
  #metal: Rgb = [255, 255, 255]
  #shine: Rgb = [255, 255, 255]
  #faint = 1
  #line = "transparent"
  #ink = false
  #ratio = 1
  #unitPx = 1
  #isStale = true
  #last: { pose: Pose; motion: Motion } | null = null

  constructor(root: HTMLElement, bodies: HTMLCanvasElement, outline: HTMLCanvasElement) {
    this.#root = root
    this.#bodies = bodies
    this.#outline = outline
    // A resize or a theme switch redraws the last pose, so a paused scene does not keep a stale sky.
    const refresh = () => {
      this.#isStale = true
      if (this.#last) this.draw(this.#last.pose, this.#last.motion)
    }
    new ResizeObserver(refresh).observe(bodies)
    new MutationObserver(refresh).observe(document.documentElement, { attributeFilter: ["data-theme", "class"] })
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", refresh)
  }

  draw(pose: Pose, motion: Motion): void {
    this.#last = { pose, motion }
    const context = this.#bodies.getContext("2d")
    const above = this.#outline.getContext("2d")
    if (!context || !above) return
    if (this.#isStale) this.#refresh()
    const shown = facets(pose, motion, this.#meshes)
    this.#begin(context)
    context.lineJoin = "round"
    // One device pixel of stroke in the facet's own color closes the seams.
    context.lineWidth = 1 / this.#unitPx
    for (const facet of shown) {
      const color = this.#color(facet)
      trace(context, facet.points)
      context.fillStyle = color
      context.strokeStyle = color
      context.fill()
      context.stroke()
    }

    this.#begin(above)
    above.lineCap = "round"
    above.lineWidth = (OUTLINE_WIDTH_PX * this.#ratio) / this.#unitPx
    above.strokeStyle = this.#line
    for (const outline of outlines(pose, motion)) {
      above.globalCompositeOperation = "source-over"
      above.globalAlpha = outline.opacity
      above.beginPath()
      const { segments } = outline
      for (let at = 0; at < segments.length; at += 4) {
        above.moveTo(segments[at], segments[at + 1])
        above.lineTo(segments[at + 2], segments[at + 3])
      }
      above.stroke()
      // A body it does not ring that passes in front of it hides it.
      above.globalCompositeOperation = "destination-out"
      above.globalAlpha = 1
      above.fillStyle = "#000"
      for (const facet of shown) {
        if (facet.depth <= outline.depth || outline.bodies.includes(dominant(facet))) continue
        trace(above, facet.points)
        above.fill()
      }
    }
    above.globalCompositeOperation = "source-over"
    above.globalAlpha = 1
  }

  #begin(context: CanvasRenderingContext2D): void {
    context.setTransform(1, 0, 0, 1, 0, 0)
    context.clearRect(0, 0, context.canvas.width, context.canvas.height)
    context.setTransform(this.#unitPx, 0, 0, this.#unitPx, 0, 0)
  }

  #color(facet: Facet): string {
    const mix = tone(facet, this.#ink)
    const silver = silvering(facet)
    const channel = (c: number) => {
      const [sky, shine] = [this.#sky[c], this.#shine[c]]
      let hue = 0
      for (let body = 0; body < BODIES; body += 1) hue += facet.weights[body] * (this.#hues[body]?.[c] ?? 0)
      hue += (this.#metal[c] - hue) * silver
      const base = sky + (hue - sky) * mix
      const glossy = base + (shine - base) * facet.gloss
      const faded = sky + (glossy - sky) * this.#faint
      return Math.round(faded + (shine - faded) * facet.sheen)
    }
    return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`
  }

  #refresh(): void {
    this.#isStale = false
    const widthPx = this.#bodies.clientWidth
    const heightPx = this.#bodies.clientHeight
    this.#ratio = Math.min(MAX_DPR, window.devicePixelRatio || 1)
    for (const canvas of [this.#bodies, this.#outline]) {
      canvas.width = Math.max(1, Math.round(widthPx * this.#ratio))
      canvas.height = Math.max(1, Math.round(heightPx * this.#ratio))
    }
    this.#unitPx = (widthPx * this.#ratio) / BACKDROP_WIDTH_UNITS
    const probe = document.createElement("span")
    probe.style.display = "none"
    this.#root.append(probe)
    const read = (property: string): Rgb => {
      probe.style.color = `var(${property})`
      return parseRgb(getComputedStyle(probe).color)
    }
    this.#hues = Array.from({ length: BODIES }, (_, body) => read(`--tl-body-${body}`))
    this.#sky = read("--tl-body-sky")
    this.#metal = read("--tl-body-metal")
    this.#shine = read("--tl-body-shine")
    this.#line = `rgb(${read("--tl-outline").join(", ")})`
    probe.remove()
    const style = getComputedStyle(this.#root)
    this.#ink = style.getPropertyValue("--tl-body-ink").trim() === "1"
    const number = (property: string) => Number.parseFloat(style.getPropertyValue(property)) || 1
    this.#faint = number("--tl-bodies-opacity") * number("--tl-backdrop-fade")
  }
}

function trace(context: CanvasRenderingContext2D, points: number[]): void {
  context.beginPath()
  context.moveTo(points[0], points[1])
  for (let at = 2; at < points.length; at += 2) context.lineTo(points[at], points[at + 1])
  context.closePath()
}

// The body whose color a facet mostly is.
function dominant(facet: Facet): number {
  return facet.weights.indexOf(Math.max(...facet.weights))
}

// A computed color as bytes: color(srgb r g b) gives fractions, rgb() gives bytes.
function parseRgb(value: string): Rgb {
  const numbers = value.match(/[\d.]+/g)?.map(Number) ?? [0, 0, 0]
  if (value.startsWith("color(")) return [numbers[0] * 255, numbers[1] * 255, numbers[2] * 255]
  return [numbers[0], numbers[1], numbers[2]]
}
