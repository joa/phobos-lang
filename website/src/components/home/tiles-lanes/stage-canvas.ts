// While the scene plays, everything that changes every frame (the particles, their glows and the
// few paths a frame restyles) is drawn on a canvas over the stage instead of written into SVG: a few
// hundred elements restyled every frame cost the browser a style pass, a layerize and a paint each
// time, a canvas costs one draw. The look still comes from the stylesheet: colors, opacities, widths
// and dashes are read from the SVG elements the build rendered (hidden while playing), again after a
// theme switch or a resize.

import type { ParticleState, PathState } from "./scene"
import { VIEW_WIDTH_UNITS } from "./scene"

interface PathLook {
  fill: string | null
  stroke: string | null
  fillOpacity: number
  strokeOpacity: number
  widthUnits: number
  dash: number[]
  round: boolean
}

export interface CanvasPath {
  element: SVGPathElement
  state: PathState | undefined
}

export class StageCanvas {
  #canvas: HTMLCanvasElement
  #context: CanvasRenderingContext2D | null
  #particles: SVGCircleElement[]
  #glows: SVGCircleElement[]
  #particleColors: string[] = []
  #glowColors: string[] = []
  #looks = new Map<SVGPathElement, PathLook>()
  #shapes = new Map<SVGPathElement, { d: string; shape: Path2D }>()
  #unitPx = 1
  #isStale = true

  constructor(canvas: HTMLCanvasElement, particles: SVGCircleElement[], glows: SVGCircleElement[]) {
    this.#canvas = canvas
    this.#context = canvas.getContext("2d")
    this.#particles = particles
    this.#glows = glows
    const refresh = () => (this.#isStale = true)
    new ResizeObserver(refresh).observe(canvas)
    new MutationObserver(refresh).observe(document.documentElement, { attributeFilter: ["data-theme", "class"] })
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", refresh)
  }

  draw(paths: CanvasPath[], particles: ParticleState[], glows: ParticleState[]): void {
    const context = this.#context
    if (!context) return
    if (this.#isStale) this.#refresh(paths)
    context.setTransform(1, 0, 0, 1, 0, 0)
    context.clearRect(0, 0, this.#canvas.width, this.#canvas.height)
    context.setTransform(this.#unitPx, 0, 0, this.#unitPx, 0, 0)
    for (const path of paths) this.#drawPath(context, path)
    glows.forEach((glow, index) => {
      if (glow.opacity <= 0 || glow.r <= 0) return
      const color = this.#glowColors[index] ?? "transparent"
      const gradient = context.createRadialGradient(glow.x, glow.y, 0, glow.x, glow.y, glow.r)
      gradient.addColorStop(0, withAlpha(color, 0.32))
      gradient.addColorStop(0.45, withAlpha(color, 0.1))
      gradient.addColorStop(1, withAlpha(color, 0))
      context.globalAlpha = glow.opacity
      context.fillStyle = gradient
      context.fillRect(glow.x - glow.r, glow.y - glow.r, glow.r * 2, glow.r * 2)
    })
    particles.forEach((particle, index) => {
      if (particle.opacity <= 0 || particle.r <= 0) return
      context.globalAlpha = particle.opacity
      context.fillStyle = this.#particleColors[index] ?? "transparent"
      context.beginPath()
      context.arc(particle.x, particle.y, particle.r, 0, Math.PI * 2)
      context.fill()
    })
    context.globalAlpha = 1
  }

  #drawPath(context: CanvasRenderingContext2D, { element, state }: CanvasPath): void {
    const look = this.#looks.get(element)
    const d = state?.d ?? element.getAttribute("d") ?? ""
    const opacity = state?.opacity ?? 1
    if (!look || !d || opacity <= 0) return
    let cached = this.#shapes.get(element)
    if (!cached || cached.d !== d) {
      cached = { d, shape: new Path2D(d) }
      this.#shapes.set(element, cached)
    }
    if (look.fill) {
      context.globalAlpha = opacity * look.fillOpacity
      context.fillStyle = look.fill
      context.fill(cached.shape)
    }
    if (look.stroke) {
      context.globalAlpha = opacity * look.strokeOpacity
      context.strokeStyle = look.stroke
      context.lineWidth = state?.width ?? look.widthUnits
      context.lineCap = look.round ? "round" : "butt"
      context.lineJoin = "round"
      context.setLineDash(look.dash)
      context.stroke(cached.shape)
      context.setLineDash([])
    }
  }

  #refresh(paths: CanvasPath[]): void {
    this.#isStale = false
    const ratio = window.devicePixelRatio || 1
    const widthPx = this.#canvas.clientWidth
    const heightPx = this.#canvas.clientHeight
    this.#canvas.width = Math.max(1, Math.round(widthPx * ratio))
    this.#canvas.height = Math.max(1, Math.round(heightPx * ratio))
    this.#unitPx = (widthPx * ratio) / VIEW_WIDTH_UNITS
    this.#particleColors = this.#particles.map(circle => getComputedStyle(circle).fill)
    this.#glowColors = this.#glows.map(circle => {
      const id = /url\("?#([^")]+)"?\)/.exec(circle.getAttribute("fill") ?? "")?.[1]
      const stop = id ? document.getElementById(id)?.querySelector("stop") : null
      return stop ? getComputedStyle(stop).stopColor : "transparent"
    })
    for (const { element } of paths) {
      const style = getComputedStyle(element)
      const paint = (value: string) => (value === "none" || value === "" ? null : value)
      this.#looks.set(element, {
        fill: paint(style.fill),
        stroke: paint(style.stroke),
        fillOpacity: Number(style.fillOpacity) || 0,
        strokeOpacity: Number(style.strokeOpacity) || 0,
        widthUnits: Number.parseFloat(element.getAttribute("stroke-width") ?? style.strokeWidth) || 1,
        dash: style.strokeDasharray === "none" ? [] : style.strokeDasharray.split(/[ ,]+/).map(Number.parseFloat),
        round: style.strokeLinecap === "round",
      })
    }
  }
}

// A computed color with its alpha replaced. Computed colors come as rgb(), rgba() or color(srgb ...).
function withAlpha(color: string, alpha: number): string {
  const rgb = /^rgba?\(([^,]+),([^,]+),([^,)]+)/.exec(color)
  if (rgb) return `rgba(${rgb[1]},${rgb[2]},${rgb[3]},${alpha})`
  const srgb = /^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)/.exec(color)
  if (srgb) return `color(srgb ${srgb[1]} ${srgb[2]} ${srgb[3]} / ${alpha})`
  return color
}
