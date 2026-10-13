// The point cloud behind the drawing: one set of points that every station arranges in its own shape.
// Between stations the points lift off into depth and land in the next shape, under a camera that
// swings a little as it travels; at a station they flow along the graph's edges, follow the passes'
// cursor, or light the warps. Canvas 2D in frame units, colors read from the page's tokens, additive
// only on the dark theme. Draws only what render() is asked for; the scene owns the frame loop.

import { easeInOut, easeOut } from "../../../lib/figure/math"
import { Polyline } from "../../../lib/figure/polyline"
import { warpLightMs } from "./gpu"
import type { PassesTiming } from "./passes"
import type { StationId } from "./stations"
import { PARTICLES, TONES, type Frame, type StationGeometry, type Tone } from "./types"

const FOCAL_UNITS = 900
const STREAM_SPEED_UNITS_PER_MS = 0.075
const RADIUS_UNITS = [0.95, 1.4, 1.95]
const MAX_STAGGER = 0.3
const GLOW_UNITS = 7
const DROP_UNITS = 90
const DROP_MS = 700

export interface CloudStation {
  id: StationId
  geometry: StationGeometry
  timing?: PassesTiming
}

export interface CloudPose {
  /** The station being shown or moved to. */
  station: number
  moving: boolean
  /** 0 to 1 through a move. */
  progress: number
  /** Time since the station came on. */
  stationMs: number
  /** Turn direction of the camera in this move, +1 or -1. */
  swing: number
}

function hash(index: number, salt: number): number {
  const value = Math.sin(index * 127.1 + salt * 311.7) * 43758.5453
  return value - Math.floor(value)
}

export class Cloud {
  #canvas: HTMLCanvasElement
  #context: CanvasRenderingContext2D
  #probes: HTMLElement
  #stations: CloudStation[] = []
  #streams: Polyline[][] = []
  #frame: Frame = { width: 1200, height: 600 }
  #colors: Record<Tone, string> = Object.fromEntries(TONES.map(tone => [tone, "#888"])) as Record<Tone, string>
  #sprites: Partial<Record<Tone, HTMLCanvasElement>> = {}
  #additive = true
  #pxPerUnit = 1
  // Last drawn pose of every point, and the pose a move leaves from.
  #x = new Float32Array(PARTICLES)
  #y = new Float32Array(PARTICLES)
  #z = new Float32Array(PARTICLES)
  #a = new Float32Array(PARTICLES)
  #fromX = new Float32Array(PARTICLES)
  #fromY = new Float32Array(PARTICLES)
  #fromZ = new Float32Array(PARTICLES)
  #fromA = new Float32Array(PARTICLES)
  #fromTone: Tone[] = new Array(PARTICLES).fill("dust")
  #tone: Tone[] = new Array(PARTICLES).fill("dust")
  #size = new Uint8Array(PARTICLES)
  #hasPose = false

  constructor(canvas: HTMLCanvasElement, probes: HTMLElement) {
    this.#canvas = canvas
    const context = canvas.getContext("2d")
    if (!context) throw new Error("no 2d context")
    this.#context = context
    this.#probes = probes
    this.readColors()
  }

  setStations(stations: CloudStation[]): void {
    this.#stations = stations
    this.#frame = stations[0]?.geometry.frame ?? this.#frame
    this.#streams = stations.map(station => station.geometry.streams.map(stream => new Polyline(stream.points)))
    this.resize()
  }

  resize(): void {
    const widthPx = this.#canvas.clientWidth
    const heightPx = this.#canvas.clientHeight
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    this.#canvas.width = Math.max(1, Math.round(widthPx * dpr))
    this.#canvas.height = Math.max(1, Math.round(heightPx * dpr))
    this.#pxPerUnit = widthPx / this.#frame.width
    this.#context.setTransform(dpr * this.#pxPerUnit, 0, 0, dpr * this.#pxPerUnit, 0, 0)
  }

  readColors(): void {
    for (const tone of TONES) {
      const probe = this.#probes.querySelector<HTMLElement>(`.t-${tone}`)
      if (probe) this.#colors[tone] = getComputedStyle(probe).color
    }
    this.#additive = getComputedStyle(document.documentElement).colorScheme.trim() !== "light"
    this.#sprites = {}
  }

  /** Freezes the points where they are drawn now, as the start of a move. */
  beginMove(): void {
    this.#fromX.set(this.#x)
    this.#fromY.set(this.#y)
    this.#fromZ.set(this.#z)
    this.#fromA.set(this.#hasPose ? this.#a : new Float32Array(PARTICLES))
    this.#fromTone = this.#tone.slice()
  }

  render(pose: CloudPose): void {
    const station = this.#stations[pose.station]
    if (!station) return
    const targets = station.geometry.targets
    const { width, height } = this.#frame
    const cx = width / 2
    const cy = height / 2
    const progress = pose.moving ? pose.progress : 1
    const funnel = station.id === "gpu"
    let minY = Infinity
    let maxY = -Infinity
    if (funnel) {
      for (const point of targets) {
        if (point.alpha > 0) {
          minY = Math.min(minY, point.y)
          maxY = Math.max(maxY, point.y)
        }
      }
    }
    const yaw = Math.sin(Math.PI * progress) * 0.24 * pose.swing
    const cos = Math.cos(yaw)
    const sin = Math.sin(yaw)
    // At rest the move's time is past, so the station's own motion drives the points.
    const atMs = pose.moving ? 0 : pose.stationMs

    for (let index = 0; index < PARTICLES; index += 1) {
      const point = targets[index]
      let x = point.x
      let y = point.y
      let z = point.z
      let alpha = point.alpha
      ;[x, y, alpha] = this.#animate(station, pose.station, index, atMs, x, y, alpha)
      if (!pose.moving) {
        x += Math.sin(pose.stationMs * 0.0011 + index * 1.7) * 0.45
        y += Math.cos(pose.stationMs * 0.0009 + index * 2.3) * 0.45
      }
      this.#tone[index] = point.tone
      this.#size[index] = point.size

      if (pose.moving && this.#hasPose) {
        const stagger = funnel
          ? ((point.y - minY) / Math.max(1, maxY - minY)) * MAX_STAGGER
          : (point.x / width) * (MAX_STAGGER - 0.06) + hash(index, 1) * 0.06
        const local = Math.min(1, Math.max(0, (progress - stagger) / (1 - MAX_STAGGER)))
        const eased = easeInOut(local)
        // Most points sink into depth on the way; a few fly past the camera.
        const reach = hash(index, 2)
        const lift = Math.sin(Math.PI * eased) * (reach < 0.2 ? -(120 + reach * 900) : 140 + 420 * reach)
        const fromX = this.#fromX[index]
        const fromY = this.#fromY[index]
        if (funnel) {
          const u = 1 - eased
          x = u * u * fromX + 2 * u * eased * (cx + (hash(index, 3) - 0.5) * 120) + eased * eased * x
          y = u * u * fromY + 2 * u * eased * 20 + eased * eased * y
        } else {
          x = fromX + (x - fromX) * eased
          y = fromY + (y - fromY) * eased
        }
        z = this.#fromZ[index] + (z - this.#fromZ[index]) * eased + lift
        alpha = Math.min(1, (this.#fromA[index] * (1 - eased) + alpha * eased) * (1 + 0.35 * Math.sin(Math.PI * eased)))
        if (eased < 0.5) this.#tone[index] = this.#fromTone[index]
      }
      this.#x[index] = x
      this.#y[index] = y
      this.#z[index] = z
      this.#a[index] = alpha
    }
    this.#hasPose = true
    this.#draw(cx, cy, cos, sin)
  }

  #animate(station: CloudStation, stationIndex: number, index: number, ms: number, x: number, y: number, alpha: number): [number, number, number] {
    const point = station.geometry.targets[index]
    if (point.stream !== undefined) {
      const stream = station.geometry.streams[point.stream]
      const line = this.#streams[stationIndex][point.stream]
      const since = ms - stream.startMs
      const phase = point.phase ?? 0
      if (since <= 0) {
        const at = line.pointAtFraction(phase)
        return [at.x, at.y, 0.3]
      }
      const fraction = (phase + (since * STREAM_SPEED_UNITS_PER_MS) / line.lengthUnits) % 1
      const at = line.pointAtFraction(fraction)
      const ends = Math.min(1, fraction * 8, (1 - fraction) * 8)
      return [at.x, at.y, Math.min(1, since / 500) * ends * 0.95]
    }
    if (station.id === "passes" && station.timing && point.group !== undefined) {
      const buffer = station.timing.buffers[point.group]
      const since = ms - buffer.dropMs
      if (since <= 0) return [x, y - DROP_UNITS, 0]
      const fall = easeOut(Math.min(1, since / DROP_MS))
      const retired = buffer.retireMs === null ? 0 : Math.min(1, Math.max(0, (ms - buffer.retireMs) / DROP_MS))
      return [x, y - DROP_UNITS * (1 - fall), alpha * fall * (1 - 0.8 * retired)]
    }
    if (station.id === "gpu" && point.group !== undefined && alpha > 0) {
      const lit = ms - warpLightMs(Math.floor(point.group / 8), point.group % 8)
      const flash = lit > 0 ? Math.exp(-lit / 380) * 0.6 : 0
      const wave = lit > 0 ? 0.25 * Math.max(0, Math.sin(ms * 0.0026 - point.group * 0.42)) ** 6 : 0
      return [x, y, Math.min(1, alpha * (lit > 0 ? 0.85 : 0.45) + flash + wave)]
    }
    if (station.id === "ptx" && alpha > 0) {
      return [x, y, alpha * (0.78 + 0.22 * Math.sin(ms * 0.0024 - x * 0.018))]
    }
    return [x, y, alpha]
  }

  #draw(cx: number, cy: number, cos: number, sin: number): void {
    const context = this.#context
    const { width, height } = this.#frame
    context.save()
    context.setTransform(1, 0, 0, 1, 0, 0)
    context.clearRect(0, 0, this.#canvas.width, this.#canvas.height)
    context.restore()
    context.globalCompositeOperation = this.#additive ? "lighter" : "source-over"
    const minRadius = 0.55 / this.#pxPerUnit
    for (let index = 0; index < PARTICLES; index += 1) {
      const alpha = this.#a[index]
      if (alpha <= 0.01) continue
      const dx = this.#x[index] - cx
      const z = this.#z[index]
      const rx = dx * cos - z * sin
      const rz = dx * sin + z * cos
      const scale = FOCAL_UNITS / (FOCAL_UNITS + rz)
      if (scale <= 0) continue
      const sx = cx + rx * scale
      const sy = cy + (this.#y[index] - cy) * scale
      if (sx < -20 || sx > width + 20 || sy < -20 || sy > height + 20) continue
      const depthFade = Math.min(1, 0.3 + 0.7 * scale)
      const tone = this.#tone[index]
      const size = this.#size[index]
      const radius = Math.max(minRadius, RADIUS_UNITS[size] * scale)
      context.globalAlpha = Math.min(1, alpha * depthFade)
      if (size === 2 && this.#additive) {
        const sprite = this.#sprite(tone)
        const glow = GLOW_UNITS * scale
        context.globalAlpha = Math.min(1, alpha * depthFade * 0.22)
        context.drawImage(sprite, sx - glow, sy - glow, glow * 2, glow * 2)
        context.globalAlpha = Math.min(1, alpha * depthFade)
      }
      context.fillStyle = this.#colors[tone]
      context.beginPath()
      context.arc(sx, sy, radius, 0, Math.PI * 2)
      context.fill()
    }
    context.globalAlpha = 1
    context.globalCompositeOperation = "source-over"
  }

  #sprite(tone: Tone): HTMLCanvasElement {
    const cached = this.#sprites[tone]
    if (cached) return cached
    const sprite = document.createElement("canvas")
    sprite.width = 32
    sprite.height = 32
    const context = sprite.getContext("2d")
    if (context) {
      const gradient = context.createRadialGradient(16, 16, 0, 16, 16, 16)
      gradient.addColorStop(0, this.#colors[tone])
      gradient.addColorStop(1, "transparent")
      context.fillStyle = gradient
      context.fillRect(0, 0, 32, 32)
    }
    this.#sprites[tone] = sprite
    return sprite
  }
}
