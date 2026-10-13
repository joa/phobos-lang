// The /inference/ hero's scene as points: a stack of layers, each a lattice of points, with a faint
// haze between them, the orbit's dust, and tokens riding the loop in path.ts. composeFrame projects one
// moment into flat buffers that both the build-time SVG and the canvas draw. Pure; the buffers are
// allocated once and refilled every frame.

import { apply, CAMERA_DISTANCE, multiply, rotateX, rotateY, rotateZ, viewMatrix, type Mat3 } from "../../home/hero/geometry"
import { mulberry32 } from "../../../lib/figure/math"
import {
  arcProgress,
  ENTER_S,
  EXIT_S,
  layerX,
  LAYERS,
  loopPoint,
  ORBIT_RADIUS,
  pulse,
  rippleGlow,
  STACK_HALF_SIZE,
  stackDepth,
  stepsSinceCrossing,
  crossingS,
  TOKEN_POINTS,
  tokenCloud,
  tokenS,
  TOKENS,
  wordOf,
} from "./path"

// Screen space: a VIEW_UNITS square, as on the front page.
export const VIEW_UNITS = 1000
export const CENTER_UNITS = { x: 500, y: 520 }
export const WORLD_TO_UNITS = 200

// Ink levels, each a mix of two theme colors and an alpha, so JS never names a color: the canvas
// resolves the token names and the SVG writes color-mix() of the CSS variables.
export type Token = "brand" | "brand-hi" | "star" | "nebula"
export interface Ink {
  from: Token
  to: Token
  // Share of `to`, 0 to 1.
  mix: number
  alpha: number
}

export const LATTICE_LEVELS = 12
export const ORBIT_LEVELS = 6
export const HOT_LEVELS = 6
export const ORBIT_BASE = LATTICE_LEVELS
export const HOT_BASE = ORBIT_BASE + ORBIT_LEVELS
export const LEVELS = HOT_BASE + HOT_LEVELS

export function inkOf(level: number): Ink {
  if (level < ORBIT_BASE) {
    const t = level / (LATTICE_LEVELS - 1)
    return { from: "brand", to: "star", mix: round2(t ** 1.7), alpha: round2(0.08 + 0.92 * t ** 1.1) }
  }
  if (level < HOT_BASE) {
    const t = (level - ORBIT_BASE) / (ORBIT_LEVELS - 1)
    return { from: "nebula", to: "brand-hi", mix: round2(0.35 + t * 0.65), alpha: round2(0.16 + 0.74 * t) }
  }
  const t = (level - HOT_BASE) / (HOT_LEVELS - 1)
  return { from: "brand-hi", to: "star", mix: round2(t * 0.85), alpha: round2(0.55 + 0.45 * t) }
}

// The resting pose: the stack seen three-quarters from the left and a little above, so the tokens
// enter at the far left, near the headline, and leave toward the viewer on the right.
const POSE: Mat3 = multiply(rotateZ(-0.1), multiply(rotateX(0.38), rotateY(-0.8)))
const SWAY_PERIOD_S = 48
const SWAY = 0.07
const DUST_PERIOD_S = 220

// Each layer is a rounded square: a sparse lattice inside, a dense bright outline. The same on every
// screen, so the build-time frame and the canvas draw the same layers.
const SIDE = 9
const OUTLINE_POINTS = 84

// Haze and dust points. Each comes from its own seeded stream, so a smaller count is a prefix of a
// larger one.
export interface Counts {
  haze: number
  dust: number
}

export function countsFor(widthPx: number): Counts {
  if (widthPx < 720) return { haze: 260, dust: 180 }
  if (widthPx < 1200) return { haze: 420, dust: 240 }
  return { haze: 600, dust: 280 }
}

export interface SceneData {
  // Lattice: x, y, z, radius from the axis (0 to 1), edge (0 or 1), size factor; per point.
  lattice: Float32Array
  latticeCount: number
  // Haze: x, y, z, brightness.
  haze: Float32Array
  // Dust: angle, radial and vertical scatter, size.
  dust: Float32Array
  cloud: Float32Array
}

const LATTICE_STRIDE = 7
const CORNER = 0.35

// A point on a layer's outline, a square of half side 1 with rounded corners, at fraction t of the
// way round, evenly by length.
export function outlinePoint(t: number): [number, number] {
  const straight = 2 * (1 - CORNER)
  const bend = (Math.PI / 2) * CORNER
  const side = straight + bend
  const along = (((t % 1) + 1) % 1) * 4 * side
  const quarter = Math.floor(along / side)
  const rest = along - quarter * side
  // The right side, upward from its lower end, then its upper corner; turned for the other three.
  let u: number
  let v: number
  if (rest < straight) {
    u = 1
    v = -1 + CORNER + rest
  } else {
    const angle = (rest - straight) / CORNER
    u = 1 - CORNER + CORNER * Math.cos(angle)
    v = 1 - CORNER + CORNER * Math.sin(angle)
  }
  for (let turn = 0; turn < quarter; turn += 1) [u, v] = [-v, u]
  return [u, v]
}

export function makeScene(counts: Counts, seed = 0x1f0e): SceneData {
  const random = mulberry32(seed)
  const lattice: number[] = []
  for (let layer = 0; layer < LAYERS; layer += 1) {
    const x = layerX(layer)
    const depth = stackDepth(x)
    const push = (u: number, v: number, edge: number, jitter: number) => {
      lattice.push(
        x + (random() - 0.5) * jitter,
        v * STACK_HALF_SIZE + (random() - 0.5) * jitter,
        depth + u * STACK_HALF_SIZE + (random() - 0.5) * jitter,
        Math.min(1, Math.hypot(u, v) / Math.SQRT2),
        edge,
        edge ? 0.8 + random() * 0.3 : 0.7 + random() ** 2 * 0.5,
        layer
      )
    }
    for (let row = 1; row < SIDE - 1; row += 1) {
      for (let col = 1; col < SIDE - 1; col += 1) {
        push((col / (SIDE - 1)) * 2 - 1, (row / (SIDE - 1)) * 2 - 1, 0, 0.02)
      }
    }
    for (let index = 0; index < OUTLINE_POINTS; index += 1) {
      const [u, v] = outlinePoint(index / OUTLINE_POINTS)
      push(u, v, 1, 0.004)
    }
  }
  const hazeRandom = mulberry32(seed ^ 0x4a2e)
  const haze = new Float32Array(counts.haze * 4)
  for (let index = 0; index < counts.haze; index += 1) {
    const x = (hazeRandom() * 2 - 1) * 1.08
    // A beam along the axis, where the tokens pass, thinning outward.
    const angle = hazeRandom() * Math.PI * 2
    const spread = hazeRandom() ** 1.6 * STACK_HALF_SIZE * 0.55
    const y = Math.cos(angle) * spread
    const z = Math.sin(angle) * spread
    haze[index * 4] = x
    haze[index * 4 + 1] = y
    haze[index * 4 + 2] = stackDepth(x) + z
    haze[index * 4 + 3] = Math.exp(-((y * y + z * z) / 0.03))
  }
  const dustRandom = mulberry32(seed ^ 0xd057)
  const dust = new Float32Array(counts.dust * 4)
  for (let index = 0; index < counts.dust; index += 1) {
    dust[index * 4] = (index / counts.dust) * Math.PI * 2 + (dustRandom() - 0.5) * 0.02
    dust[index * 4 + 1] = (dustRandom() - 0.5) * 0.05
    dust[index * 4 + 2] = (dustRandom() - 0.5) * 0.025
    dust[index * 4 + 3] = 0.6 + dustRandom() * 0.5
  }
  return {
    lattice: Float32Array.from(lattice),
    latticeCount: lattice.length / LATTICE_STRIDE,
    haze,
    dust,
    cloud: tokenCloud(),
  }
}

export interface Label {
  x: number
  y: number
  text: string
  alpha: number
}

export interface Frame {
  x: Float32Array
  y: Float32Array
  // Point diameter in view units.
  size: Float32Array
  level: Uint8Array
  count: number
  // Token heads, for the glow drawn under them, and their words.
  heads: { x: number; y: number; heat: number }[]
  labels: Label[]
}

export function makeFrame(scene: SceneData): Frame {
  const capacity = scene.latticeCount + scene.haze.length / 4 + scene.dust.length / 4 + TOKENS * TOKEN_POINTS
  return {
    x: new Float32Array(capacity),
    y: new Float32Array(capacity),
    size: new Float32Array(capacity),
    level: new Uint8Array(capacity),
    count: 0,
    heads: [],
    labels: [],
  }
}

export interface Pose {
  timeS: number
  // The viewer's tilt from the pointer, radians.
  pitch: number
  yaw: number
}

const BASE_SIZE_UNITS = 1.9
const LAYER_CROSSINGS = Array.from({ length: LAYERS }, (_, layer) => crossingS(layerX(layer)))

function levelFor(base: number, levels: number, value: number): number {
  return base + Math.max(0, Math.min(levels - 1, Math.round(value * (levels - 1))))
}

export function worldMatrix(pose: Pose): Mat3 {
  const sway = rotateY(SWAY * Math.sin((pose.timeS / SWAY_PERIOD_S) * Math.PI * 2))
  return multiply(viewMatrix(pose.pitch, pose.yaw), multiply(POSE, sway))
}

export function composeFrame(scene: SceneData, pose: Pose, frame: Frame): void {
  const world = worldMatrix(pose)
  const beat = pulse(pose.timeS)
  let write = 0
  // Projects a world point through `world`, with no allocation: this runs for every point every frame.
  const push3 = (px: number, py: number, pz: number, sizeUnits: number, level: number) => {
    const x = world[0] * px + world[1] * py + world[2] * pz
    const y = world[3] * px + world[4] * py + world[5] * pz
    const z = world[6] * px + world[7] * py + world[8] * pz
    const scale = CAMERA_DISTANCE / (CAMERA_DISTANCE - z)
    frame.x[write] = CENTER_UNITS.x + x * WORLD_TO_UNITS * scale
    frame.y[write] = CENTER_UNITS.y - y * WORLD_TO_UNITS * scale
    frame.size[write] = sizeUnits * scale
    frame.level[write] = level
    write += 1
  }

  // The orbit's dust, turning slowly: its far half alone completes the ring the tokens ride.
  const dustTurn = (pose.timeS / DUST_PERIOD_S) * Math.PI * 2
  for (let index = 0; index < scene.dust.length / 4; index += 1) {
    const d = index * 4
    const theta = scene.dust[d] + dustTurn
    const r = ORBIT_RADIUS + scene.dust[d + 1]
    const px = r * Math.cos(theta)
    const pz = r * Math.sin(theta)
    const near = Math.max(0, (world[6] * px + world[7] * scene.dust[d + 2] + world[8] * pz) / ORBIT_RADIUS)
    push3(px, scene.dust[d + 2], pz, 2 * scene.dust[d + 3], levelFor(ORBIT_BASE, ORBIT_LEVELS, 0.1 + 0.3 * near + 0.08 * beat))
  }

  // The haze along the stack's axis, brighter while a token passes.
  for (let index = 0; index < scene.haze.length / 4; index += 1) {
    const h = index * 4
    const value = 0.05 + 0.25 * scene.haze[h + 3] * (0.6 + 0.4 * beat)
    push3(scene.haze[h], scene.haze[h + 1], scene.haze[h + 2], BASE_SIZE_UNITS * 0.7, levelFor(0, LATTICE_LEVELS, value))
  }

  // The layers: a dim lattice with bright edges, each rippling as a token passes through it.
  const glows = LAYER_CROSSINGS.map(s => stepsSinceCrossing(s, pose.timeS))
  for (let index = 0; index < scene.latticeCount; index += 1) {
    const o = index * LATTICE_STRIDE
    const radius = scene.lattice[o + 3]
    const edge = scene.lattice[o + 4]
    const glow = rippleGlow(glows[scene.lattice[o + 6]], radius)
    // Lit from the upper left, like the moon on the front page.
    const shade = 0.9 + 0.1 * (scene.lattice[o + 1] / STACK_HALF_SIZE) - 0.06 * (scene.lattice[o] / 1.05)
    const value = (0.14 + 0.7 * edge) * shade + 0.75 * glow
    const level = glow > 0.35 ? levelFor(HOT_BASE, HOT_LEVELS, (glow - 0.35) / 0.65) : levelFor(0, LATTICE_LEVELS, Math.min(1, value))
    push3(scene.lattice[o], scene.lattice[o + 1], scene.lattice[o + 2], BASE_SIZE_UNITS * scene.lattice[o + 5] * (1 + 0.6 * glow), level)
  }

  // Tokens: small bright clouds, hottest as they leave the stack, cooling round the arc.
  frame.heads.length = 0
  frame.labels.length = 0
  const { cloud } = scene
  for (let slot = 0; slot < TOKENS; slot += 1) {
    const s = tokenS(slot, pose.timeS)
    const arc = arcProgress(s)
    const inStack = arc < 0
    const sinceExit = inStack ? 1 : arc
    const heat = inStack ? 0.55 + 0.45 * ((s - ENTER_S) / (EXIT_S - ENTER_S)) : Math.max(0.3, 1 - sinceExit * 1.6)
    for (let particle = 0; particle < TOKEN_POINTS; particle += 1) {
      const p = particle * 4
      const lag = cloud[p] * (inStack ? 1.8 : 1)
      const base = loopPoint(s - lag)
      const fade = 1 - cloud[p] / 0.035
      push3(base[0], base[1] + cloud[p + 1], base[2] + cloud[p + 2], BASE_SIZE_UNITS * cloud[p + 3] * (1 + 0.5 * heat), levelFor(HOT_BASE, HOT_LEVELS, heat * (0.45 + 0.55 * fade)))
    }
    const head = apply(world, loopPoint(s))
    const scale = CAMERA_DISTANCE / (CAMERA_DISTANCE - head[2])
    const x = CENTER_UNITS.x + head[0] * WORLD_TO_UNITS * scale
    const y = CENTER_UNITS.y - head[1] * WORLD_TO_UNITS * scale
    frame.heads.push({ x, y, heat })
    // The word shows once the token is out, and fades before it goes back in.
    if (!inStack) {
      const alpha = Math.min(1, arc / 0.08, (1 - arc) / 0.22)
      if (alpha > 0.02) frame.labels.push({ x, y, text: wordOf(slot, pose.timeS).trim(), alpha: round2(alpha) })
    }
  }
  frame.count = write
}

function round2(value: number): number {
  return Math.round(value * 100) / 100
}
