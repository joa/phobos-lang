// The token loop of the /inference/ hero, in world units (the stack's half length is about 1): a closed
// path that runs straight through the layer stack, left to right, and comes back round the orbit's
// near side. Tokens ride it in step: every STEP_S one token passes through the stack, the decode
// step, and the train surges and eases with it. Everything here is a pure function of time, so the
// build-time frame and the canvas agree and the loop has no seam.

import { mulberry32 } from "../../../lib/figure/math"
import type { Vec3 } from "../../home/hero/geometry"

export const ORBIT_RADIUS = 1.95
// The far half of the token path is flattened to this share of the radius, so it runs nearly straight
// through the stack, which sits on it.
export const FLAT = 0.11
export const STACK_HALF_LENGTH = 1.05
export const STACK_HALF_SIZE = 0.56
export const LAYERS = 7
// Tokens on the loop at once.
export const TOKENS = 5
export const STEP_S = 1.9
// How strongly the train surges within a step, 0 for an even flow, below 1 so it never stops.
const SURGE = 0.72

// The answer the loop generates, a token at a time. Leading spaces mark word starts, as a tokenizer
// would.
export const WORDS = [
  " Because",
  " the",
  " universe",
  " is",
  " young",
  " and",
  " expanding",
  ",",
  " so",
  " light",
  " from",
  " distant",
  " stars",
  " has",
  " not",
  " reached",
  " us",
  ".",
] as const

// A turn of the token path, angle phi: 0 at the right end, PI/2 nearest the viewer, PI at the left
// end, then the flattened far half back through the stack.
export function pathPoint(phi: number): Vec3 {
  const sin = Math.sin(phi)
  return [ORBIT_RADIUS * Math.cos(phi), 0, ORBIT_RADIUS * sin * (sin < 0 ? FLAT : 1)]
}

// The stack's axis runs along x at the depth the flattened half crosses it.
export function stackDepth(x: number): number {
  const c = Math.max(-1, Math.min(1, x / ORBIT_RADIUS))
  return -ORBIT_RADIUS * FLAT * Math.sqrt(1 - c * c)
}

export function layerX(layer: number): number {
  return -STACK_HALF_LENGTH + (2 * STACK_HALF_LENGTH * layer) / (LAYERS - 1)
}

// Arc length table, so tokens move at an even speed along the path.
const SAMPLES = 720
const table = (() => {
  const lengths = new Float64Array(SAMPLES + 1)
  let previous = pathPoint(Math.PI)
  for (let index = 1; index <= SAMPLES; index += 1) {
    const point = pathPoint(Math.PI + (index / SAMPLES) * Math.PI * 2)
    lengths[index] = lengths[index - 1] + Math.hypot(point[0] - previous[0], point[2] - previous[2])
    previous = point
  }
  return lengths
})()
export const PATH_LENGTH = table[SAMPLES]

// The point at arc fraction s of the loop, s = 0 at the left end, where tokens enter the stack.
export function loopPoint(s: number): Vec3 {
  const target = (((s % 1) + 1) % 1) * PATH_LENGTH
  let low = 0
  let high = SAMPLES
  while (high - low > 1) {
    const middle = (low + high) >> 1
    if (table[middle] <= target) low = middle
    else high = middle
  }
  const span = table[high] - table[low] || 1
  const index = low + (target - table[low]) / span
  return pathPoint(Math.PI + (index / SAMPLES) * Math.PI * 2)
}

// The arc fraction at which the path crosses x on its way through the stack.
export function crossingS(x: number): number {
  const phi = Math.PI + Math.acos(Math.max(-1, Math.min(1, -x / ORBIT_RADIUS)))
  const at = ((phi - Math.PI) / (Math.PI * 2)) * SAMPLES
  const low = Math.floor(at)
  const high = Math.min(SAMPLES, low + 1)
  return (table[low] + (table[high] - table[low]) * (at - low)) / PATH_LENGTH
}

export const ENTER_S = crossingS(-STACK_HALF_LENGTH)
export const EXIT_S = crossingS(STACK_HALF_LENGTH)

// Steps taken by time t: rises by one every STEP_S, fastest mid-step. Monotonic.
export function steps(timeS: number): number {
  const u = timeS / STEP_S
  return u - (SURGE * Math.sin(Math.PI * 2 * u)) / (Math.PI * 2)
}

// The pulse of a step, 1 as the train moves fastest, 0 at the turn of a step.
export function pulse(timeS: number): number {
  return (1 - Math.cos(Math.PI * 2 * (timeS / STEP_S))) / 2
}

// Token `slot`'s head on the loop.
export function tokenS(slot: number, timeS: number): number {
  const s = (steps(timeS) + slot) / TOKENS
  return s - Math.floor(s)
}

// The word a token carries: it changes as the token leaves the stack, and the words leave in order.
export function wordOf(slot: number, timeS: number): string {
  const sinceExit = steps(timeS) + slot - TOKENS * EXIT_S
  const lap = Math.floor(sinceExit / TOKENS)
  const index = lap * TOKENS - slot
  return WORDS[((index % WORDS.length) + WORDS.length) % WORDS.length]
}

// Steps since a token last crossed arc fraction s; tokens are TOKENS apart, so one crosses every step.
export function stepsSinceCrossing(s: number, timeS: number): number {
  const since = steps(timeS) - TOKENS * s
  return since - Math.floor(since)
}

// How brightly a layer point glows at radius r (0 at the axis, 1 at the edge) after a token passed
// through: a flash at the axis and a ring running out to the edge.
export function rippleGlow(stepsSince: number, r: number): number {
  const fade = (1 - stepsSince) ** 2
  const ringR = stepsSince * 3
  const ring = Math.exp(-(((r - ringR) / 0.18) ** 2))
  const flash = Math.exp(-stepsSince * 9) * Math.exp(-((r / 0.35) ** 2))
  return Math.min(1, ring * fade + flash)
}

// The share of the loop a token is on the return arc, 0 as it leaves the stack and 1 as it re-enters.
export function arcProgress(s: number): number {
  if (s >= ENTER_S && s <= EXIT_S) return -1
  const span = 1 - EXIT_S + ENTER_S
  return (s > EXIT_S ? s - EXIT_S : s + 1 - EXIT_S) / span
}

// A seeded scatter for each token's particles: lag behind the head (arc fraction), offsets across.
export const TOKEN_POINTS = 40
export function tokenCloud(seed = 0x70ce): Float32Array {
  const random = mulberry32(seed)
  const cloud = new Float32Array(TOKEN_POINTS * 4)
  for (let index = 0; index < TOKEN_POINTS; index += 1) {
    const angle = random() * Math.PI * 2
    const spread = Math.sqrt(random()) * 0.06
    cloud[index * 4] = random() ** 1.8 * 0.035
    cloud[index * 4 + 1] = Math.cos(angle) * spread
    cloud[index * 4 + 2] = Math.sin(angle) * spread
    cloud[index * 4 + 3] = 0.7 + random() * 0.7
  }
  return cloud
}
