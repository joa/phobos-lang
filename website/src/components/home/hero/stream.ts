// The loop's one event: a few hundred points lift off the moon, ride the orbit's near side, settle into
// a grid of tiles that lights up tile by tile, then ride the far side home and sink back into the
// surface. Each point's place is a pure function of the cycle time, and at the start and end of a
// cycle every point is on the surface, so the loop has no seam.

import { easeInOutSine } from "../../../lib/figure/math"
import { apply, multiply, normalize, orbitPoint, rotateX, rotateY, type Vec3 } from "./geometry"

export const CYCLE_S = 26
export const TILES_PER_SIDE = 4
export const CELLS_PER_SIDE = 4
export const STREAM_POINTS = (TILES_PER_SIDE * CELLS_PER_SIDE) ** 2

// Phase durations in seconds, in order. The event starts EVENT_START_S into the cycle.
export const PHASES_S = { lift: 1.9, ride: 3.1, gather: 1.7, hold: 3.8, depart: 1.6, rideBack: 2.6, sink: 2 } as const
export type PhaseName = keyof typeof PHASES_S | "rest"
const PHASE_ORDER = Object.keys(PHASES_S) as (keyof typeof PHASES_S)[]
const EVENT_S = PHASE_ORDER.reduce((sum, name) => sum + PHASES_S[name], 0)
export const EVENT_START_S = 2.5
// Tiles leave one after another; a tile's cells nearly together.
const TILE_STAGGER_S = 0.11
const CELL_STAGGER_S = 0.012
export const MAX_DELAY_S = (TILES_PER_SIDE ** 2 - 1) * TILE_STAGGER_S + (CELLS_PER_SIDE ** 2 - 1) * CELL_STAGGER_S

// Orbit angles: joining on the near right, across the near side to the grid by the left end, home
// round the far side.
const THETA_JOIN = 0.55
const THETA_GRID = Math.PI - 0.1
const THETA_SINK = Math.PI + 1.6

// The grid floats above the orbit's left end, between the headline and the body, turned toward it.
export const GRID_CENTER: Vec3 = [-1.0, 1.72, 0.3]
const GRID_FRAME = multiply(rotateY(0.36), rotateX(-0.16))
const CELL_PITCH = 0.05
const TILE_GAP = 0.115
const TILE_PITCH = (CELLS_PER_SIDE - 1) * CELL_PITCH + TILE_GAP

export interface StreamState {
  phase: PhaseName
  // Progress through the phase, 0 to 1, already eased.
  t: number
  // Seconds into the hold, for the tile sweep; 0 elsewhere.
  holdS: number
}

export function tileOf(slot: number): number {
  return Math.floor(slot / CELLS_PER_SIDE ** 2)
}

export function slotDelayS(slot: number): number {
  return tileOf(slot) * TILE_STAGGER_S + (slot % CELLS_PER_SIDE ** 2) * CELL_STAGGER_S
}

export function streamState(slot: number, cycleS: number): StreamState {
  let local = cycleS - EVENT_START_S - slotDelayS(slot)
  if (local <= 0 || local >= EVENT_S) return { phase: "rest", t: 0, holdS: 0 }
  for (const name of PHASE_ORDER) {
    const durationS = PHASES_S[name]
    if (local < durationS) {
      return { phase: name, t: easeInOutSine(local / durationS), holdS: name === "hold" ? local : 0 }
    }
    local -= durationS
  }
  return { phase: "rest", t: 0, holdS: 0 }
}

// The slot's cell in the grid, in world units.
export function gridSlot(slot: number): Vec3 {
  const tile = tileOf(slot)
  const cell = slot % CELLS_PER_SIDE ** 2
  const half = ((TILES_PER_SIDE - 1) * TILE_PITCH + (CELLS_PER_SIDE - 1) * CELL_PITCH) / 2
  const u = (tile % TILES_PER_SIDE) * TILE_PITCH + (cell % CELLS_PER_SIDE) * CELL_PITCH - half
  const v = half - (Math.floor(tile / TILES_PER_SIDE) * TILE_PITCH + Math.floor(cell / CELLS_PER_SIDE) * CELL_PITCH)
  const offset = apply(GRID_FRAME, [u, v, 0])
  return [GRID_CENTER[0] + offset[0], GRID_CENTER[1] + offset[1], GRID_CENTER[2] + offset[2]]
}

// How brightly a tile glows during the hold: a sweep in raster order, one tile after another.
export function tileGlow(tile: number, holdS: number): number {
  const peakS = 0.5 + tile * 0.17
  return Math.exp(-(((holdS - peakS) / 0.28) ** 2))
}

// Each slot's own scatter around the orbit, from its index, so the stream reads as a cloud.
export function slotJitter(slot: number): { join: number; grid: number; sink: number; radial: number; lift: number } {
  const hash = (k: number) => {
    const x = Math.sin(slot * 12.9898 + k * 78.233) * 43758.5453
    return x - Math.floor(x) - 0.5
  }
  return {
    join: hash(1) * 0.35,
    grid: hash(2) * 0.12,
    sink: hash(3) * 0.4,
    radial: hash(4) * 0.09,
    lift: hash(5) * 0.07,
  }
}

function mix(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

// Where a streaming point is: `surface` is its place on the turning body now, `outward` the way off it.
// Returns null when the point is at rest on the surface.
export function streamPosition(slot: number, state: StreamState, surface: Vec3, outward: Vec3): Vec3 | null {
  if (state.phase === "rest") return null
  const jitter = slotJitter(slot)
  const along = (theta: number) => orbitPoint(theta, jitter.radial, jitter.lift)
  const join = THETA_JOIN + jitter.join
  const grid = THETA_GRID + jitter.grid
  const sink = THETA_SINK + jitter.sink
  const { t } = state
  const arc = Math.sin(Math.PI * t)
  switch (state.phase) {
    case "lift": {
      const p = mix(surface, along(join), t)
      const up = normalize(outward)
      return [p[0] + up[0] * arc * 0.35, p[1] + up[1] * arc * 0.35, p[2] + up[2] * arc * 0.35]
    }
    case "ride":
      return along(join + (grid - join) * t)
    case "gather": {
      const p = mix(along(grid), gridSlot(slot), t)
      return [p[0] - arc * 0.12, p[1], p[2] + arc * 0.1]
    }
    case "hold": {
      const p = gridSlot(slot)
      return [p[0], p[1] + Math.sin(state.holdS * 1.6 + slot * 0.05) * 0.008, p[2]]
    }
    case "depart": {
      const p = mix(gridSlot(slot), along(grid), t)
      return [p[0] - arc * 0.12, p[1], p[2] - arc * 0.1]
    }
    case "rideBack":
      return along(grid + (sink - grid) * t)
    case "sink": {
      const p = mix(along(sink), surface, t)
      const up = normalize(outward)
      return [p[0] + up[0] * arc * 0.3, p[1] + up[1] * arc * 0.3, p[2] + up[2] * arc * 0.3]
    }
  }
}
