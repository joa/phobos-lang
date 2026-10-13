// The hero's moon as a point cloud: a lumpy ellipsoid with craters (Stickney the largest), sampled
// with a seeded generator, so the build-time frame and the live canvas draw the same body. Pure, so it
// runs in Node, in tests and in the browser alike.
//
// Units: the body's long semi-axis is 1. World axes: x right, y up, z toward the viewer. Screen
// coordinates are in a VIEW_UNITS square viewBox, y down.

import { mulberry32 } from "../../../lib/figure/math"

export type Vec3 = [number, number, number]
// Row-major 3 x 3.
export type Mat3 = [number, number, number, number, number, number, number, number, number]

export const VIEW_UNITS = 1000
export const CENTER_UNITS = { x: 500, y: 500 }
// The body's long semi-axis on screen at depth 0.
export const BODY_RADIUS_UNITS = 170
// Phobos is about 27 x 22 x 18 km.
export const SEMI_AXES: Vec3 = [1, 0.82, 0.68]
export const CAMERA_DISTANCE = 5.5

// One body point: position, outward normal, albedo and a size factor.
export const STRIDE = 8

export interface Crater {
  direction: Vec3
  // Angular radius on the unit sphere, radians.
  radius: number
  // Depth and rim height, as fractions of the local radius.
  depth: number
  rim: number
}

export interface Relief {
  height: number
  // 1 at a crater's center falling to 0 near its wall; the largest over all craters.
  floor: number
  // 1 on a rim crest.
  rim: number
}

export function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / length, v[1] / length, v[2] / length]
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

export function multiply(a: Mat3, b: Mat3): Mat3 {
  const out = new Array(9).fill(0) as Mat3
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      out[row * 3 + col] = a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col]
    }
  }
  return out
}

export function apply(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ]
}

export function rotateX(angle: number): Mat3 {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  return [1, 0, 0, 0, c, -s, 0, s, c]
}

export function rotateY(angle: number): Mat3 {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  return [c, 0, s, 0, 1, 0, -s, 0, c]
}

export function rotateZ(angle: number): Mat3 {
  const c = Math.cos(angle)
  const s = Math.sin(angle)
  return [c, -s, 0, s, c, 0, 0, 0, 1]
}

// The body spins about its short axis, which stands upright in the world, leaning toward the viewer.
const BODY_TO_WORLD: Mat3 = [1, 0, 0, 0, 0, 1, 0, -1, 0]
export const BODY_LEAN = { pitch: 0.42, roll: -0.22 }

export function bodyMatrix(spin: number): Mat3 {
  return multiply(rotateZ(BODY_LEAN.roll), multiply(rotateX(BODY_LEAN.pitch), multiply(rotateY(spin), BODY_TO_WORLD)))
}

// The viewer's tilt, from the pointer: applied to everything in the world.
export function viewMatrix(pitch: number, yaw: number): Mat3 {
  return multiply(rotateX(pitch), rotateY(yaw))
}

export interface Projected {
  x: number
  y: number
  // Perspective scale: 1 at depth 0, larger nearer the viewer.
  scale: number
}

export function project(point: Vec3): Projected {
  const scale = CAMERA_DISTANCE / (CAMERA_DISTANCE - point[2])
  return {
    x: CENTER_UNITS.x + point[0] * BODY_RADIUS_UNITS * scale,
    y: CENTER_UNITS.y - point[1] * BODY_RADIUS_UNITS * scale,
    scale,
  }
}

export function ellipsoidRadius(direction: Vec3): number {
  const [a, b, c] = SEMI_AXES
  return 1 / Math.sqrt((direction[0] / a) ** 2 + (direction[1] / b) ** 2 + (direction[2] / c) ** 2)
}

// Stickney sits near the end of the long axis, facing the viewer in the opening frame.
export const STICKNEY: Crater = { direction: normalize([0.9, -0.22, 0.38]), radius: 0.5, depth: 0.15, rim: 0.045 }

export function makeCraters(seed = 0x57c4e7): Crater[] {
  const random = mulberry32(seed)
  const craters = [STICKNEY]
  for (let index = 0; index < 30; index += 1) {
    const radius = 0.05 + random() ** 2 * 0.2
    craters.push({ direction: randomDirection(random), radius, depth: radius * 0.5, rim: radius * 0.16 })
  }
  return craters
}

export function relief(direction: Vec3, craters: readonly Crater[]): Relief {
  let height = 0
  let floor = 0
  let rim = 0
  for (const crater of craters) {
    const cosine = dot(direction, crater.direction)
    if (cosine < Math.cos(Math.min(Math.PI, crater.radius * 1.7))) continue
    const r = Math.acos(Math.min(1, cosine)) / crater.radius
    if (r < 1) height -= crater.depth * (1 - r * r)
    const crest = Math.exp(-(((r - 1) / 0.2) ** 2))
    height += crater.rim * crest
    floor = Math.max(floor, r < 0.9 ? 1 - r / 0.9 : 0)
    rim = Math.max(rim, crest)
  }
  return { height, floor, rim }
}

// Low-frequency lumps that make the ellipsoid a potato.
export function lumps(d: Vec3): number {
  return (
    0.06 * Math.sin(2.7 * d[0] + 1.3) * Math.sin(2.1 * d[1] - 0.6) +
    0.035 * Math.sin(3.9 * d[2] + 2.2 * d[0] + 0.4) +
    0.02 * Math.sin(6.1 * d[1] + 5.3 * d[2])
  )
}

export function surfaceRadius(direction: Vec3, craters: readonly Crater[]): number {
  return ellipsoidRadius(direction) * (1 + lumps(direction) + relief(direction, craters).height)
}

function randomDirection(random: () => number): Vec3 {
  const z = 2 * random() - 1
  const phi = 2 * Math.PI * random()
  const s = Math.sqrt(1 - z * z)
  return [s * Math.cos(phi), s * Math.sin(phi), z]
}

// Grooves: Phobos's parallel striations, here a faint density banding across the body.
const GROOVE_AXIS = normalize([0.25, 0.95, -0.2])
const NORMAL_STEP = 0.012

// `count` body points, STRIDE floats each, deterministic for a seed. Density follows the relief:
// sparse crater floors, dense rims, faint grooves.
export function makeBody(count: number, seed = 0x0b05, craters = makeCraters()): Float32Array {
  const random = mulberry32(seed)
  const points = new Float32Array(count * STRIDE)
  let filled = 0
  let attempts = 0
  while (filled < count && attempts < count * 40) {
    attempts += 1
    const d = randomDirection(random)
    const shape = relief(d, craters)
    const radius = ellipsoidRadius(d) * (1 + lumps(d) + shape.height)
    let density = (0.72 + 3 * lumps(d)) * (1 - 0.8 * shape.floor) + 0.75 * shape.rim
    density *= 0.88 + 0.12 * Math.cos(34 * dot(d, GROOVE_AXIS))
    // Uniform directions crowd where the body is narrow; weigh by area.
    density *= radius * radius
    if (random() > density) continue

    const normal = surfaceNormal(d, radius, craters)
    const albedo = (0.78 + 0.22 * random()) * (1 - 0.45 * shape.floor) * (1 + 0.35 * shape.rim)
    const offset = filled * STRIDE
    points[offset] = d[0] * radius
    points[offset + 1] = d[1] * radius
    points[offset + 2] = d[2] * radius
    points[offset + 3] = normal[0]
    points[offset + 4] = normal[1]
    points[offset + 5] = normal[2]
    points[offset + 6] = Math.min(1.2, albedo)
    points[offset + 7] = 0.7 + random() ** 2 * 0.9
    filled += 1
  }
  return filled === count ? points : points.slice(0, filled * STRIDE)
}

function surfaceNormal(d: Vec3, radius: number, craters: readonly Crater[]): Vec3 {
  const helper: Vec3 = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]
  const t1 = normalize(cross(d, helper))
  const t2 = cross(d, t1)
  const at = (t: Vec3): Vec3 => {
    const direction = normalize([d[0] + t[0] * NORMAL_STEP, d[1] + t[1] * NORMAL_STEP, d[2] + t[2] * NORMAL_STEP])
    const r = surfaceRadius(direction, craters)
    return [direction[0] * r, direction[1] * r, direction[2] * r]
  }
  const p: Vec3 = [d[0] * radius, d[1] * radius, d[2] * radius]
  const a = at(t1)
  const b = at(t2)
  const normal = normalize(cross([a[0] - p[0], a[1] - p[1], a[2] - p[2]], [b[0] - p[0], b[1] - p[1], b[2] - p[2]]))
  return dot(normal, d) < 0 ? [-normal[0], -normal[1], -normal[2]] : normal
}

// The orbit: a circle in a plane tipped toward the viewer, so its near half runs below and in front
// of the body and its far half above and behind.
export const ORBIT = { radius: 1.95, incline: 0.3, roll: -0.16 }
const ORBIT_FRAME = multiply(rotateZ(ORBIT.roll), rotateX(ORBIT.incline))

// A point on the orbit at an angle, pushed out by `radial` and up out of the plane by `lift`.
export function orbitPoint(theta: number, radial = 0, lift = 0): Vec3 {
  const r = ORBIT.radius + radial
  return apply(ORBIT_FRAME, [r * Math.cos(theta), lift, r * Math.sin(theta)])
}
