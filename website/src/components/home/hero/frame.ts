// One frame of the hero scene as flat buffers of screen points, each with a size and an ink level, in
// paint order: the orbit's far side, the body, the orbit's near side. Both the build-time SVG frame
// and the canvas draw from it. Pure; the buffers are allocated once and refilled every frame.

import {
  apply,
  bodyMatrix,
  BODY_RADIUS_UNITS,
  CAMERA_DISTANCE,
  CENTER_UNITS,
  makeBody,
  multiply,
  normalize,
  orbitPoint,
  STRIDE,
  viewMatrix,
  type Mat3,
  type Vec3,
} from "./geometry"
import { mulberry32 } from "../../../lib/figure/math"
import { CYCLE_S, STREAM_POINTS, streamPosition, streamState, tileGlow, tileOf } from "./stream"

// Ink levels. Each is a mix of two theme colors and an alpha; JS never names a color, the canvas
// resolves the token names, the SVG writes them as color-mix() of the CSS variables.
export type Token = "brand" | "brand-hi" | "star"
export interface Ink {
  from: Token
  to: Token
  // Share of `to`, 0 to 1.
  mix: number
  alpha: number
}

export const BODY_LEVELS = 14
export const ORBIT_LEVELS = 6
export const HOT_LEVELS = 6
export const ORBIT_BASE = BODY_LEVELS
export const HOT_BASE = BODY_LEVELS + ORBIT_LEVELS
export const LEVELS = HOT_BASE + HOT_LEVELS

export function inkOf(level: number): Ink {
  if (level < ORBIT_BASE) {
    const t = level / (BODY_LEVELS - 1)
    return { from: "brand", to: "star", mix: round2(t ** 1.6), alpha: round2(0.05 + 0.95 * t ** 1.1) }
  }
  if (level < HOT_BASE) {
    const t = (level - ORBIT_BASE) / (ORBIT_LEVELS - 1)
    return { from: "brand", to: "brand-hi", mix: round2(t * 0.6), alpha: round2(0.16 + 0.74 * t) }
  }
  const t = (level - HOT_BASE) / (HOT_LEVELS - 1)
  return { from: "brand-hi", to: "star", mix: round2(t * 0.85), alpha: round2(0.55 + 0.45 * t) }
}

// "emit": points are light on a dark sky. "stipple": ink on a pale sky, so shadow is denser.
export type InkMode = "emit" | "stipple"

// Light from the left and a little above and in front, toward the headline.
export const LIGHT: Vec3 = normalize([-0.74, 0.4, 0.56])
const SPIN_PERIOD_S = 96
// The opening pose shows Stickney toward the viewer.
const SPIN_START = -0.35
const ORBIT_PERIOD_S = 54

export const DUST_POINTS = 240
export const PACKETS = 5
export const PACKET_POINTS = 26
const EXTRA_POINTS = DUST_POINTS + PACKETS * PACKET_POINTS + STREAM_POINTS

export interface SceneData {
  body: Float32Array
  bodyCount: number
  // Body point index of each stream slot, and the slot of each body point (-1 for none).
  slotPoint: Int32Array
  pointSlot: Int16Array
  dust: Float32Array
  packets: Float32Array
}

export function makeScene(bodyCount: number, seed = 0x0b05): SceneData {
  const body = makeBody(bodyCount, seed)
  const count = body.length / STRIDE
  const slotPoint = new Int32Array(STREAM_POINTS)
  const pointSlot = new Int16Array(count).fill(-1)
  const stride = count / STREAM_POINTS
  for (let slot = 0; slot < STREAM_POINTS; slot += 1) {
    const index = Math.floor(slot * stride)
    slotPoint[slot] = index
    pointSlot[index] = slot
  }
  const random = mulberry32(seed ^ 0x5eed)
  // Dust: base angle, radial and vertical scatter, size.
  const dust = new Float32Array(DUST_POINTS * 4)
  for (let index = 0; index < DUST_POINTS; index += 1) {
    dust[index * 4] = (index / DUST_POINTS) * Math.PI * 2 + (random() - 0.5) * 0.02
    dust[index * 4 + 1] = (random() - 0.5) * 0.035
    dust[index * 4 + 2] = (random() - 0.5) * 0.02
    dust[index * 4 + 3] = 0.6 + random() * 0.5
  }
  // Packet particles: lag behind the head (radians), radial and vertical offset, sway phase.
  const packets = new Float32Array(PACKETS * PACKET_POINTS * 4)
  for (let index = 0; index < PACKETS * PACKET_POINTS; index += 1) {
    const angle = random() * Math.PI * 2
    const spread = Math.sqrt(random()) * 0.07
    packets[index * 4] = random() ** 1.7 * 0.5
    packets[index * 4 + 1] = Math.cos(angle) * spread
    packets[index * 4 + 2] = Math.sin(angle) * spread * 0.7
    packets[index * 4 + 3] = random() * Math.PI * 2
  }
  return { body, bodyCount: count, slotPoint, pointSlot, dust, packets }
}

export interface Frame {
  x: Float32Array
  y: Float32Array
  // Point diameter in view units.
  size: Float32Array
  level: Uint8Array
  count: number
  // Points before farEnd are behind the body, before bodyEnd on it, the rest in front.
  farEnd: number
  bodyEnd: number
  // The body's silhouette, for the occluder painted under it.
  silhouette: { x: number; y: number; rx: number; ry: number; angle: number }
}

export function makeFrame(scene: SceneData): Frame {
  const capacity = scene.bodyCount + EXTRA_POINTS
  return {
    x: new Float32Array(capacity),
    y: new Float32Array(capacity),
    size: new Float32Array(capacity),
    level: new Uint8Array(capacity),
    count: 0,
    farEnd: 0,
    bodyEnd: 0,
    silhouette: { x: CENTER_UNITS.x, y: CENTER_UNITS.y, rx: BODY_RADIUS_UNITS, ry: BODY_RADIUS_UNITS, angle: 0 },
  }
}

export interface Pose {
  timeS: number
  // The viewer's tilt from the pointer, radians.
  pitch: number
  yaw: number
}

// Scratch for the orbit and stream points, which split into far and near halves.
interface Extra {
  x: Float32Array
  y: Float32Array
  z: Float32Array
  size: Float32Array
  level: Uint8Array
  count: number
}

const extras = new WeakMap<Frame, Extra>()

function extraOf(frame: Frame): Extra {
  let extra = extras.get(frame)
  if (!extra) {
    extra = {
      x: new Float32Array(EXTRA_POINTS),
      y: new Float32Array(EXTRA_POINTS),
      z: new Float32Array(EXTRA_POINTS),
      size: new Float32Array(EXTRA_POINTS),
      level: new Uint8Array(EXTRA_POINTS),
      count: 0,
    }
    extras.set(frame, extra)
  }
  return extra
}

const BASE_SIZE_UNITS = 1.9
const ORBIT_SIZE_UNITS = 2

function levelFor(base: number, levels: number, value: number): number {
  return base + Math.max(0, Math.min(levels - 1, Math.round(value * (levels - 1))))
}

export function spinAt(timeS: number): number {
  return SPIN_START + (timeS / SPIN_PERIOD_S) * Math.PI * 2
}

export function composeFrame(scene: SceneData, pose: Pose, mode: InkMode, frame: Frame): void {
  const view = viewMatrix(pose.pitch, pose.yaw)
  const world: Mat3 = multiply(view, bodyMatrix(spinAt(pose.timeS)))
  const light = LIGHT
  const cycleS = ((pose.timeS % CYCLE_S) + CYCLE_S) % CYCLE_S
  const extra = extraOf(frame)
  extra.count = 0

  const pushExtra = (point: Vec3, sizeUnits: number, level: number) => {
    const scale = CAMERA_DISTANCE / (CAMERA_DISTANCE - point[2])
    const index = extra.count
    extra.x[index] = CENTER_UNITS.x + point[0] * BODY_RADIUS_UNITS * scale
    extra.y[index] = CENTER_UNITS.y - point[1] * BODY_RADIUS_UNITS * scale
    extra.z[index] = point[2]
    extra.size[index] = sizeUnits * scale
    extra.level[index] = level
    extra.count += 1
  }

  // The orbit's dust: a dotted ring turning slowly, brighter on its near side.
  const dustTurn = (pose.timeS / (ORBIT_PERIOD_S * 4)) * Math.PI * 2
  for (let index = 0; index < DUST_POINTS; index += 1) {
    const d = index * 4
    const theta = scene.dust[d] + dustTurn
    const point = apply(view, orbitPoint(theta, scene.dust[d + 1], scene.dust[d + 2]))
    const near = Math.max(0, point[2] / 2)
    pushExtra(point, ORBIT_SIZE_UNITS * scene.dust[d + 3], levelFor(ORBIT_BASE, ORBIT_LEVELS, 0.12 + 0.35 * near))
  }

  // Packets: small clouds riding the orbit, strung out behind their heads.
  const flow = (pose.timeS / ORBIT_PERIOD_S) * Math.PI * 2
  for (let packet = 0; packet < PACKETS; packet += 1) {
    const head = (packet / PACKETS) * Math.PI * 2 + flow
    for (let particle = 0; particle < PACKET_POINTS; particle += 1) {
      const p = (packet * PACKET_POINTS + particle) * 4
      const lag = scene.packets[p]
      const sway = Math.sin(head * 3 + scene.packets[p + 3]) * 0.012
      const point = apply(view, orbitPoint(head - lag, scene.packets[p + 1] + sway, scene.packets[p + 2]))
      const fade = 1 - lag / 0.5
      const near = Math.max(0, point[2] / 2)
      pushExtra(point, ORBIT_SIZE_UNITS * (1.1 + fade * 0.6), levelFor(ORBIT_BASE, ORBIT_LEVELS, 0.25 + 0.55 * fade + 0.2 * near))
    }
  }

  // The body, with its streaming points drawn as part of the orbit layer while they are away.
  const { body, bodyCount, pointSlot } = scene
  const out = frame
  let bodyWrite = 0
  // Body points go after the far orbit points, which are not counted yet; write them to the tail and
  // move them once the far count is known.
  const tail = out.x.length - bodyCount
  for (let index = 0; index < bodyCount; index += 1) {
    const o = index * STRIDE
    const px = body[o]
    const py = body[o + 1]
    const pz = body[o + 2]
    const wx = world[0] * px + world[1] * py + world[2] * pz
    const wy = world[3] * px + world[4] * py + world[5] * pz
    const wz = world[6] * px + world[7] * py + world[8] * pz
    const nx = world[0] * body[o + 3] + world[1] * body[o + 4] + world[2] * body[o + 5]
    const ny = world[3] * body[o + 3] + world[4] * body[o + 4] + world[5] * body[o + 5]
    const nz = world[6] * body[o + 3] + world[7] * body[o + 4] + world[8] * body[o + 5]

    const slot = pointSlot[index]
    if (slot >= 0) {
      const state = streamState(slot, cycleS)
      const at = streamPosition(slot, state, [wx, wy, wz], [nx, ny, nz])
      if (at) {
        const glow = state.phase === "hold" ? tileGlow(tileOf(slot), state.holdS) : 0
        const away = state.phase === "lift" ? state.t : state.phase === "sink" ? 1 - state.t : 1
        const level =
          glow > 0.08
            ? levelFor(HOT_BASE, HOT_LEVELS, 0.35 + 0.65 * glow)
            : state.phase === "hold" || state.phase === "gather" || state.phase === "depart"
              ? levelFor(HOT_BASE, HOT_LEVELS, 0.05)
              : levelFor(ORBIT_BASE, ORBIT_LEVELS, 0.45 + 0.55 * away)
        pushExtra(at, BASE_SIZE_UNITS * (1 + 0.35 * away + 0.5 * glow), level)
        continue
      }
    }

    // Cull the far side, softly: the radial direction says which way the point faces.
    const radius = Math.hypot(wx, wy, wz) || 1
    const facing = wz / radius
    const fade = smoothstep(-0.3, 0.15, facing)
    if (fade < 0.03) continue
    const lambert = Math.max(0, nx * light[0] + ny * light[1] + nz * light[2])
    const rim = (1 - Math.max(0, nz)) ** 3
    const albedo = body[o + 6]
    const value =
      mode === "emit"
        ? ((0.07 + 1.2 * lambert ** 1.1) * albedo + rim * 0.06) * fade
        : (0.1 + 0.8 * (1 - lambert) ** 1.2 * albedo + rim * 0.08) * fade
    const scale = CAMERA_DISTANCE / (CAMERA_DISTANCE - wz)
    const write = tail + bodyWrite
    out.x[write] = CENTER_UNITS.x + wx * BODY_RADIUS_UNITS * scale
    out.y[write] = CENTER_UNITS.y - wy * BODY_RADIUS_UNITS * scale
    out.size[write] = BASE_SIZE_UNITS * body[o + 7] * scale * (0.8 + 0.2 * fade) * (0.85 + 0.45 * Math.min(1, value))
    out.level[write] = levelFor(0, BODY_LEVELS, Math.min(1, value))
    bodyWrite += 1
  }

  // The silhouette: the long axis's projection gives its length and angle.
  const longAxis = apply(world, [1, 0, 0])
  const reach = Math.hypot(longAxis[0], longAxis[1])
  const silhouette = out.silhouette
  silhouette.x = CENTER_UNITS.x
  silhouette.y = CENTER_UNITS.y
  silhouette.rx = BODY_RADIUS_UNITS * Math.max(0.8, reach) * 0.96
  silhouette.ry = BODY_RADIUS_UNITS * 0.74
  silhouette.angle = Math.atan2(-longAxis[1], longAxis[0])

  // Far orbit points, hidden where the body covers them.
  let write = 0
  const cos = Math.cos(silhouette.angle)
  const sin = Math.sin(silhouette.angle)
  for (let index = 0; index < extra.count; index += 1) {
    if (extra.z[index] >= 0) continue
    const dx = extra.x[index] - silhouette.x
    const dy = extra.y[index] - silhouette.y
    const along = (dx * cos + dy * sin) / silhouette.rx
    const across = (-dx * sin + dy * cos) / silhouette.ry
    if (along * along + across * across < 0.92) continue
    out.x[write] = extra.x[index]
    out.y[write] = extra.y[index]
    out.size[write] = extra.size[index]
    out.level[write] = extra.level[index]
    write += 1
  }
  out.farEnd = write
  out.x.copyWithin(write, tail, tail + bodyWrite)
  out.y.copyWithin(write, tail, tail + bodyWrite)
  out.size.copyWithin(write, tail, tail + bodyWrite)
  out.level.copyWithin(write, tail, tail + bodyWrite)
  write += bodyWrite
  out.bodyEnd = write
  for (let index = 0; index < extra.count; index += 1) {
    if (extra.z[index] < 0) continue
    out.x[write] = extra.x[index]
    out.y[write] = extra.y[index]
    out.size[write] = extra.size[index]
    out.level[write] = extra.level[index]
    write += 1
  }
  out.count = write
}

function smoothstep(edge0: number, edge1: number, value: number): number {
  const t = Math.max(0, Math.min(1, (value - edge0) / (edge1 - edge0)))
  return t * t * (3 - 2 * t)
}

function round2(value: number): number {
  return Math.round(value * 100) / 100
}
