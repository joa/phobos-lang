// The three beats of the front page's tiles-and-lanes scene, as pure functions of time. A beat lists
// the elements it draws (dot layers, outlines, a pool of particles, labels), and frame(t) says where
// each one is: positions in viewBox units, opacities, the code line to highlight. The build renders
// a settled frame into the page; the browser calls the same functions every frame.
//
// Beat 1, tiles: three floors stacked in depth, global memory, shared memory and registers. A tile
// of A and one of B lift out of global memory as point clouds into shared memory, then flow into the
// accumulator, one k step after another. Beat 2, lanes: one warp's 32 lanes in a row. Each lane loads
// its own run of W and x, then the partial sums cross between lanes over five xor stages. Beat 3,
// one kernel at both levels, is in mixed.ts and draws with the helpers exported here.
//
// World units are tensor elements; one dot of a matrix stands for STRIDE x STRIDE of them.

import { clamp, easeInOutSine, mulberry32 } from "../../../lib/figure/math"
import { bezier3, fitView, project, vec, type Box, type Camera, type Vec3, type View } from "./camera"
import type { LanesData, TilesData } from "./data"

export const VIEW_WIDTH_UNITS = 600
export const VIEW_HEIGHT_UNITS = 400
// Floor labels sit left of their floors, so the drawing leaves them a margin.
const LABEL_MARGIN_UNITS = 128
export const FIT_BOX: Box = {
  x: LABEL_MARGIN_UNITS,
  y: 20,
  width: VIEW_WIDTH_UNITS - LABEL_MARGIN_UNITS - 14,
  height: VIEW_HEIGHT_UNITS - 40,
}

export type BeatName = "tiles" | "lanes" | "mixed"
export type Space = "global" | "shared" | "reg"

export interface Shot {
  camera: Camera
  view: View
}

export interface PathSpec {
  key: string
  className: string
}

export interface ParticleSpec {
  space: Space
  depth: 0 | 1 | 2
}

export interface LabelSpec {
  key: string
  text: string
  // A second, quieter line.
  detail?: string
  space?: Space
  anchor?: "start" | "middle" | "end"
}

// A missing d keeps the path's shape (it changes only with the camera); "" clears it.
export interface PathState {
  d?: string
  opacity: number
  width?: number
}

export interface ParticleState {
  x: number
  y: number
  r: number
  opacity: number
}

export interface LabelState {
  x: number
  y: number
  opacity: number
}

export interface Frame {
  paths: Record<string, PathState>
  particles: ParticleState[]
  // A soft glow behind each moving cloud, at its middle.
  glows: ParticleState[]
  labels: Record<string, LabelState>
  // The code line to highlight, by its mark in the excerpt (see data.ts), and which entry of the
  // beat's meta list is current (-1 for none).
  mark: string | null
  metaIndex: number
}

export interface Beat {
  name: BeatName
  durationMs: number
  // The moment the static frame shows.
  settledMs: number
  paths: PathSpec[]
  particles: ParticleSpec[]
  glows: Space[]
  labels: LabelSpec[]
  // Where the camera is at a moment of the beat.
  shot(tMs: number): Shot
  // The cameras the beat arrives from and leaves toward in a transition.
  enter: Shot
  exit: Shot
  // Paths that change only with the camera.
  still(camera: Camera, view: View): Record<string, PathState>
  // Everything that moves with time.
  frame(tMs: number, camera: Camera, view: View): Frame
}

export const HIDDEN: ParticleState = { x: 0, y: 0, r: 0, opacity: 0 }
const PARTICLE_RADIUS_UNITS = [1.1, 1.8, 2.6]
const DOT_WIDTH_UNITS = 2.1
const GLOW_RADIUS_UNITS = 26

// Path data for a dot at each point: a zero-length segment that a round cap draws as a dot.
export function dots(points: Vec3[], camera: Camera, view: View): string {
  let d = ""
  for (const point of points) {
    const p = project(point, camera, view)
    d += `M${round(p.x)} ${round(p.y)}h0`
  }
  return d
}

export function polygon(points: Vec3[], camera: Camera, view: View): string {
  return (
    points
      .map((point, index) => {
        const p = project(point, camera, view)
        return `${index === 0 ? "M" : "L"}${round(p.x)} ${round(p.y)}`
      })
      .join("") + "Z"
  )
}

// A rectangle on a floor at height y, spanning x0..x1 across and z0..z1 in depth.
export function rect(x0: number, x1: number, z0: number, z1: number, y: number): Vec3[] {
  return [vec(x0, y, z0), vec(x1, y, z0), vec(x1, y, z1), vec(x0, y, z1)]
}

function grid(x0: number, z0: number, cols: number, rows: number, pitch: number, y: number): Vec3[] {
  const points: Vec3[] = []
  for (let row = 0; row < rows; row += 1)
    for (let col = 0; col < cols; col += 1) points.push(vec(x0 + (col + 0.5) * pitch, y, z0 + (row + 0.5) * pitch))
  return points
}

export function centroid(points: Vec3[]): Vec3 {
  const sum = points.reduce((acc, p) => vec(acc.x + p.x, acc.y + p.y, acc.z + p.z), vec(0, 0, 0))
  return vec(sum.x / points.length, sum.y / points.length, sum.z / points.length)
}

// How far a particle with this delay is along a flow at progress f: 0 before it leaves, 1 once landed.
export function travel(f: number, delay: number, span: number): number {
  return easeInOutSine(clamp((f - delay) / span, 0, 1))
}

// A particle on its curve: visible only in flight, fading in and out at the ends.
export function inFlight(
  from: Vec3,
  control: Vec3,
  to: Vec3,
  t: number,
  depth: 0 | 1 | 2,
  camera: Camera,
  view: View
): ParticleState {
  if (t <= 0 || t >= 1) return HIDDEN
  const p = project(bezier3(from, control, to, t), camera, view)
  return {
    x: round(p.x),
    y: round(p.y),
    r: round(PARTICLE_RADIUS_UNITS[depth] * Math.min(2.4, p.near) * (view.scale / 2.2)),
    opacity: round(Math.sin(Math.PI * t) ** 0.6 * [0.45, 0.7, 1][depth]),
  }
}

// The glow behind a cloud: at the mean of its visible particles, as bright as they are on average.
export function glowOf(cloud: ParticleState[], view: View): ParticleState {
  const shown = cloud.filter(particle => particle.opacity > 0)
  if (shown.length < 3) return HIDDEN
  const x = shown.reduce((sum, p) => sum + p.x, 0) / shown.length
  const y = shown.reduce((sum, p) => sum + p.y, 0) / shown.length
  const presence = shown.length / cloud.length
  return { x: round(x), y: round(y), r: round(GLOW_RADIUS_UNITS * (view.scale / 2.2)), opacity: round(presence * 0.9) }
}

export function depthOf(random: () => number): 0 | 1 | 2 {
  const value = random()
  return value < 0.2 ? 2 : value < 0.55 ? 1 : 0
}

export function phase(tMs: number, startMs: number, durationMs: number): number {
  return clamp((tMs - startMs) / durationMs, 0, 1)
}

// Dimensions in labels keep together when a narrow stage wraps them.
export const dims = (rows: number, cols: number) => `${rows}\u00a0x\u00a0${cols}`

// Beat 1: tiles.

const STRIDE = 2
const GAP = 6
const Y_SHARED = 56
const Y_REG = 106
export const FLOOR_PAD = 4
const ITERATIONS = 3
const ITER_MS = 2600
const INTRO_MS = 300
const OUTRO_MS = 900
const SLIDE_MS = 300
const A_LIFT = { startMs: 300, durationMs: 1000 }
const B_LIFT = { startMs: 800, durationMs: 1000 }
const DOT_FLOW = { startMs: 1700, durationMs: 900 }
const DUST_PER_LIFT = 20
// Global memory's dots rest dim, so the tiles being read stand out.
const GLOBAL_DIM = 0.4

interface Flight {
  from: Vec3
  control: Vec3
  to: Vec3
  delay: number
  depth: 0 | 1 | 2
}

export function tilesBeat(data: TilesData): Beat {
  const { M, N, K, tileM, tileN, tileK } = data
  const kts = data.kts.slice(0, ITERATIONS)
  // A sits left of the output, B behind it, C under the accumulator: the matmul laid flat.
  const aX0 = -GAP - K
  const bZ0 = -GAP - K
  const sharedAX0 = -GAP - tileK
  const sharedBZ0 = -GAP - tileK
  const globalA = grid(aX0, 0, K / STRIDE, M / STRIDE, STRIDE, 0)
  const globalB = grid(0, bZ0, N / STRIDE, K / STRIDE, STRIDE, 0)
  const globalC = grid(0, 0, N / STRIDE, M / STRIDE, STRIDE, 0)
  const sharedA = grid(sharedAX0, 0, tileK / STRIDE, tileM / STRIDE, STRIDE, Y_SHARED)
  const sharedB = grid(0, sharedBZ0, tileN / STRIDE, tileK / STRIDE, STRIDE, Y_SHARED)
  const acc = grid(0, 0, tileN / STRIDE, tileM / STRIDE, STRIDE, Y_REG)
  const floors = {
    global: rect(aX0 - FLOOR_PAD, N + FLOOR_PAD, bZ0 - FLOOR_PAD, M + FLOOR_PAD, 0),
    shared: rect(sharedAX0 - FLOOR_PAD, tileN + FLOOR_PAD, sharedBZ0 - FLOOR_PAD, tileM + FLOOR_PAD, Y_SHARED),
    reg: rect(-FLOOR_PAD, tileN + FLOOR_PAD, -FLOOR_PAD, tileM + FLOOR_PAD, Y_REG),
  }

  // A tile of A at k step kt: tileM rows, tileK columns; B's is tileK rows, tileN columns.
  const aTileAt = (kt: number) => grid(aX0 + kt, 0, tileK / STRIDE, tileM / STRIDE, STRIDE, 0)
  const bTileAt = (kt: number) => grid(0, bZ0 + kt, tileN / STRIDE, tileK / STRIDE, STRIDE, 0)
  const lift = (from: Vec3, to: Vec3, rise: number): Vec3 =>
    vec((from.x + to.x) / 2, Math.max(from.y, to.y) + rise, (from.z + to.z) / 2)

  const random = mulberry32(7)
  const liftFlights = (fromAt: (kt: number) => Vec3[], to: Vec3[]) => {
    const flights: { tile: number; delay: number; depth: 0 | 1 | 2; jitter: Vec3 }[] = []
    to.forEach((_, tile) => flights.push({ tile, delay: random() * 0.3, depth: depthOf(random), jitter: vec(0, 0, 0) }))
    for (let dust = 0; dust < DUST_PER_LIFT; dust += 1) {
      flights.push({
        tile: Math.floor(random() * to.length),
        delay: random() * 0.35,
        depth: 0,
        jitter: vec((random() - 0.5) * 14, (random() - 0.5) * 10, (random() - 0.5) * 14),
      })
    }
    return (kt: number): Flight[] => {
      const from = fromAt(kt)
      return flights.map(({ tile, delay, depth, jitter }) => {
        const start = from[tile]
        const end = to[tile]
        const control = lift(start, end, 16)
        return { from: start, to: end, delay, depth, control: vec(control.x + jitter.x, control.y + jitter.y, control.z + jitter.z) }
      })
    }
  }
  const aFlights = liftFlights(aTileAt, sharedA)
  const bFlights = liftFlights(bTileAt, sharedB)
  // Into the accumulator: every element of a reaches across its row of acc, every element of b down
  // its column, so each lands somewhere on its line.
  const dotFlights: Flight[] = []
  const colsOfAcc = tileN / STRIDE
  const rowsOfAcc = tileM / STRIDE
  sharedA.forEach((from, index) => {
    const row = Math.floor(index / (tileK / STRIDE))
    const to = acc[row * colsOfAcc + Math.floor(random() * colsOfAcc)]
    dotFlights.push({ from, to, control: lift(from, to, 6), delay: random() * 0.3, depth: depthOf(random) })
  })
  sharedB.forEach((from, index) => {
    const col = index % colsOfAcc
    const to = acc[Math.floor(random() * rowsOfAcc) * colsOfAcc + col]
    dotFlights.push({ from, to, control: lift(from, to, 6), delay: random() * 0.3, depth: depthOf(random) })
  })

  const all = [floors.global, floors.shared, floors.reg].flat()
  const restCamera: Camera = { target: centroid(all), yawDeg: 45, pitchDeg: 24, distance: 520, focal: 520 }
  const restView = fitView(all, restCamera, FIT_BOX)
  const accCenter = vec(tileN / 2, Y_REG, tileM / 2)
  const awayCamera: Camera = { ...restCamera, target: accCenter, pitchDeg: 40, distance: 90 }
  const awayView: View = { scale: restView.scale, cx: VIEW_WIDTH_UNITS / 2, cy: VIEW_HEIGHT_UNITS / 2 }

  const ktAt = (tMs: number) => {
    const iteration = clamp(Math.floor((tMs - INTRO_MS) / ITER_MS), 0, kts.length - 1)
    return { iteration, local: tMs - INTRO_MS - iteration * ITER_MS }
  }

  return {
    name: "tiles",
    durationMs: INTRO_MS + kts.length * ITER_MS + OUTRO_MS,
    settledMs: INTRO_MS + ITER_MS + DOT_FLOW.startMs + DOT_FLOW.durationMs * 0.45,
    paths: [
      { key: "floor-global", className: "tl-floor sp-global" },
      { key: "dots-c", className: "tl-dots tl-faint sp-global" },
      { key: "dots-a", className: "tl-dots sp-global" },
      { key: "dots-b", className: "tl-dots sp-global" },
      { key: "tile-a", className: "tl-outline sp-global" },
      { key: "tile-b", className: "tl-outline sp-global" },
      { key: "tile-a-dots", className: "tl-dots sp-global" },
      { key: "tile-b-dots", className: "tl-dots sp-global" },
      { key: "guides", className: "tl-guide" },
      { key: "floor-shared", className: "tl-floor sp-shared" },
      { key: "shared-a", className: "tl-dots sp-shared" },
      { key: "shared-b", className: "tl-dots sp-shared" },
      { key: "floor-reg", className: "tl-floor sp-reg" },
      { key: "acc", className: "tl-dots sp-reg" },
    ],
    particles: [
      ...aFlights(0).map(f => ({ space: "global" as const, depth: f.depth })),
      ...bFlights(0).map(f => ({ space: "global" as const, depth: f.depth })),
      ...dotFlights.map(f => ({ space: "shared" as const, depth: f.depth })),
    ],
    glows: ["global", "global", "shared"],
    labels: [
      { key: "reg", text: "registers", detail: `acc ${dims(tileM, tileN)}`, space: "reg", anchor: "end" },
      { key: "shared", text: "shared memory", detail: `a ${dims(tileM, tileK)}, b ${dims(tileK, tileN)}`, space: "shared", anchor: "end" },
      { key: "global", text: "global memory", detail: M === N && N === K ? `A, B, C ${dims(M, M)}` : `A ${dims(M, K)}, B ${dims(K, N)}`, space: "global", anchor: "end" },
    ],
    shot: () => ({ camera: restCamera, view: restView }),
    enter: { camera: awayCamera, view: awayView },
    exit: { camera: awayCamera, view: awayView },
    still(camera, view) {
      const width = (point: Vec3) => round(DOT_WIDTH_UNITS * (view.scale / 2.2) * project(point, camera, view).near)
      return {
        "floor-global": { d: polygon(floors.global, camera, view), opacity: 1 },
        "floor-shared": { d: polygon(floors.shared, camera, view), opacity: 1 },
        // The accumulator's column: its corners dropped to the tile of C it becomes.
        guides: {
          d: rect(0, tileN, 0, tileM, 0)
            .map(corner => {
              const top = project(vec(corner.x, Y_REG, corner.z), camera, view)
              const bottom = project(corner, camera, view)
              return `M${round(top.x)} ${round(top.y)}L${round(bottom.x)} ${round(bottom.y)}`
            })
            .join(""),
          opacity: 1,
        },
        "floor-reg": { d: polygon(floors.reg, camera, view), opacity: 1 },
        "dots-a": { d: dots(globalA, camera, view), opacity: GLOBAL_DIM, width: width(centroid(globalA)) },
        "dots-b": { d: dots(globalB, camera, view), opacity: GLOBAL_DIM, width: width(centroid(globalB)) },
        "dots-c": { d: dots(globalC, camera, view), opacity: GLOBAL_DIM, width: width(centroid(globalC)) },
        "shared-a": { d: dots(sharedA, camera, view), opacity: 1, width: width(centroid(sharedA)) },
        "shared-b": { d: dots(sharedB, camera, view), opacity: 1, width: width(centroid(sharedB)) },
        acc: { d: dots(acc, camera, view), opacity: 1, width: width(centroid(acc)) },
      }
    },
    frame(tMs, camera, view) {
      const { iteration, local } = ktAt(tMs)
      const kt = kts[iteration]
      const previousKt = kts[Math.max(0, iteration - 1)]
      const slide = easeInOutSine(phase(local, 0, SLIDE_MS))
      const shownKt = previousKt + (kt - previousKt) * (iteration === 0 ? 1 : slide)
      const aLift = phase(local, A_LIFT.startMs, A_LIFT.durationMs)
      const bLift = phase(local, B_LIFT.startMs, B_LIFT.durationMs)
      const dotFlow = phase(local, DOT_FLOW.startMs, DOT_FLOW.durationMs)
      const before = tMs < INTRO_MS

      const particles: ParticleState[] = []
      const glows: ParticleState[] = []
      const fly = (flights: Flight[], f: number, span: number) => {
        const cloud = flights.map(flight =>
          inFlight(flight.from, flight.control, flight.to, travel(f, flight.delay, span), flight.depth, camera, view)
        )
        particles.push(...cloud)
        glows.push(glowOf(cloud, view))
      }
      fly(aFlights(kt), aLift, 0.65)
      fly(bFlights(kt), bLift, 0.65)
      fly(dotFlights, dotFlow, 0.7)

      // Shared tiles fill as their clouds land and fade once the dot has used them.
      const consumed = phase(local, ITER_MS - 300, 300)
      const fill = (f: number) => (before ? 0.18 : 0.18 + 0.82 * clamp((f - 0.35) / 0.65, 0, 1) * (1 - 0.6 * consumed))
      const accLevel = before ? 0 : (iteration + clamp((dotFlow - 0.3) / 0.7, 0, 1)) / kts.length
      const tileGlow = (f: number) => (before ? 0.35 : 0.35 + 0.65 * Math.sin(Math.PI * clamp(f * 1.4, 0, 1)))

      const mark = before ? null : local < B_LIFT.startMs ? "a" : local < DOT_FLOW.startMs ? "b" : "dot"
      const tileDotWidth = round(DOT_WIDTH_UNITS * 1.15 * (view.scale / 2.2) * project(vec(0, 0, 0), camera, view).near)
      const labels: Record<string, LabelState> = {
        global: leftOf(floors.global, camera, view),
        shared: leftOf(floors.shared, camera, view),
        reg: leftOf(floors.reg, camera, view),
      }
      return {
        paths: {
          "tile-a": { d: polygon(rect(aX0 + shownKt, aX0 + shownKt + tileK, 0, tileM, 0), camera, view), opacity: tileGlow(aLift) },
          "tile-b": { d: polygon(rect(0, tileN, bZ0 + shownKt, bZ0 + shownKt + tileK, 0), camera, view), opacity: tileGlow(bLift) },
          "tile-a-dots": { d: dots(aTileAt(shownKt), camera, view), opacity: 1, width: tileDotWidth },
          "tile-b-dots": { d: dots(bTileAt(shownKt), camera, view), opacity: 1, width: tileDotWidth },
          "shared-a": { opacity: round(fill(aLift)) },
          "shared-b": { opacity: round(fill(bLift)) },
          acc: { opacity: round(0.16 + 0.84 * accLevel) },
        },
        particles,
        glows,
        labels,
        mark,
        metaIndex: before ? 0 : iteration,
      }
    },
  }
}

// A floor's label: left of its leftmost corner.
export function leftOf(points: Vec3[], camera: Camera, view: View): LabelState {
  const projected = points.map(point => project(point, camera, view))
  const left = projected.reduce((best, p) => (p.x < best.x ? p : best))
  return { x: round(left.x - 12), y: round(left.y), opacity: 1 }
}

// Beat 2: lanes.

const Y_STRIP = -30
const STRIP_Z = { w: 8, x: 18 }
const STRIP_ROW_GAP = 4
const LOAD_MS = 1100
const STAGE_MS = 1050
const LANES_INTRO_MS = 300
const STAGES_START_MS = LANES_INTRO_MS + 2 * LOAD_MS + 200
const LANES_OUTRO_MS = 1000
const PER_LANE_LOADS = 2
const PER_LANE_SHUFFLES = 2
const ARC_SAMPLES = 14

export function lanesBeat(data: LanesData): Beat {
  const { lanes, runElems, masks } = data
  const kSteps = data.kSteps
  const pitch = runElems
  const laneAt = (lane: number) => vec(lane * pitch + pitch / 2, 0, 0)
  const lanePoints = Array.from({ length: lanes }, (_, lane) => laneAt(lane))
  const width = lanes * pitch
  // Each k step's run of W and of x is a row of `width` elements under the lanes.
  const stripRow = (z: number) => Array.from({ length: width }, (_, e) => vec(e + 0.5, Y_STRIP, z))
  const wRows = kSteps.map((_, step) => stripRow(STRIP_Z.w + step * STRIP_ROW_GAP))
  const xRows = kSteps.map((_, step) => stripRow(STRIP_Z.x + step * STRIP_ROW_GAP))
  const floors = {
    reg: rect(-FLOOR_PAD, width + FLOOR_PAD, -8, 8, 0),
    global: rect(-FLOOR_PAD, width + FLOOR_PAD, STRIP_Z.w - 4, STRIP_Z.x + (kSteps.length - 1) * STRIP_ROW_GAP + 4, Y_STRIP),
  }

  const random = mulberry32(11)
  // Loads: each lane takes PER_LANE_LOADS elements of its run from W and as many from x.
  const loadSpecs = Array.from({ length: lanes * PER_LANE_LOADS * 2 }, (_, index) => {
    const lane = Math.floor(index / (PER_LANE_LOADS * 2))
    const fromW = index % 2 === 0
    return { lane, fromW, offset: Math.floor(random() * runElems), delay: random() * 0.3, depth: depthOf(random) }
  })
  const shuffleSpecs = Array.from({ length: lanes * PER_LANE_SHUFFLES }, (_, index) => ({
    lane: Math.floor(index / PER_LANE_SHUFFLES),
    delay: (index % PER_LANE_SHUFFLES) * 0.12 + random() * 0.12,
    depth: depthOf(random),
  }))
  const arcControl = (lane: number, partner: number): Vec3 => {
    const a = laneAt(lane)
    const b = laneAt(partner)
    const span = Math.abs(b.x - a.x)
    // The two directions of a pair take opposite sides, so an exchange reads as a loop in depth.
    const side = lane < partner ? -1 : 1
    return vec((a.x + b.x) / 2, 5 + span * 0.42, side * (2 + span * 0.16))
  }

  const all = [...floors.reg, ...floors.global, vec(width / 2, 5 + (width / 2) * 0.42, 0)]
  const restCamera: Camera = { target: centroid(all), yawDeg: -16, pitchDeg: 22, distance: 420, focal: 420 }
  const restView = fitView(all, restCamera, FIT_BOX)
  const awayCamera: Camera = { ...restCamera, pitchDeg: 40, distance: restCamera.distance * 3.2 }
  const awayView: View = { scale: restView.scale, cx: VIEW_WIDTH_UNITS / 2, cy: VIEW_HEIGHT_UNITS / 2 }

  const stageAt = (tMs: number) => Math.floor((tMs - STAGES_START_MS) / STAGE_MS)

  return {
    name: "lanes",
    durationMs: STAGES_START_MS + masks.length * STAGE_MS + LANES_OUTRO_MS,
    settledMs: STAGES_START_MS + STAGE_MS * 0.5,
    paths: [
      { key: "floor-global", className: "tl-floor sp-global" },
      { key: "strip-w", className: "tl-dots sp-global" },
      { key: "strip-x", className: "tl-dots tl-faint sp-global" },
      { key: "floor-reg", className: "tl-floor sp-reg" },
      { key: "arcs", className: "tl-arc sp-reg" },
      { key: "lanes", className: "tl-dots tl-lane sp-reg" },
    ],
    particles: [
      ...loadSpecs.map(spec => ({ space: "global" as const, depth: spec.depth })),
      ...shuffleSpecs.map(spec => ({ space: "reg" as const, depth: spec.depth })),
    ],
    glows: [],
    labels: [
      { key: "reg", text: "registers", detail: `one warp, ${lanes} lanes`, space: "reg", anchor: "end" },
      { key: "global", text: "global memory", detail: "W[row,\u00a0:], x", space: "global", anchor: "end" },
      { key: "lane0", text: "lane 0", space: "reg", anchor: "middle" },
      { key: "lastLane", text: `lane ${lanes - 1}`, space: "reg", anchor: "middle" },
    ],
    shot: () => ({ camera: restCamera, view: restView }),
    enter: { camera: awayCamera, view: awayView },
    exit: { camera: awayCamera, view: awayView },
    still(camera, view) {
      const dotWidth = (point: Vec3) => round(1.5 * (view.scale / 2.2) * project(point, camera, view).near)
      return {
        "floor-global": { d: polygon(floors.global, camera, view), opacity: 1 },
        "floor-reg": { d: polygon(floors.reg, camera, view), opacity: 1 },
        "strip-w": { d: dots(wRows.flat(), camera, view), opacity: 1, width: dotWidth(centroid(wRows.flat())) },
        "strip-x": { d: dots(xRows.flat(), camera, view), opacity: 1, width: dotWidth(centroid(xRows.flat())) },
      }
    },
    frame(tMs, camera, view) {
      const particles: ParticleState[] = []
      // Which k step is loading, and how far.
      const loadIndex = clamp(Math.floor((tMs - LANES_INTRO_MS) / LOAD_MS), 0, kSteps.length - 1)
      const load = phase(tMs - loadIndex * LOAD_MS, LANES_INTRO_MS, LOAD_MS)
      for (const spec of loadSpecs) {
        const row = (spec.fromW ? wRows : xRows)[loadIndex]
        const from = row[spec.lane * runElems + spec.offset]
        const to = laneAt(spec.lane)
        const control = vec((from.x + to.x) / 2, 4, (from.z + to.z) / 2)
        particles.push(inFlight(from, control, to, travel(load, spec.delay, 0.6), spec.depth, camera, view))
      }

      const stage = stageAt(tMs)
      const inStages = stage >= 0 && stage < masks.length
      const stageT = inStages ? phase(tMs, STAGES_START_MS + stage * STAGE_MS, STAGE_MS) : 0
      const mask = inStages ? masks[stage] : 0
      for (const spec of shuffleSpecs) {
        if (!inStages) {
          particles.push(HIDDEN)
          continue
        }
        const partner = spec.lane ^ mask
        particles.push(
          inFlight(
            laneAt(spec.lane),
            arcControl(spec.lane, partner),
            laneAt(partner),
            travel(stageT, spec.delay, 0.7),
            spec.depth,
            camera,
            view
          )
        )
      }

      let arcs = ""
      if (inStages) {
        for (let lane = 0; lane < lanes; lane += 1) {
          const partner = lane ^ mask
          const control = arcControl(lane, partner)
          for (let sample = 0; sample <= ARC_SAMPLES; sample += 1) {
            const p = project(bezier3(laneAt(lane), control, laneAt(partner), sample / ARC_SAMPLES), camera, view)
            arcs += `${sample === 0 ? "M" : "L"}${round(p.x)} ${round(p.y)}`
          }
        }
      }
      const arcOpacity = inStages ? Math.sin(Math.PI * stageT) ** 0.5 : 0

      // A lane's dot grows as its value gathers more partial sums: one run after the loads, all 32 at the end.
      const loaded = tMs < LANES_INTRO_MS ? 0 : clamp((tMs - LANES_INTRO_MS) / (2 * LOAD_MS), 0, 1)
      const stagesDone = clamp((tMs - STAGES_START_MS) / STAGE_MS, 0, masks.length)
      const laneWidth = round((1.4 + 1.4 * loaded + 0.55 * stagesDone) * (view.scale / 2.2) * project(lanePoints[lanes / 2], camera, view).near)
      const mark = tMs < LANES_INTRO_MS ? null : tMs < STAGES_START_MS ? (load < 0.35 ? "at" : "acc") : "sum"
      const lane0 = project(lanePoints[0], camera, view)
      const lastLane = project(lanePoints[lanes - 1], camera, view)
      return {
        paths: {
          arcs: { d: arcs, opacity: round(arcOpacity) },
          lanes: { d: dots(lanePoints, camera, view), opacity: round(0.45 + 0.55 * Math.min(1, loaded + stagesDone / masks.length)), width: laneWidth },
        },
        particles,
        glows: [],
        labels: {
          global: leftOf(floors.global, camera, view),
          reg: leftOf(floors.reg, camera, view),
          lane0: { x: round(lane0.x), y: round(lane0.y + 20), opacity: 1 },
          lastLane: { x: round(lastLane.x), y: round(lastLane.y + 20), opacity: 1 },
        },
        mark,
        metaIndex: inStages ? stage : stagesDone >= masks.length ? masks.length : -1,
      }
    },
  }
}

export function round(value: number): number {
  return Math.round(value * 10) / 10
}
