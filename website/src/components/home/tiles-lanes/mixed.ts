// Beat 3 of the tiles-and-lanes scene: one kernel at both levels. Three floors as in beat 1, each
// as long as the row of x: global memory holds x and a row of W per warp, shared memory the tile xs,
// registers the CTA's warps. First the whole CTA copies x into xs, a cloud of one particle per
// thread that takes part. Then the camera swings down to warp 0, whose lanes read their runs of xs,
// and of their row of W straight from global memory, at the starts the trace gives, before the
// butterfly of beat 2 folds their sums.

import { clamp, easeInOutSine, mulberry32 } from "../../../lib/figure/math"
import { bezier3, fitView, lerpCamera, project, vec, type Camera, type Vec3 } from "./camera"
import type { MixedData } from "./data"
import {
  centroid,
  depthOf,
  dims,
  dots,
  FIT_BOX,
  FLOOR_PAD,
  glowOf,
  HIDDEN,
  inFlight,
  leftOf,
  phase,
  polygon,
  rect,
  round,
  travel,
  VIEW_HEIGHT_UNITS,
  VIEW_WIDTH_UNITS,
  type Beat,
  type LabelState,
  type ParticleState,
  type Shot,
} from "./scene"

// One dot of a strip stands for this many elements.
const STRIDE = 2
const Y_SHARED = 64
const Y_REG = 128
const Z_X = 6
const ROW_GAP = 4
const WARP_GAP = 5
const INTRO_MS = 300
const COPY = { startMs: 300, durationMs: 1300 }
const BARRIER = { startMs: 1500, durationMs: 400 }
const DIVE = { startMs: 1800, durationMs: 1500 }
const LOADS_START_MS = 3000
const LOAD_MS = 1050
const STAGE_MS = 850
const OUTRO_MS = 900
const SHUFFLES_PER_LANE = 2
const ARC_SAMPLES = 14
const ARC_RISE = 0.3
// The labels of the floors' long edges need a wider margin than the other beats leave them.
const BOX = { ...FIT_BOX, x: FIT_BOX.x + 16, width: FIT_BOX.width - 16 }
// The other warps stay as a faint row once the camera is down at warp 0.
const OTHER_WARPS_DIM = 0.2

export function mixedBeat(data: MixedData): Beat {
  const { K, lanes, warps, runElems, kSteps, tileRuns, weightRuns, masks, copyWidth, copyThreads } = data
  const stagesStartMs = LOADS_START_MS + kSteps.length * LOAD_MS + 200
  const strip = (y: number, z: number) =>
    Array.from({ length: K / STRIDE }, (_, d) => vec(d * STRIDE + STRIDE / 2, y, z))
  // Warp w's row of W sits behind x, warp 0's nearest.
  const zW = (warp: number) => Z_X - 2 * ROW_GAP - warp * ROW_GAP
  const zWarp = (warp: number) => Z_X - warp * WARP_GAP
  const globalX = strip(0, Z_X)
  const globalW = Array.from({ length: warps }, (_, warp) => strip(0, zW(warp)))
  const sharedX = strip(Y_SHARED, Z_X)
  // The lanes spread over the whole row, so each k step's runs fan out from one half of xs to all
  // of them.
  const lanePitch = K / lanes
  const laneAt = (lane: number, warp = 0) => vec((lane + 0.5) * lanePitch, Y_REG, zWarp(warp))
  const warpRow = (warp: number) => Array.from({ length: lanes }, (_, lane) => laneAt(lane, warp))
  const lanePoints = warpRow(0)
  const otherLanes = Array.from({ length: warps - 1 }, (_, warp) => warpRow(warp + 1)).flat()
  const floors = {
    global: rect(-FLOOR_PAD, K + FLOOR_PAD, zW(warps - 1) - FLOOR_PAD, Z_X + FLOOR_PAD, 0),
    shared: rect(-FLOOR_PAD, K + FLOOR_PAD, Z_X - FLOOR_PAD, Z_X + FLOOR_PAD, Y_SHARED),
    reg: rect(-FLOOR_PAD, K + FLOOR_PAD, zWarp(warps - 1) - FLOOR_PAD, Z_X + FLOOR_PAD, Y_REG),
  }

  const random = mulberry32(5)
  // The copy: thread t moves elements t * copyWidth onward, as the trace's thread map says.
  const copyFlights = Array.from({ length: copyThreads }, (_, thread) => {
    const from = vec(thread * copyWidth + copyWidth / 2, 0, Z_X)
    const to = vec(from.x, Y_SHARED, Z_X)
    const control = vec(
      from.x + (random() - 0.5) * 24,
      Y_SHARED * 0.5 + 10 + random() * 10,
      Z_X + (random() - 0.5) * 18
    )
    return { from, to, control, delay: random() * 0.3, depth: depthOf(random) }
  })
  // Loads: a particle from the lane's run of xs and one from its run of W.
  const loadSpecs = Array.from({ length: lanes * 2 }, (_, index) => ({
    lane: Math.floor(index / 2),
    fromTile: index % 2 === 0,
    delay: random() * 0.3,
    depth: depthOf(random),
  }))
  const shuffleSpecs = Array.from({ length: lanes * SHUFFLES_PER_LANE }, (_, index) => ({
    lane: Math.floor(index / SHUFFLES_PER_LANE),
    delay: (index % SHUFFLES_PER_LANE) * 0.12 + random() * 0.12,
    depth: depthOf(random),
  }))
  const arcControl = (lane: number, partner: number): Vec3 => {
    const a = laneAt(lane)
    const b = laneAt(partner)
    const span = Math.abs(b.x - a.x)
    const side = lane < partner ? -1 : 1
    return vec((a.x + b.x) / 2, Y_REG + 4 + span * ARC_RISE, Z_X + side * (2 + span * 0.1))
  }
  const arcTop = vec(K / 2, Y_REG + 4 + (lanes / 2) * lanePitch * ARC_RISE, Z_X)

  const all = [...floors.global, ...floors.shared, ...floors.reg, arcTop]
  const overviewCamera: Camera = { target: centroid(all), yawDeg: 30, pitchDeg: 32, distance: 900, focal: 900 }
  const overview: Shot = { camera: overviewCamera, view: fitView(all, overviewCamera, BOX) }
  const closeCamera: Camera = {
    target: centroid([...lanePoints, ...sharedX]),
    yawDeg: 0,
    pitchDeg: 18,
    distance: 330,
    focal: 330,
  }
  const close: Shot = { camera: closeCamera, view: fitView(all, closeCamera, BOX) }
  const center = { cx: VIEW_WIDTH_UNITS / 2, cy: VIEW_HEIGHT_UNITS / 2 }
  const enter: Shot = {
    camera: { ...overviewCamera, pitchDeg: 44, distance: overviewCamera.distance * 3 },
    view: { scale: overview.view.scale, ...center },
  }
  const exit: Shot = {
    camera: { ...closeCamera, target: laneAt(lanes / 2), pitchDeg: 30, distance: closeCamera.distance / 3.5 },
    view: { scale: close.view.scale, ...center },
  }
  const dive = (tMs: number) => easeInOutSine(phase(tMs, DIVE.startMs, DIVE.durationMs))
  const stageAt = (tMs: number) => Math.floor((tMs - stagesStartMs) / STAGE_MS)

  return {
    name: "mixed",
    durationMs: stagesStartMs + masks.length * STAGE_MS + OUTRO_MS,
    settledMs: LOADS_START_MS + LOAD_MS * 1.45,
    paths: [
      { key: "floor-global", className: "tl-floor sp-global" },
      { key: "w-rows", className: "tl-dots tl-faint sp-global" },
      { key: "x", className: "tl-dots sp-global" },
      { key: "floor-shared", className: "tl-floor sp-shared" },
      { key: "barrier", className: "tl-outline sp-shared" },
      { key: "xs", className: "tl-dots sp-shared" },
      { key: "floor-reg", className: "tl-floor sp-reg" },
      { key: "warps", className: "tl-dots tl-lane sp-reg" },
      { key: "arcs", className: "tl-arc sp-reg" },
      { key: "lanes", className: "tl-dots tl-lane sp-reg" },
    ],
    particles: [
      ...copyFlights.map(flight => ({ space: "global" as const, depth: flight.depth })),
      ...loadSpecs.map(spec => ({
        space: spec.fromTile ? ("shared" as const) : ("global" as const),
        depth: spec.depth,
      })),
      ...shuffleSpecs.map(spec => ({ space: "reg" as const, depth: spec.depth })),
    ],
    glows: ["global"],
    labels: [
      { key: "reg", text: "registers", detail: `${warps} warps, ${lanes} lanes`, space: "reg", anchor: "end" },
      { key: "shared", text: "shared memory", detail: `xs ${dims(1, K)}`, space: "shared", anchor: "end" },
      { key: "global", text: "global memory", detail: "x, W[row,\u00a0:]", space: "global", anchor: "end" },
    ],
    shot(tMs) {
      const t = dive(tMs)
      if (t <= 0) return overview
      if (t >= 1) return close
      // Fitted again at every step, since a view between two fits fits neither camera.
      const camera = lerpCamera(overview.camera, close.camera, t)
      return { camera, view: fitView(all, camera, BOX) }
    },
    enter,
    exit,
    still(camera, view) {
      const dotWidth = (point: Vec3) => round(1.5 * (view.scale / 2.2) * project(point, camera, view).near)
      const wDots = globalW.flat()
      return {
        "floor-global": { d: polygon(floors.global, camera, view), opacity: 1 },
        "floor-shared": { d: polygon(floors.shared, camera, view), opacity: 1 },
        "floor-reg": { d: polygon(floors.reg, camera, view), opacity: 1 },
        "w-rows": { d: dots(wDots, camera, view), opacity: 1, width: dotWidth(centroid(wDots)) },
        x: { d: dots(globalX, camera, view), opacity: 0.55, width: dotWidth(centroid(globalX)) },
        xs: { d: dots(sharedX, camera, view), width: dotWidth(centroid(sharedX)), opacity: 1 },
      }
    },
    frame(tMs, camera, view) {
      const particles: ParticleState[] = []
      const before = tMs < INTRO_MS
      const copy = phase(tMs, COPY.startMs, COPY.durationMs)
      const copyCloud = copyFlights.map(flight =>
        inFlight(flight.from, flight.control, flight.to, travel(copy, flight.delay, 0.65), flight.depth, camera, view)
      )
      particles.push(...copyCloud)

      const loadIndex = clamp(Math.floor((tMs - LOADS_START_MS) / LOAD_MS), 0, kSteps.length - 1)
      const load = phase(tMs - loadIndex * LOAD_MS, LOADS_START_MS, LOAD_MS)
      for (const spec of loadSpecs) {
        const start = (spec.fromTile ? tileRuns : weightRuns)[loadIndex][spec.lane]
        const from = vec(start + runElems / 2, spec.fromTile ? Y_SHARED : 0, spec.fromTile ? Z_X : zW(0))
        const to = laneAt(spec.lane)
        const control = vec((from.x + to.x) / 2, Math.max(from.y, Y_SHARED) + 6, (from.z + to.z) / 2 + 3)
        particles.push(inFlight(from, control, to, travel(load, spec.delay, 0.6), spec.depth, camera, view))
      }

      const stage = stageAt(tMs)
      const inStages = stage >= 0 && stage < masks.length
      const stageT = inStages ? phase(tMs, stagesStartMs + stage * STAGE_MS, STAGE_MS) : 0
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

      // xs fills as the cloud lands; the barrier after the copy shows as the shared floor lighting up.
      const filled = before ? 0 : clamp((copy - 0.3) / 0.7, 0, 1)
      const barrier = data.copyBarrier ? Math.sin(Math.PI * phase(tMs, BARRIER.startMs, BARRIER.durationMs)) : 0
      const down = dive(tMs)
      const loaded = clamp((tMs - LOADS_START_MS) / (kSteps.length * LOAD_MS), 0, 1)
      const stagesDone = clamp((tMs - stagesStartMs) / STAGE_MS, 0, masks.length)
      const laneWidth = round(
        (1.4 + 1.4 * loaded + 0.55 * stagesDone) *
          (view.scale / 2.2) *
          project(lanePoints[lanes / 2], camera, view).near
      )
      const dotWidth = (point: Vec3) => round(1.5 * (view.scale / 2.2) * project(point, camera, view).near)

      let mark: string | null = null
      if (!before) mark = tMs < LOADS_START_MS ? "stage" : tMs < stagesStartMs ? (load < 0.35 ? "at" : "acc") : "sum"
      const metaIndex = before
        ? -1
        : tMs < LOADS_START_MS
          ? 0
          : tMs < stagesStartMs
            ? 1 + loadIndex
            : 1 + kSteps.length + (stagesDone >= masks.length ? 1 : 0)

      const labels: Record<string, LabelState> = {
        global: leftOf(floors.global, camera, view),
        shared: leftOf(floors.shared, camera, view),
        reg: leftOf(floors.reg, camera, view),
      }
      return {
        paths: {
          xs: { opacity: round(0.15 + 0.85 * filled) },
          barrier: { d: polygon(floors.shared, camera, view), opacity: round(0.7 * barrier) },
          warps: {
            d: dots(otherLanes, camera, view),
            opacity: round(0.55 - (0.55 - OTHER_WARPS_DIM) * down),
            width: dotWidth(centroid(otherLanes)),
          },
          arcs: { d: arcs, opacity: round(inStages ? Math.sin(Math.PI * stageT) ** 0.5 : 0) },
          lanes: {
            d: dots(lanePoints, camera, view),
            opacity: round(0.55 + 0.45 * Math.min(1, loaded + stagesDone / masks.length)),
            width: laneWidth,
          },
        },
        particles,
        glows: [glowOf(copyCloud, view)],
        labels,
        mark,
        metaIndex,
      }
    },
  }
}
