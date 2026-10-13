import { describe, expect, it } from "vitest"
import {
  bodyMatrix,
  CENTER_UNITS,
  makeBody,
  orbitPoint,
  ORBIT,
  project,
  SEMI_AXES,
  STRIDE,
  VIEW_UNITS,
} from "../src/components/home/hero/geometry"
import { composeFrame, inkOf, LEVELS, makeFrame, makeScene } from "../src/components/home/hero/frame"
import {
  CYCLE_S,
  EVENT_START_S,
  gridSlot,
  MAX_DELAY_S,
  PHASES_S,
  STREAM_POINTS,
  streamPosition,
  streamState,
} from "../src/components/home/hero/stream"
import { staticFrame } from "../src/components/home/hero/static-frame"

const EVENT_S = Object.values(PHASES_S).reduce((sum, durationS) => sum + durationS, 0)

describe("hero body", () => {
  it("is the same for the same seed", () => {
    expect(makeBody(500, 7)).toEqual(makeBody(500, 7))
    expect(makeBody(500, 7)).not.toEqual(makeBody(500, 8))
  })

  it("fills the requested count with unit normals near the ellipsoid", () => {
    const body = makeBody(2000)
    expect(body.length).toBe(2000 * STRIDE)
    for (let index = 0; index < 2000; index += 1) {
      const o = index * STRIDE
      const [x, y, z] = [body[o], body[o + 1], body[o + 2]]
      const shell = (x / SEMI_AXES[0]) ** 2 + (y / SEMI_AXES[1]) ** 2 + (z / SEMI_AXES[2]) ** 2
      expect(shell).toBeGreaterThan(0.55)
      expect(shell).toBeLessThan(1.45)
      expect(Math.hypot(body[o + 3], body[o + 4], body[o + 5])).toBeCloseTo(1, 4)
      // Normals point outward.
      expect(x * body[o + 3] + y * body[o + 4] + z * body[o + 5]).toBeGreaterThan(0)
    }
  })

  it("turns rigidly", () => {
    const m = bodyMatrix(1.3)
    const column = (c: number) => Math.hypot(m[c], m[3 + c], m[6 + c])
    for (let c = 0; c < 3; c += 1) expect(column(c)).toBeCloseTo(1, 10)
  })
})

describe("hero projection", () => {
  it("puts the origin at the center and brings nearer points closer to the viewer", () => {
    const origin = project([0, 0, 0])
    expect(origin.x).toBe(CENTER_UNITS.x)
    expect(origin.y).toBe(CENTER_UNITS.y)
    expect(origin.scale).toBe(1)
    const far = project([1, 1, -1])
    const near = project([1, 1, 1])
    expect(near.scale).toBeGreaterThan(1)
    expect(far.scale).toBeLessThan(1)
    expect(near.x - CENTER_UNITS.x).toBeGreaterThan(far.x - CENTER_UNITS.x)
    // y is down on screen.
    expect(near.y).toBeLessThan(CENTER_UNITS.y)
  })

  it("tips the orbit's near half below the body", () => {
    const front = orbitPoint(Math.PI / 2)
    const back = orbitPoint(-Math.PI / 2)
    expect(front[2]).toBeGreaterThan(0)
    expect(back[2]).toBeLessThan(0)
    expect(project(front).y).toBeGreaterThan(project(back).y)
    expect(Math.hypot(...orbitPoint(0.7))).toBeCloseTo(ORBIT.radius, 10)
  })
})

describe("hero stream", () => {
  it("fits in a cycle, with every point resting at both ends", () => {
    expect(EVENT_START_S + MAX_DELAY_S + EVENT_S).toBeLessThan(CYCLE_S)
    for (let slot = 0; slot < STREAM_POINTS; slot += 1) {
      expect(streamState(slot, 0).phase).toBe("rest")
      expect(streamState(slot, CYCLE_S - 1e-6).phase).toBe("rest")
    }
  })

  it("leaves from the surface and lands back on it, so the loop has no seam", () => {
    const surface: [number, number, number] = [0.3, 0.6, 0.5]
    const outward: [number, number, number] = [0.3, 0.6, 0.5]
    for (const slot of [0, 77, STREAM_POINTS - 1]) {
      const start = streamPosition(slot, { phase: "lift", t: 0, holdS: 0 }, surface, outward)
      const end = streamPosition(slot, { phase: "sink", t: 1, holdS: 0 }, surface, outward)
      for (let axis = 0; axis < 3; axis += 1) {
        expect(start?.[axis]).toBeCloseTo(surface[axis], 9)
        expect(end?.[axis]).toBeCloseTo(surface[axis], 9)
      }
    }
  })

  it("moves continuously through the cycle", () => {
    const surface: [number, number, number] = [0.2, 0.5, 0.7]
    for (const slot of [3, 130, 255]) {
      let previous = surface
      for (let cycleS = 0; cycleS < CYCLE_S; cycleS += 0.01) {
        const at = streamPosition(slot, streamState(slot, cycleS), surface, surface) ?? surface
        expect(Math.hypot(at[0] - previous[0], at[1] - previous[1], at[2] - previous[2])).toBeLessThan(0.08)
        previous = at
      }
    }
  })

  it("gives every slot its own grid cell", () => {
    const cells = new Set<string>()
    for (let slot = 0; slot < STREAM_POINTS; slot += 1) {
      cells.add(gridSlot(slot).map(value => value.toFixed(4)).join(","))
    }
    expect(cells.size).toBe(STREAM_POINTS)
  })
})

describe("hero frame", () => {
  it("keeps every point on the view and every level in range, over a whole cycle", () => {
    const scene = makeScene(1500)
    const frame = makeFrame(scene)
    for (let timeS = 0; timeS < CYCLE_S; timeS += 0.5) {
      composeFrame(scene, { timeS, pitch: 0.16, yaw: -0.26 }, "emit", frame)
      expect(frame.farEnd).toBeLessThanOrEqual(frame.bodyEnd)
      expect(frame.bodyEnd).toBeLessThanOrEqual(frame.count)
      for (let index = 0; index < frame.count; index += 1) {
        expect(frame.x[index]).toBeGreaterThan(0)
        expect(frame.x[index]).toBeLessThan(VIEW_UNITS)
        expect(frame.y[index]).toBeGreaterThan(0)
        expect(frame.y[index]).toBeLessThan(VIEW_UNITS)
        expect(frame.level[index]).toBeLessThan(LEVELS)
      }
    }
  })

  it("lights the body from the left by night", () => {
    const scene = makeScene(3000)
    const frame = makeFrame(scene)
    composeFrame(scene, { timeS: 0, pitch: 0, yaw: 0 }, "emit", frame)
    let left = 0
    let right = 0
    for (let index = frame.farEnd; index < frame.bodyEnd; index += 1) {
      if (frame.x[index] < CENTER_UNITS.x - 60) left += inkOf(frame.level[index]).alpha
      if (frame.x[index] > CENTER_UNITS.x + 60) right += inkOf(frame.level[index]).alpha
    }
    expect(left).toBeGreaterThan(right * 2)
  })

  it("draws the build-time frame from the same scene, deterministically", () => {
    const frame = staticFrame(600)
    expect(frame.paths.length).toBeGreaterThan(5)
    expect(JSON.stringify(staticFrame(600))).toBe(JSON.stringify(frame))
  })
})
