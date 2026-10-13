import { describe, expect, it } from "vitest"
import {
  arcProgress,
  crossingS,
  ENTER_S,
  EXIT_S,
  layerX,
  LAYERS,
  loopPoint,
  ORBIT_RADIUS,
  rippleGlow,
  STACK_HALF_LENGTH,
  STEP_S,
  steps,
  stepsSinceCrossing,
  tokenS,
  TOKENS,
  wordOf,
  WORDS,
} from "../src/components/inference/hero/path"
import { composeFrame, countsFor, inkOf, LEVELS, makeFrame, makeScene, outlinePoint, VIEW_UNITS } from "../src/components/inference/hero/model"
import { STATIC_TIME_S, staticFrame } from "../src/components/inference/hero/static-frame"

describe("inference hero path", () => {
  it("is closed and continuous", () => {
    const start = loopPoint(0)
    const end = loopPoint(0.99999)
    expect(Math.hypot(start[0] - end[0], start[2] - end[2])).toBeLessThan(1e-3)
    for (let index = 0; index < 1000; index += 1) {
      const a = loopPoint(index / 1000)
      const b = loopPoint((index + 1) / 1000)
      expect(Math.hypot(a[0] - b[0], a[2] - b[2])).toBeLessThan(0.03)
    }
  })

  it("enters the stack at the left end and leaves at the right", () => {
    expect(loopPoint(ENTER_S)[0]).toBeCloseTo(-STACK_HALF_LENGTH, 2)
    expect(loopPoint(EXIT_S)[0]).toBeCloseTo(STACK_HALF_LENGTH, 2)
    expect(loopPoint(0)[0]).toBeCloseTo(-ORBIT_RADIUS, 3)
    const crossings = Array.from({ length: LAYERS }, (_, layer) => crossingS(layerX(layer)))
    for (let layer = 1; layer < LAYERS; layer += 1) expect(crossings[layer]).toBeGreaterThan(crossings[layer - 1])
  })

  it("steps forward by one every step, never backward", () => {
    expect(steps(STEP_S * 4) - steps(0)).toBeCloseTo(4, 9)
    let previous = steps(0)
    for (let timeS = 0.01; timeS < STEP_S * 3; timeS += 0.01) {
      const now = steps(timeS)
      expect(now).toBeGreaterThan(previous)
      previous = now
    }
  })

  it("repeats token positions every step, shifted by one slot", () => {
    for (let slot = 0; slot < TOKENS; slot += 1) {
      const next = tokenS((slot + 1) % TOKENS, 1.3)
      expect(tokenS(slot, 1.3 + STEP_S)).toBeCloseTo(next, 9)
    }
  })

  it("lets one token through the stack per step", () => {
    const layer = crossingS(0)
    expect(stepsSinceCrossing(layer, 2.7)).toBeGreaterThanOrEqual(0)
    expect(stepsSinceCrossing(layer, 2.7)).toBeLessThan(1)
    expect(stepsSinceCrossing(layer, 2.7 + STEP_S)).toBeCloseTo(stepsSinceCrossing(layer, 2.7), 9)
  })

  it("emits the words in order", () => {
    const emitted: string[] = []
    let last = new Map<number, string>()
    for (let timeS = 0; timeS < STEP_S * (WORDS.length + 2); timeS += STEP_S / 20) {
      const now = new Map<number, string>()
      for (let slot = 0; slot < TOKENS; slot += 1) {
        const word = wordOf(slot, timeS)
        now.set(slot, word)
        if (last.has(slot) && last.get(slot) !== word) emitted.push(word)
      }
      last = now
    }
    expect(emitted.length).toBeGreaterThan(WORDS.length)
    const first = WORDS.indexOf(emitted[0] as (typeof WORDS)[number])
    emitted.forEach((word, index) => expect(word).toBe(WORDS[(first + index) % WORDS.length]))
  })

  it("changes a token's word as it leaves the stack, not on the arc", () => {
    for (let timeS = 0; timeS < STEP_S * 6; timeS += STEP_S / 40) {
      for (let slot = 0; slot < TOKENS; slot += 1) {
        const arc = arcProgress(tokenS(slot, timeS))
        if (arc > 0.05 && arc < 0.95) expect(wordOf(slot, timeS + STEP_S / 200)).toBe(wordOf(slot, timeS))
      }
    }
  })

  it("ripples fade out before the next token", () => {
    expect(rippleGlow(0, 0)).toBeCloseTo(1, 1)
    for (let r = 0; r <= 1; r += 0.1) expect(rippleGlow(0.999, r)).toBeLessThan(0.01)
  })
})

describe("inference hero frame", () => {
  it("is the same for the same seed", () => {
    expect(makeScene(countsFor(1000), 3)).toEqual(makeScene(countsFor(1000), 3))
  })

  it("keeps every point and level inside the view", () => {
    const scene = makeScene(countsFor(1600))
    const frame = makeFrame(scene)
    for (const timeS of [0, 1.1, STATIC_TIME_S, 33.3]) {
      composeFrame(scene, { timeS, pitch: 0.14, yaw: -0.22 }, frame)
      expect(frame.count).toBe(frame.x.length)
      for (let index = 0; index < frame.count; index += 1) {
        expect(frame.x[index]).toBeGreaterThan(0)
        expect(frame.x[index]).toBeLessThan(VIEW_UNITS)
        expect(frame.y[index]).toBeGreaterThan(0)
        expect(frame.y[index]).toBeLessThan(VIEW_UNITS)
        expect(frame.level[index]).toBeLessThan(LEVELS)
      }
      expect(frame.heads).toHaveLength(TOKENS)
    }
  })

  it("draws fewer points on small screens", () => {
    const small = makeScene(countsFor(390))
    const large = makeScene(countsFor(1920))
    expect(makeFrame(small).x.length).toBeLessThan(makeFrame(large).x.length - 400)
  })

  it("has inks with alphas in range", () => {
    for (let level = 0; level < LEVELS; level += 1) {
      const ink = inkOf(level)
      expect(ink.alpha).toBeGreaterThan(0)
      expect(ink.alpha).toBeLessThanOrEqual(1)
      expect(ink.mix).toBeGreaterThanOrEqual(0)
      expect(ink.mix).toBeLessThanOrEqual(1)
    }
  })

  it("opens on a full frame with words on the arc", () => {
    const frame = staticFrame()
    expect(frame.paths.length).toBeGreaterThan(10)
    expect(frame.labels.length).toBeGreaterThanOrEqual(2)
    expect(frame.heads.some(head => head.heat > 0.8)).toBe(true)
  })
})

describe("inference hero layer outline", () => {
  it("is evenly spaced round a rounded square", () => {
    const count = 120
    const points = Array.from({ length: count }, (_, index) => outlinePoint(index / count))
    const gaps = points.map((point, index) => {
      const next = points[(index + 1) % count]
      return Math.hypot(next[0] - point[0], next[1] - point[1])
    })
    expect(Math.max(...gaps) / Math.min(...gaps)).toBeLessThan(1.1)
    // Every point is on the square's sides or on a corner's quarter circle.
    for (const [u, v] of points) {
      const onSide = Math.abs(Math.max(Math.abs(u), Math.abs(v)) - 1) < 1e-9
      const onCorner = Math.abs(Math.hypot(Math.abs(u) - 0.65, Math.abs(v) - 0.65) - 0.35) < 1e-9
      expect(onSide || onCorner).toBe(true)
    }
    for (const [u, v] of points) expect(Math.max(Math.abs(u), Math.abs(v))).toBeLessThanOrEqual(1 + 1e-9)
  })
})
