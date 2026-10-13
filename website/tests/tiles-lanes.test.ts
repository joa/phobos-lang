import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { fitView, project, vec, type Camera } from "../src/components/home/tiles-lanes/camera"
import { sceneData, xorMask } from "../src/components/home/tiles-lanes/data"
import { tokenize } from "../src/components/home/tiles-lanes/highlight"
import {
  lanesBeat,
  tilesBeat,
  VIEW_HEIGHT_UNITS,
  VIEW_WIDTH_UNITS,
  type Beat,
} from "../src/components/home/tiles-lanes/scene"
import type { Trace } from "../src/lib/viz/trace"

const trace = (name: string): Trace =>
  JSON.parse(readFileSync(resolve(process.cwd(), `src/data/traces/${name}.sm_75.json`), "utf8")) as Trace
const gemm = trace("gemm_fp32")
const matvec = trace("lanes_matvec")
const data = sceneData(gemm, matvec)

describe("tiles-lanes data", () => {
  it("reads the gemm's tiles from its trace", () => {
    expect(data.tiles).toMatchObject({ tileM: 32, tileN: 32, tileK: 4, threads: 256, M: 64, N: 64, K: 64 })
    expect(data.tiles.kts.slice(0, 3)).toEqual([0, 4, 8])
  })

  it("reads the lanes matvec's runs and xor stages from its trace", () => {
    expect(data.lanes).toMatchObject({ lanes: 32, runElems: 4, kSteps: [0, 128] })
    expect(data.lanes.masks).toEqual([16, 8, 4, 2, 1])
  })

  it("refuses partners that are not an xor", () => {
    expect(xorMask([1, 0, 3, 2])).toBe(1)
    expect(() => xorMask([1, 2, 3, 0])).toThrow()
  })

  it("shows real source lines, with the marks on the right ones", () => {
    for (const [code, source] of [
      [data.tilesCode, gemm.source.text],
      [data.lanesCode, matvec.source.text],
    ] as const) {
      expect(code.lines.length).toBeGreaterThanOrEqual(3)
      expect(code.lines.length).toBeLessThanOrEqual(5)
      for (const line of code.lines) expect(source).toContain(line.trim())
    }
    expect(data.tilesCode.lines[data.tilesCode.marks.dot]).toContain("acc += dot(a, b)")
    expect(data.lanesCode.lines[data.lanesCode.marks.sum]).toContain("sum<over ctx.lane>(acc)")
  })

  it("highlights without losing text", () => {
    for (const line of [...data.tilesCode.lines, ...data.lanesCode.lines])
      expect(
        tokenize(line)
          .map(token => token.text)
          .join("")
      ).toBe(line)
    expect(tokenize("var a = 4").map(token => token.kind)).toEqual(["kw", "plain", "num"])
  })
})

describe("tiles-lanes camera", () => {
  const camera: Camera = { target: vec(0, 0, 0), yawDeg: 30, pitchDeg: 20, distance: 100, focal: 100 }
  const view = { scale: 2, cx: 300, cy: 200 }

  it("puts the target at the view's center", () => {
    expect(project(vec(0, 0, 0), camera, view)).toMatchObject({ x: 300, y: 200, near: 1 })
  })

  it("draws nearer points larger, and up as up", () => {
    const level: Camera = { ...camera, yawDeg: 0, pitchDeg: 0 }
    expect(project(vec(0, 0, 10), level, view).near).toBeGreaterThan(1)
    expect(project(vec(0, 0, -10), level, view).near).toBeLessThan(1)
    expect(project(vec(0, 10, 0), camera, view).y).toBeLessThan(200)
  })

  it("fits points into a box", () => {
    const points = [vec(-50, 0, -50), vec(50, 0, 50), vec(0, 40, 0)]
    const box = { x: 10, y: 20, width: 400, height: 300 }
    const fitted = fitView(points, camera, box)
    for (const point of points) {
      const p = project(point, camera, fitted)
      expect(p.x).toBeGreaterThanOrEqual(box.x - 1e-6)
      expect(p.x).toBeLessThanOrEqual(box.x + box.width + 1e-6)
      expect(p.y).toBeGreaterThanOrEqual(box.y - 1e-6)
      expect(p.y).toBeLessThanOrEqual(box.y + box.height + 1e-6)
    }
  })
})

describe("tiles-lanes beats", () => {
  const beats: Beat[] = [tilesBeat(data.tiles), lanesBeat(data.lanes)]

  it("keeps every frame finite, the right size and inside the stage", () => {
    for (const beat of beats) {
      const { camera, view } = beat.rest
      const still = beat.still(camera, view)
      for (let tMs = 0; tMs <= beat.durationMs; tMs += 97) {
        const frame = beat.frame(tMs, camera, view)
        expect(frame.particles).toHaveLength(beat.particles.length)
        expect(frame.glows).toHaveLength(beat.glows.length)
        for (const particle of [...frame.particles, ...frame.glows]) {
          for (const value of Object.values(particle)) expect(Number.isFinite(value)).toBe(true)
          if (particle.opacity > 0) {
            expect(particle.x).toBeGreaterThan(0)
            expect(particle.x).toBeLessThan(VIEW_WIDTH_UNITS)
            expect(particle.y).toBeGreaterThan(0)
            expect(particle.y).toBeLessThan(VIEW_HEIGHT_UNITS)
          }
        }
        for (const label of Object.values(frame.labels)) {
          expect(label.x).toBeGreaterThan(0)
          expect(label.x).toBeLessThan(VIEW_WIDTH_UNITS)
          expect(label.y).toBeGreaterThan(0)
          expect(label.y).toBeLessThan(VIEW_HEIGHT_UNITS)
        }
        for (const [key, path] of Object.entries(frame.paths)) {
          expect(beat.paths.some(spec => spec.key === key)).toBe(true)
          expect(path.d ?? still[key]?.d).toBeDefined()
          expect(path.d ?? "").not.toContain("NaN")
        }
      }
      for (const spec of beat.paths) expect(still[spec.key] ?? beat.frame(0, camera, view).paths[spec.key]).toBeDefined()
      for (const label of beat.labels) expect(beat.frame(0, camera, view).labels[label.key]).toBeDefined()
    }
  })

  it("loops: a beat starts and ends with nothing in flight", () => {
    for (const beat of beats) {
      const { camera, view } = beat.rest
      for (const tMs of [0, beat.durationMs]) {
        const frame = beat.frame(tMs, camera, view)
        expect(frame.particles.every(particle => particle.opacity === 0)).toBe(true)
      }
    }
  })

  it("walks the lanes through the trace's xor stages in order", () => {
    const beat = beats[1]
    const { camera, view } = beat.rest
    const seen: number[] = []
    for (let tMs = 0; tMs <= beat.durationMs; tMs += 50) {
      const index = beat.frame(tMs, camera, view).metaIndex
      if (index >= 0 && index < data.lanes.masks.length && seen.at(-1) !== index) seen.push(index)
    }
    expect(seen.map(index => data.lanes.masks[index])).toEqual([16, 8, 4, 2, 1])
  })
})
