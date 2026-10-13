import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { fitView, project, vec, type Camera } from "../src/components/home/tiles-lanes/camera"
import { sceneData, xorMask } from "../src/components/home/tiles-lanes/data"
import { tokenize } from "../src/components/home/tiles-lanes/highlight"
import { mixedBeat } from "../src/components/home/tiles-lanes/mixed"
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
const staged = trace("staged_matvec")
const data = sceneData(gemm, matvec, staged)

describe("tiles-lanes data", () => {
  it("reads the gemm's tiles from its trace", () => {
    expect(data.tiles).toMatchObject({ tileM: 32, tileN: 32, tileK: 4, threads: 256, M: 64, N: 64, K: 64 })
    expect(data.tiles.kts.slice(0, 3)).toEqual([0, 4, 8])
  })

  it("reads the lanes matvec's runs and xor stages from its trace", () => {
    expect(data.lanes).toMatchObject({ lanes: 32, runElems: 4, kSteps: [0, 128] })
    expect(data.lanes.masks).toEqual([16, 8, 4, 2, 1])
  })

  it("reads the staged matvec's tile copy, its lanes' runs and xor stages from its trace", () => {
    expect(data.mixed).toMatchObject({
      K: 256,
      threads: 128,
      lanes: 32,
      warps: 4,
      tileElems: 256,
      copyWidth: 4,
      copyThreads: 64,
      copyBarrier: true,
      runElems: 4,
      kSteps: [0, 128],
    })
    expect(data.mixed.masks).toEqual([16, 8, 4, 2, 1])
    // Lane l reads xs and its row of W at the same column, l runs into the k step.
    data.mixed.kSteps.forEach((k, step) => {
      expect(data.mixed.tileRuns[step]).toEqual(Array.from({ length: 32 }, (_, lane) => k + 4 * lane))
      expect(data.mixed.weightRuns[step]).toEqual(data.mixed.tileRuns[step])
    })
  })

  it("refuses partners that are not an xor", () => {
    expect(xorMask([1, 0, 3, 2])).toBe(1)
    expect(() => xorMask([1, 2, 3, 0])).toThrow()
  })

  it("shows real source lines, with the marks on the right ones", () => {
    for (const [code, source, most] of [
      [data.tilesCode, gemm.source.text, 5],
      [data.lanesCode, matvec.source.text, 5],
      [data.mixedCode, staged.source.text, 8],
    ] as const) {
      expect(code.lines.length).toBeGreaterThanOrEqual(3)
      expect(code.lines.length).toBeLessThanOrEqual(most)
      for (const line of code.lines) expect(source).toContain(line.trim())
    }
    expect(data.tilesCode.lines[data.tilesCode.marks.dot]).toContain("acc += dot(a, b)")
    expect(data.lanesCode.lines[data.lanesCode.marks.sum]).toContain("sum<over ctx.lane>(acc)")
    // One excerpt, both levels: a whole-tile copy, then lane runs read from that tile.
    const mixed = data.mixedCode
    expect(mixed.lines[mixed.marks.stage]).toBe("var xs = x[0 :+ 1, :]")
    expect(mixed.lines[mixed.marks.at]).toContain("ctx.lane")
    expect(mixed.lines[mixed.marks.acc]).toContain("xs[0, at :+ 4]")
    expect(mixed.lines[mixed.marks.sum]).toContain("sum<over ctx.lane>(acc)")
    expect(mixed.marks.stage).toBeLessThan(mixed.marks.at)
    expect(mixed.marks.at).toBeLessThan(mixed.marks.sum)
  })

  it("highlights without losing text", () => {
    for (const line of [...data.tilesCode.lines, ...data.lanesCode.lines, ...data.mixedCode.lines])
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
  const beats: Beat[] = [tilesBeat(data.tiles), lanesBeat(data.lanes), mixedBeat(data.mixed)]

  it("keeps every frame finite, the right size and inside the stage", () => {
    for (const beat of beats) {
      for (let tMs = 0; tMs <= beat.durationMs; tMs += 97) {
        const { camera, view } = beat.shot(tMs)
        const still = beat.still(camera, view)
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
      const { camera, view } = beat.shot(0)
      const still = beat.still(camera, view)
      for (const spec of beat.paths) expect(still[spec.key] ?? beat.frame(0, camera, view).paths[spec.key]).toBeDefined()
      for (const label of beat.labels) expect(beat.frame(0, camera, view).labels[label.key]).toBeDefined()
    }
  })

  it("keeps every path of the frame the page is built with inside the stage", () => {
    // The SVG lets paths overflow, so one outside the viewBox would draw over the code beside it. A
    // dot is a point followed by h0, which carries no coordinate.
    const MARGIN_UNITS = 8
    for (const beat of beats) {
      const { camera, view } = beat.shot(beat.settledMs)
      const paths = { ...beat.still(camera, view), ...beat.frame(beat.settledMs, camera, view).paths }
      for (const path of Object.values(paths)) {
        const numbers = (path.d ?? "").replaceAll("h0", "").match(/-?\d+(?:\.\d+)?/g)?.map(Number) ?? []
        for (let at = 0; at + 1 < numbers.length; at += 2) {
          expect(numbers[at]).toBeGreaterThan(-MARGIN_UNITS)
          expect(numbers[at]).toBeLessThan(VIEW_WIDTH_UNITS + MARGIN_UNITS)
          expect(numbers[at + 1]).toBeGreaterThan(-MARGIN_UNITS)
          expect(numbers[at + 1]).toBeLessThan(VIEW_HEIGHT_UNITS + MARGIN_UNITS)
        }
      }
    }
  })

  it("keeps the particles within budget", () => {
    for (const beat of beats) expect(beat.particles.length).toBeLessThanOrEqual(192)
  })

  it("loops: a beat starts and ends with nothing in flight", () => {
    for (const beat of beats) {
      for (const tMs of [0, beat.durationMs]) {
        const { camera, view } = beat.shot(tMs)
        const frame = beat.frame(tMs, camera, view)
        expect(frame.particles.every(particle => particle.opacity === 0)).toBe(true)
      }
    }
  })

  it("walks the lanes through the trace's xor stages in order", () => {
    const beat = beats[1]
    const { camera, view } = beat.shot(0)
    const seen: number[] = []
    for (let tMs = 0; tMs <= beat.durationMs; tMs += 50) {
      const index = beat.frame(tMs, camera, view).metaIndex
      if (index >= 0 && index < data.lanes.masks.length && seen.at(-1) !== index) seen.push(index)
    }
    expect(seen.map(index => data.lanes.masks[index])).toEqual([16, 8, 4, 2, 1])
  })

  it("plays the mixed beat as its code reads: the tile copy, the lane runs, then the fold", () => {
    const beat = beats[2]
    const marks: string[] = []
    const metas: number[] = []
    for (let tMs = 0; tMs <= beat.durationMs; tMs += 25) {
      const { camera, view } = beat.shot(tMs)
      const frame = beat.frame(tMs, camera, view)
      if (frame.mark !== null && marks.at(-1) !== frame.mark) marks.push(frame.mark)
      if (metas.at(-1) !== frame.metaIndex) metas.push(frame.metaIndex)
    }
    expect(marks).toEqual(["stage", "at", "acc", "at", "acc", "sum"])
    expect(metas).toEqual([-1, 0, 1, 2, 3, 4])
  })

  it("moves the mixed beat's camera down from the CTA's floors to the warp between the copy and the loads", () => {
    const beat = beats[2]
    const start = beat.shot(0)
    const end = beat.shot(beat.durationMs)
    expect(end.camera.distance).toBeLessThan(start.camera.distance)
    expect(end.camera.pitchDeg).toBeLessThan(start.camera.pitchDeg)
    expect(beat.shot(beat.settledMs)).toEqual(end)
    // The copy flies under the overview.
    const copying = beat.frame(1000, start.camera, start.view)
    expect(beat.shot(1000)).toEqual(start)
    expect(copying.particles.some(particle => particle.opacity > 0)).toBe(true)
  })
})
