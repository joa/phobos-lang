import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import QdotMappingFigure, { READOUTS } from "../src/components/figures/qdot-mapping/qdot-mapping.figure"
import {
  BYTES_PER_LANE,
  LANES,
  SHUFFLE_MASKS,
  STEP_BYTES,
  STEPS,
  butterfly,
  partnerOf,
  qdotGeometry,
} from "../src/components/figures/qdot-mapping/geometry"
import { inside, overlaps, textRect } from "../src/components/inference/figure-geometry"

describe("qdot-mapping logic", () => {
  it("reads the lanes matvec the way phobos-gguf writes it", () => {
    const source = readFileSync(resolve(process.cwd(), "../phobos-gguf/src/backend/device/kernels/quant.rs"), "utf8")
    expect(source).toContain(`for k in range(ctx.lane * ${BYTES_PER_LANE}, K, ctx.lanes * ${BYTES_PER_LANE})`)
    expect(source).toContain("sum<over ctx.lane>(acc)")
    expect(STEP_BYTES).toBe(512)
    expect(STEPS).toBe(2)
  })

  it("folds every lane's partial sum into every lane in five xor stages", () => {
    expect(SHUFFLE_MASKS).toEqual([16, 8, 4, 2, 1])
    const stages = butterfly()
    expect(stages).toHaveLength(5)
    for (const held of stages[stages.length - 1]) expect(held.size).toBe(LANES)
    // Each stage doubles what a lane holds.
    stages.forEach((stage, index) => {
      for (const held of stage) expect(held.size).toBe(2 ** (index + 1))
    })
    for (let lane = 0; lane < LANES; lane += 1) expect(partnerOf(partnerOf(lane, 8), 8)).toBe(lane)
  })

  it("writes ASCII readouts", () => {
    for (const text of Object.values(READOUTS)) expect(text).toMatch(/^[\x20-\x7e]+$/)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`qdot-mapping geometry, ${layout}`, () => {
    const geometry = qdotGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps everything on the plate and the labels apart", () => {
      expect(inside(geometry.matrix.bounds, plateRect)).toBe(true)
      for (const cell of [...geometry.output.cells, ...geometry.lanes.cells, ...geometry.activation.blocks]) {
        expect(inside(cell, plateRect)).toBe(true)
      }
      for (const cell of geometry.output.cells) expect(overlaps(cell, geometry.matrix.bounds)).toBe(false)
      const longest = { ...geometry.stageLabel, text: "lane 0 writes the sum" }
      const labels = [
        geometry.activation.label,
        geometry.matrix.label,
        geometry.output.label,
        geometry.lanes.label,
        longest,
      ].map(textRect)
      for (const rect of labels) expect(inside(rect, plateRect), JSON.stringify(rect)).toBe(true)
      for (let a = 0; a < labels.length; a += 1) {
        for (let b = a + 1; b < labels.length; b += 1) expect(overlaps(labels[a], labels[b]), `${a} ${b}`).toBe(false)
      }
      for (const rect of labels) expect(overlaps(rect, geometry.matrix.bounds)).toBe(false)
    })

    it("puts every lane tick of a step inside that step's band", () => {
      for (let step = 0; step < STEPS; step += 1) {
        const band = geometry.stepRect(step)
        for (let lane = 0; lane < LANES; lane += 1) {
          const tick = geometry.laneTick(step, lane)
          expect(tick.x).toBeGreaterThanOrEqual(band.x)
          expect(tick.x).toBeLessThanOrEqual(band.x + band.width)
        }
      }
    })

    it("renders statically for both mappings", () => {
      for (const mapping of ["thread", "warp"] as const) {
        const render = QdotMappingFigure.renderStatic(layout, { mapping })
        expect(render.readout).toBe(READOUTS[mapping])
        expect(render.drawing).not.toMatch(/NaN/)
      }
    })
  })
}
