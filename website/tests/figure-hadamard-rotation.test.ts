import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import HadamardRotationFigure from "../src/components/figures/hadamard-rotation/hadamard-rotation.figure"
import { hadamardGeometry } from "../src/components/figures/hadamard-rotation/geometry"
import {
  FITS,
  LANE_COUNT,
  MODEL_BLOCK,
  PTQ1_0_BLOCK_BYTES,
  PTQ1_0_BLOCK_WEIGHTS,
  ROTATED,
  SIGNED,
  hadamard,
  readoutFor,
  roundTernary,
} from "../src/components/figures/hadamard-rotation/model"
import { inside, overlaps, textRect } from "../src/components/inference/figure-geometry"

const norm = (values: number[]) => Math.hypot(...values)

describe("hadamard-rotation model", () => {
  it("matches phobos-gguf's block sizes", () => {
    const backend = readFileSync(resolve(process.cwd(), "../phobos-gguf/src/backend/mod.rs"), "utf8")
    expect(backend).toContain(`pub const HADAMARD_BLOCK: usize = ${MODEL_BLOCK};`)
    const ptq = readFileSync(resolve(process.cwd(), "../phobos-gguf/src/quant/ptq1_0.rs"), "utf8")
    expect(ptq).toContain(`FILE_BLOCK: usize = ${PTQ1_0_BLOCK_WEIGHTS};`)
    expect(ptq).toContain(`FILE_BLOCK_BYTES: usize = ${PTQ1_0_BLOCK_BYTES};`)
  })

  it("rotates without changing lengths and undoes itself", () => {
    expect(norm(ROTATED)).toBeCloseTo(norm(SIGNED), 9)
    const back = hadamard(ROTATED)
    back.forEach((value, lane) => expect(value).toBeCloseTo(SIGNED[lane], 9))
  })

  it("keeps the product: (W H)(H x) = W x", () => {
    const x = Array.from({ length: LANE_COUNT }, (_, lane) => Math.sin(lane + 1))
    const dot = (a: number[], b: number[]) => a.reduce((sum, value, lane) => sum + value * b[lane], 0)
    expect(dot(hadamard(SIGNED), hadamard(x))).toBeCloseTo(dot(SIGNED, x), 9)
  })

  it("rounds with less error after rotating, as the readouts say", () => {
    expect(FITS.on.fit.error).toBeLessThan(FITS.off.fit.error)
    expect(FITS.off.fit.codes.filter(code => code === 0)).toHaveLength(15)
    expect(readoutFor("off")).toContain(`${Math.round(FITS.off.fit.error * 100)}%`)
    expect(readoutFor("on")).toContain(`${Math.round(FITS.on.fit.error * 100)}%`)
    for (const rotation of ["off", "on"] as const) expect(readoutFor(rotation)).toMatch(/^[\x20-\x7e]+$/)
  })

  it("finds the scale that rounds a ternary row exactly", () => {
    const fit = roundTernary([2, -2, 0, 2])
    expect(fit.error).toBeCloseTo(0, 9)
    expect(fit.codes).toEqual([1, -1, 0, 1])
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`hadamard-rotation geometry, ${layout}`, () => {
    const geometry = hadamardGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps bars, labels and the network on the plate and apart", () => {
      for (const rotation of ["off", "on"] as const) {
        const { values, fit } = FITS[rotation]
        values.forEach((value, lane) => {
          expect(inside(geometry.bar(geometry.rightAxis, lane, value), plateRect)).toBe(true)
          const level = geometry.rightAxis + fit.scale * geometry.unitsPerValue
          expect(level).toBeLessThan(geometry.widthUnits)
        })
      }
      FITS.off.values.forEach((value, lane) => {
        const bar = geometry.bar(geometry.leftAxis, lane, value)
        expect(inside(bar, plateRect)).toBe(true)
        expect(bar.x + bar.width).toBeLessThan(geometry.network.left)
      })
      const { labels } = geometry
      const all = [labels.left, labels.network, labels.right, labels.signs, labels.stages, ...labels.note].map(textRect)
      for (const rect of all) expect(inside(rect, plateRect), JSON.stringify(rect)).toBe(true)
      for (let a = 0; a < all.length; a += 1) {
        for (let b = a + 1; b < all.length; b += 1) expect(overlaps(all[a], all[b]), `${a} ${b}`).toBe(false)
      }
      for (const note of labels.note.map(textRect)) expect(note.y).toBeGreaterThan(geometry.codeLabelY + 6)
    })

    it("renders statically both ways", () => {
      for (const rotation of ["off", "on"] as const) {
        const render = HadamardRotationFigure.renderStatic(layout, { rotation })
        expect(render.readout).toBe(readoutFor(rotation))
        expect(render.drawing).not.toMatch(/NaN/)
      }
    })
  })
}
