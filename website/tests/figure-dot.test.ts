import { describe, expect, it } from "vitest"
import { LABEL_UNITS, MIN_LABEL_UNITS } from "../src/lib/figure/layout"
import DotFigure from "../src/components/figures/dot/dot.figure"
import {
  dotGeometry,
  FRAGMENTS,
  FRAGMENTS_PER_WARP,
  fragmentWarp,
  K_STEPS,
  slicesOf,
  WARPS,
  type Rect,
} from "../src/components/figures/dot/geometry"
import { textWidthUnits } from "../src/components/figures/ctx-hierarchy/text"

function inside(inner: Rect, outer: Rect): boolean {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  )
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

describe("dot model", () => {
  it("tiles acc's 16 x 16 fragments over the CTA's 8 warps, 2 each", () => {
    expect(WARPS).toBe(8)
    expect(FRAGMENTS_PER_WARP).toBe(2)
    const counts = new Map<number, number>()
    for (let row = 0; row < FRAGMENTS; row += 1)
      for (let col = 0; col < FRAGMENTS; col += 1) {
        const warp = fragmentWarp(row, col)
        counts.set(warp, (counts.get(warp) ?? 0) + 1)
      }
    expect([...counts.keys()].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7])
    for (const count of counts.values()) expect(count).toBe(2)
  })

  it("slices A and B the way gemm's k loop does", () => {
    expect(slicesOf(1)).toEqual({ a: "A[0:64, 16:32]", b: "B[16:32, 0:64]", kt: 16 })
    expect(K_STEPS).toBe(4)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`dot geometry, ${layout}`, () => {
    const geometry = dotGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }
    const blocks = [
      geometry.globalA.bounds,
      geometry.globalB.bounds,
      geometry.sharedA.bounds,
      geometry.sharedB.bounds,
      geometry.acc.bounds,
    ]

    it("keeps the blocks on the plate and apart", () => {
      expect(geometry.heightUnits).toBeLessThanOrEqual(layout === "wide" ? 340 : 540)
      for (const block of blocks) expect(inside(block, plateRect)).toBe(true)
      for (let i = 0; i < blocks.length; i += 1)
        for (let j = i + 1; j < blocks.length; j += 1) expect(overlaps(blocks[i], blocks[j])).toBe(false)
    })

    it("arranges a left of acc and b above it", () => {
      const { sharedA, sharedB, acc } = geometry
      expect(sharedA.bounds.x + sharedA.bounds.width).toBeLessThan(acc.bounds.x)
      expect(sharedA.bounds.y).toBe(acc.bounds.y)
      expect(sharedB.bounds.y + sharedB.bounds.height).toBeLessThan(acc.bounds.y)
      expect(sharedB.bounds.x).toBe(acc.bounds.x)
    })

    it("runs every k step's wires from its panel to the shared tiles", () => {
      for (let kStep = 0; kStep < K_STEPS; kStep += 1) {
        const a = geometry.wireA(kStep)
        const b = geometry.wireB(kStep)
        expect(a.end.x).toBeLessThan(geometry.sharedA.bounds.x)
        expect(b.end.y).toBeLessThan(geometry.sharedB.bounds.y)
        for (const point of [...a.points, ...b.points])
          expect(inside({ ...point, width: 0, height: 0 }, plateRect)).toBe(true)
      }
    })

    it("keeps labels on the plate and clear of the blocks", () => {
      const labels = [
        geometry.globalA.label,
        geometry.globalB.label,
        geometry.sharedA.label,
        geometry.sharedB.label,
        geometry.acc.label,
        geometry.tensorcore,
        ...geometry.code,
      ]
      for (const at of labels) {
        // The code lines are small labels.
        const widthUnits = textWidthUnits(at.text, geometry.code.includes(at) ? MIN_LABEL_UNITS : LABEL_UNITS)
        const left = at.anchor === "end" ? at.x - widthUnits : at.x
        expect(left).toBeGreaterThanOrEqual(8)
        expect(left + widthUnits).toBeLessThanOrEqual(geometry.widthUnits - 8)
        expect(at.y).toBeGreaterThan(30)
        expect(at.y).toBeLessThan(geometry.heightUnits - 8)
        const box = { x: left, y: at.y - 6, width: widthUnits, height: 12 }
        for (const block of blocks) expect(overlaps(box, block)).toBe(false)
      }
      for (const note of geometry.notes) expect(note.y).toBeLessThan(geometry.heightUnits - 8)
    })
  })
}

describe("dot static render", () => {
  it("settles with all four k steps added", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = DotFigure.renderStatic(layout, { kt: "3", path: "fma" })
      expect(render.readout).toMatch(/^kt = 48: A\[0:64, 48:64\] and B\[48:64, 0:64\]/)
      expect(render.readout).toContain("FMA path")
      expect(render.drawing).toContain("k steps added: 4 of 4")
    }
  })

  it("names the f16 rounding on the tensor-core path", () => {
    const render = DotFigure.renderStatic("wide", { kt: "0", path: "tc" })
    expect(render.readout).toContain("rounded to f16")
    expect(render.drawing).toContain("k steps added: 1 of 4")
  })
})
