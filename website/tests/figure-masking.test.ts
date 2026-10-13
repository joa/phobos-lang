import { describe, expect, it } from "vitest"
import MaskingFigure from "../src/components/figures/masking/masking.figure"
import {
  elementMasked,
  extentsOf,
  GRID_COLS,
  GRID_ROWS,
  M,
  maskingGeometry,
  N,
  parseBlock,
  TILE,
  tileStats,
  type Rect,
} from "../src/components/figures/masking/geometry"
import { EXTENT_MULTIPLE } from "../src/components/figures/ctx-hierarchy/rules"
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

describe("masking model", () => {
  it("uses extents SPEC allows, ragged against the tile", () => {
    expect(M % EXTENT_MULTIPLE).toBe(0)
    expect(N % EXTENT_MULTIPLE).toBe(0)
    expect(M % TILE).not.toBe(0)
    expect([GRID_ROWS, GRID_COLS]).toEqual([3, 4])
  })

  it("masks the corner tile past both edges", () => {
    expect(tileStats({ x: 2, y: 3 }, false)).toEqual({ rows: 4, cols: 4, inBounds: 16, masked: 48 })
    expect(tileStats({ x: 2, y: 0 }, false)).toEqual({ rows: 4, cols: 8, inBounds: 32, masked: 32 })
    expect(tileStats({ x: 1, y: 1 }, false).masked).toBe(0)
    expect(elementMasked({ x: 2, y: 3 }, 3, 3, false)).toBe(false)
    expect(elementMasked({ x: 2, y: 3 }, 4, 0, false)).toBe(true)
  })

  it("masks nothing under @aligned", () => {
    expect(extentsOf(true)).toEqual({ m: 24, n: 32 })
    for (let x = 0; x < GRID_ROWS; x += 1)
      for (let y = 0; y < GRID_COLS; y += 1) expect(tileStats({ x, y }, true).masked).toBe(0)
  })

  it("falls back to the corner for a bad pick", () => {
    expect(parseBlock("1,2")).toEqual({ x: 1, y: 2 })
    expect(parseBlock("9,9")).toEqual({ x: 2, y: 3 })
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`masking geometry, ${layout}`, () => {
    const geometry = maskingGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps the tensor, the tile and the labels on the plate and apart", () => {
      expect(inside(geometry.tensorBounds, plateRect)).toBe(true)
      expect(inside(geometry.sharedBounds, plateRect)).toBe(true)
      expect(overlaps(geometry.tensorBounds, geometry.sharedBounds)).toBe(false)
      expect(inside(geometry.extentBox(extentsOf(true)), plateRect)).toBe(true)
      const texts = [
        { ...geometry.code, width: textWidthUnits(geometry.code.text) },
        { ...geometry.promise, width: textWidthUnits(geometry.promise.text) },
        ...geometry.notes.map(note => ({ ...note, width: textWidthUnits("read 0, stores skipped") })),
      ]
      for (const text of texts) {
        expect(text.x + text.width).toBeLessThanOrEqual(geometry.widthUnits - 8)
        expect(text.y).toBeLessThan(geometry.heightUnits - 8)
        const box = { x: text.x, y: text.y - 7, width: text.width, height: 14 }
        expect(overlaps(box, geometry.sharedBounds)).toBe(false)
        expect(overlaps(box, geometry.tensorBounds)).toBe(false)
      }
    })

    it("routes the wire from the tensor to the tile for every pick", () => {
      for (let x = 0; x < GRID_ROWS; x += 1) {
        for (let y = 0; y < GRID_COLS; y += 1) {
          const wire = geometry.wire({ x, y })
          for (const point of wire.points) expect(inside({ ...point, width: 0, height: 0 }, plateRect)).toBe(true)
          expect(wire.points.length).toBeLessThanOrEqual(4)
        }
      }
    })
  })
}

describe("masking static render", () => {
  it("settles on the corner tile with its masks", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = MaskingFigure.renderStatic(layout, { tile: "2,3", aligned: "off" })
      expect(render.readout).toBe(
        "block (2, 3): 16 of 64 elements are in bounds; 48 read as 0 and their stores are skipped."
      )
      // 48 masked cells in the shared tile and 48 past the edge in the tensor.
      expect(render.drawing.match(/chart-cell is-warn sp-cost/g)?.length).toBe(96)
    }
  })

  it("shows no mask under @aligned", () => {
    const render = MaskingFigure.renderStatic("wide", { tile: "2,3", aligned: "on" })
    expect(render.readout).toContain("checked once at entry")
    expect(render.drawing).not.toContain("chart-cell is-warn")
  })
})
