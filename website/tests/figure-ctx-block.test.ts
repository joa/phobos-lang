import { describe, expect, it } from "vitest"
import CtxBlockFigure from "../src/components/figures/ctx-block/ctx-block.figure"
import {
  blockGeometry,
  blockModel,
  DEFAULT_M,
  DEFAULT_N,
  isWholeTile,
  MAX_BLOCKS,
  parsePick,
  PANEL_LINES,
  readoutFor,
  sliceText,
} from "../src/components/figures/ctx-block/geometry"
import { textWidthUnits } from "../src/components/figures/ctx-hierarchy/text"

describe("ctx-block model", () => {
  it("rounds the grid up and clamps the picked CTA into it", () => {
    const model = blockModel(DEFAULT_M, DEFAULT_N, { x: 9, y: 9 }, false)
    expect([model.blocksX, model.blocksY, model.blocksZ]).toEqual([5, 7, 1])
    expect(model.pick).toEqual({ x: 4, y: 6 })
    expect(sliceText(model)).toBe("C[256:320, 384:448]")
    expect([model.rowsInside, model.colsInside]).toEqual([32, 32])
    expect(isWholeTile(model)).toBe(false)
  })

  it("maps ctx.block.x to rows and ctx.block.y to columns, as gemm does", () => {
    const model = blockModel(DEFAULT_M, DEFAULT_N, parsePick("1,2"), false)
    expect(sliceText(model)).toBe("C[64:128, 128:192]")
    expect(isWholeTile(model)).toBe(true)
  })

  it("names the grid and the tile in the readout", () => {
    expect(readoutFor(blockModel(288, 416, { x: 1, y: 2 }, false))).toBe(
      "C[288, 416] in 64 x 64 tiles needs a grid of 5 x 7 CTAs. CTA (1, 2) owns C[64:128, 128:192], a whole tile."
    )
    expect(readoutFor(blockModel(288, 416, { x: 4, y: 0 }, true))).toContain("5 x 7 x 3 CTAs")
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`ctx-block geometry, ${layout}`, () => {
    const geometry = blockGeometry(layout)
    const gridUnits = MAX_BLOCKS * geometry.tileUnits
    const ghostUnits = 2 * geometry.ghostStepUnits

    it("keeps the largest grid and its batches on the plate", () => {
      const last = geometry.tileRect(MAX_BLOCKS - 1, MAX_BLOCKS - 1)
      expect(last.x + last.width + ghostUnits).toBeLessThan(geometry.widthUnits - 8)
      expect(last.y + last.height).toBeLessThan(geometry.heightUnits - 8)
      expect(geometry.originY - ghostUnits).toBeGreaterThan(geometry.colIndex(0).y + 6)
    })

    it("keeps the axis labels and the panel clear of the grid", () => {
      const cols = geometry.axisCols
      expect(cols.x + textWidthUnits(cols.text ?? "")).toBeLessThan(geometry.widthUnits - 8)
      expect(cols.y).toBeGreaterThan(32)
      expect(textWidthUnits(geometry.axisRows.text ?? "")).toBeLessThan(gridUnits)
      expect(geometry.axisRows.x - 6).toBeGreaterThan(0)
      expect(geometry.panel).toHaveLength(PANEL_LINES)
      const widest = textWidthUnits("ctx.blocks.z = 3, one grid per batch")
      for (const line of geometry.panel) {
        expect(line.x + widest).toBeLessThan(geometry.widthUnits - 8)
        expect(line.y).toBeLessThan(geometry.heightUnits - 8)
        if (layout === "wide") expect(line.x).toBeGreaterThan(geometry.originX + gridUnits + ghostUnits)
        else expect(line.y).toBeGreaterThan(geometry.originY + gridUnits + 12)
      }
    })
  })
}

describe("ctx-block static render", () => {
  it("settles on the default grid and pick in both layouts", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = CtxBlockFigure.renderStatic(layout, { m: 288, n: 416, cta: "1,2", z: "off" })
      expect(render.readout).toBe(readoutFor(blockModel(288, 416, { x: 1, y: 2 }, false)))
      expect(render.drawing).toContain('aria-label="CTA (1, 2)" aria-pressed="true"')
      expect(render.drawing).toContain("ctx.blocks.x = ceil(288 / 64) = 5")
      expect(render.drawing.match(/chart-box is-hot/g)?.length).toBe(1)
    }
  })
})
