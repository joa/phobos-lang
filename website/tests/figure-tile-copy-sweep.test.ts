import { describe, expect, it } from "vitest"
import { LABEL_UNITS } from "../src/lib/figure/layout"
import TileCopySweepFigure from "../src/components/figures/tile-copy-sweep/tile-copy-sweep.figure"
import {
  footprintBytes,
  liveAfter,
  PROGRAM,
  RULER_BYTES,
  RUN,
  runOf,
  STEPS,
  threadOf,
  THREADS,
  TILE_BYTES,
  TILE_COLS,
  TILE_ROWS,
  tileCopyGeometry,
  type Rect,
} from "../src/components/figures/tile-copy-sweep/geometry"
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

describe("tile-copy-sweep model", () => {
  it("gives every element of the tile to exactly one thread, 16 bytes each", () => {
    const seen = new Set<string>()
    for (let thread = 0; thread < THREADS; thread += 1) {
      const { row, col } = runOf(thread)
      for (let index = 0; index < RUN; index += 1) {
        expect(threadOf(row, col + index)).toBe(thread)
        seen.add(`${row},${col + index}`)
      }
    }
    expect(seen.size).toBe(TILE_ROWS * TILE_COLS)
    expect(RUN * 4).toBe(16)
    expect(TILE_BYTES).toBe(512)
  })

  it("reuses a's bytes for b only after a's last use", () => {
    expect(liveAfter("var", "copy").map(a => a.name)).toEqual(["a"])
    expect(liveAfter("var", "read").map(a => a.name)).toEqual(["a", "s"])
    const reuse = liveAfter("var", "reuse")
    expect(reuse.map(a => a.name)).toEqual(["b", "s"])
    expect(reuse[0].offsetBytes).toBe(0)
    for (const step of STEPS) {
      const live = liveAfter("var", step)
      for (const a of live)
        for (const b of live)
          if (a !== b)
            expect(a.offsetBytes + a.sizeBytes <= b.offsetBytes || b.offsetBytes + b.sizeBytes <= a.offsetBytes).toBe(
              true
            )
    }
    expect(footprintBytes("var")).toBe(544)
    expect(footprintBytes("var")).toBeLessThanOrEqual(RULER_BYTES)
  })

  it("gives a view no bytes", () => {
    expect(liveAfter("let", "copy")).toEqual([])
    expect(liveAfter("let", "read")).toEqual([])
    expect(footprintBytes("let")).toBe(TILE_BYTES)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`tile-copy-sweep geometry, ${layout}`, () => {
    const geometry = tileCopyGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps the blocks on the plate and apart", () => {
      const sBounds = {
        ...geometry.sCells[0],
        height: geometry.sCells[7].y + geometry.sCells[7].height - geometry.sCells[0].y,
      }
      const blocks = [geometry.tensorBounds, geometry.tileBounds, sBounds]
      for (const block of blocks) expect(inside(block, plateRect)).toBe(true)
      expect(overlaps(blocks[0], blocks[1])).toBe(false)
      expect(overlaps(blocks[1], blocks[2])).toBe(false)
      expect(
        inside(geometry.slice, {
          ...geometry.tensorBounds,
          x: geometry.tensorBounds.x - 2,
          y: geometry.tensorBounds.y - 2,
          width: geometry.tensorBounds.width + 4,
          height: geometry.tensorBounds.height + 4,
        })
      ).toBe(true)
      for (const hit of geometry.runHits)
        expect(
          inside(hit, {
            ...geometry.tileBounds,
            x: geometry.tileBounds.x - 1,
            y: geometry.tileBounds.y - 1,
            width: geometry.tileBounds.width + 2,
            height: geometry.tileBounds.height + 2,
          })
        ).toBe(true)
    })

    it("fits the program and the ruler", () => {
      const longest = Math.max(...[...PROGRAM.var, ...PROGRAM.let].map(line => textWidthUnits(line, LABEL_UNITS)))
      for (const line of geometry.program) {
        const box = { x: line.text.x, y: line.text.y - 7, width: longest, height: 14 }
        expect(box.x + box.width).toBeLessThanOrEqual(geometry.widthUnits - 8)
        expect(overlaps(box, geometry.tensorBounds)).toBe(false)
        expect(overlaps(box, geometry.tileBounds)).toBe(false)
        if (layout === "wide") expect(box.x + box.width).toBeLessThan(geometry.ruler.x)
      }
      expect(geometry.rulerX(RULER_BYTES)).toBeLessThanOrEqual(geometry.widthUnits - 14)
      expect(geometry.ruler.tickY).toBeLessThan(geometry.heightUnits - 6)
      for (const point of geometry.wire.points) expect(inside({ ...point, width: 0, height: 0 }, plateRect)).toBe(true)
    })
  })
}

describe("tile-copy-sweep static render", () => {
  it("settles on a live tile and its rowsum", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = TileCopySweepFigure.renderStatic(layout, { bind: "var", step: "read", thread: "9" })
      expect(render.readout).toMatch(/^rowsum\(a\) is a's last use/)
      expect(render.drawing).toContain("shared  a  8 x 16 f32, 512 B")
    }
  })

  it("draws no tile for a view", () => {
    const render = TileCopySweepFigure.renderStatic("wide", { bind: "let", step: "copy", thread: "9" })
    expect(render.readout).toContain("takes no shared memory")
    expect(render.drawing).toContain("a view: 0 bytes")
  })
})
