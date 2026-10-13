import { describe, expect, it } from "vitest"
import {
  ctaCopyGeometry,
  elementsOf,
  K_STEPS,
  PER_THREAD,
  sliceOf,
  THREADS,
  TILE,
  type Rect,
} from "../src/components/figures/cta-copy/geometry"
import CtaCopyFigure from "../src/components/figures/cta-copy/cta-copy.figure"
import { threadOf } from "../src/components/figures/cta-copy/geometry"

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

describe("cta-copy thread map", () => {
  it("gives every element of the tile to exactly one thread", () => {
    const seen = new Set<string>()
    for (let thread = 0; thread < THREADS; thread += 1) {
      const elements = elementsOf(thread)
      expect(elements).toHaveLength(PER_THREAD)
      for (const [row, col] of elements) {
        expect(threadOf(row, col)).toBe(thread)
        seen.add(`${row},${col}`)
      }
    }
    expect(seen.size).toBe(TILE * TILE)
  })

  it("slices A the way gemm does", () => {
    expect(sliceOf(0)).toEqual({ rows: [8, 16], cols: [0, 8], kt: 0 })
    expect(sliceOf(3)).toEqual({ rows: [8, 16], cols: [24, 32], kt: 24 })
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`cta-copy geometry, ${layout}`, () => {
    const geometry = ctaCopyGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps every block on the plate and apart", () => {
      const blocks = [geometry.global.bounds, geometry.registers.bounds, geometry.shared.bounds]
      for (const block of blocks) expect(inside(block, plateRect)).toBe(true)
      expect(overlaps(blocks[0], blocks[1])).toBe(false)
      expect(overlaps(blocks[1], blocks[2])).toBe(false)
      expect(overlaps(blocks[0], blocks[2])).toBe(false)
    })

    it("keeps every tile outline inside the tensor's margin", () => {
      for (let kStep = 0; kStep < K_STEPS; kStep += 1) {
        const tile = geometry.tileRect(kStep)
        expect(
          inside(tile, {
            ...geometry.global.bounds,
            x: geometry.global.bounds.x - 3,
            y: geometry.global.bounds.y - 3,
            width: geometry.global.bounds.width + 6,
            height: geometry.global.bounds.height + 6,
          })
        ).toBe(true)
      }
    })

    it("starts and ends the wires beside their blocks", () => {
      const { wireLoad, wireStore, registers, shared } = geometry
      expect(wireLoad.end.x).toBeLessThan(registers.bounds.x)
      expect(wireStore.start.x).toBeGreaterThan(registers.bounds.x + registers.bounds.width)
      expect(wireStore.end.x).toBeLessThan(shared.bounds.x)
      for (const point of [...wireLoad.points, ...wireStore.points])
        expect(inside({ ...point, width: 0, height: 0 }, plateRect)).toBe(true)
    })

    it("keeps labels on the plate", () => {
      const labels = [geometry.global.label, geometry.registers.label, geometry.shared.label, geometry.barrier.label]
      for (const label of labels) {
        const widthUnits = label.text.length * 7.3
        expect(label.x).toBeGreaterThanOrEqual(0)
        expect(label.x + widthUnits).toBeLessThanOrEqual(geometry.widthUnits)
        expect(label.y).toBeGreaterThan(24)
      }
    })
  })
}

describe("cta-copy static render", () => {
  it("draws the settled end state in both layouts", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = CtaCopyFigure.renderStatic(layout, { phase: "store", kt: "0" })
      expect(render.readout).toMatch(/^After the barrier/)
      expect(render.drawing).not.toContain("is-warn")
      expect(render.drawing.match(/chart-cell is-on/g)?.length).toBe(3 * TILE * TILE)
    }
  })
})
