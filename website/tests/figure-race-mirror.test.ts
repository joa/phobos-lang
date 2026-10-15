import { describe, expect, it } from "vitest"
import { textWidthUnits } from "../src/components/figures/ctx-hierarchy/text"
import {
  mirrorOf,
  raceMirrorGeometry,
  THREADS,
  warpOf,
  type Rect,
  type TextAt,
} from "../src/components/figures/race-mirror/geometry"
import RaceMirrorFigure from "../src/components/figures/race-mirror/race-mirror.figure"
import { LABEL_UNITS } from "../src/lib/figure/layout"

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

// A label's box, at the larger label size, centered on its y.
function box(at: TextAt): Rect {
  const width = textWidthUnits(at.text, LABEL_UNITS)
  const x = at.anchor === "middle" ? at.x - width / 2 : at.anchor === "end" ? at.x - width : at.x
  return { x, y: at.y - LABEL_UNITS / 2, width, height: LABEL_UNITS }
}

describe("race-mirror threads", () => {
  it("loads each element from a thread of the other warp", () => {
    for (let thread = 0; thread < THREADS; thread += 1) {
      expect(mirrorOf(mirrorOf(thread))).toBe(thread)
      expect(warpOf(mirrorOf(thread))).not.toBe(warpOf(thread))
    }
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`race-mirror geometry, ${layout}`, () => {
    const geometry = raceMirrorGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }
    const cells = [...geometry.threads, ...geometry.tile, ...geometry.reads]
    const labels = [
      ...geometry.rowLabels,
      ...geometry.warpLabels,
      geometry.barrierLabel,
      geometry.writerLabel,
      geometry.readerLabel,
    ]

    it("keeps every cell and wire on the plate", () => {
      for (const cell of cells) expect(inside(cell, plateRect)).toBe(true)
      for (const line of [...geometry.writes, ...geometry.loads])
        for (const point of line.points) expect(inside({ ...point, width: 0, height: 0 }, plateRect)).toBe(true)
    })

    it("keeps the cells of a band apart", () => {
      for (const band of [geometry.threads, geometry.tile, geometry.reads])
        for (let index = 1; index < band.length; index += 1) expect(overlaps(band[index - 1], band[index])).toBe(false)
    })

    it("wires each load from its mirror's element to its own register", () => {
      geometry.loads.forEach((line, thread) => {
        const from = geometry.tile[mirrorOf(thread)]
        const to = geometry.reads[thread]
        const center = (rect: Rect) => (layout === "wide" ? rect.x + rect.width / 2 : rect.y + rect.height / 2)
        const along = (point: { x: number; y: number }) => (layout === "wide" ? point.x : point.y)
        expect(along(line.start)).toBeCloseTo(center(from))
        expect(along(line.end)).toBeCloseTo(center(to))
      })
    })

    it("puts the barrier between the tile and the reads", () => {
      for (const { from, to } of geometry.barrier) {
        const cross = layout === "wide" ? from.y : from.x
        const tileEnd = layout === "wide" ? geometry.tile[0].y + geometry.tile[0].height : geometry.tile[0].x + geometry.tile[0].width
        const readsStart = layout === "wide" ? geometry.reads[0].y : geometry.reads[0].x
        expect(cross).toBeGreaterThan(tileEnd)
        expect(cross).toBeLessThan(readsStart)
        expect(layout === "wide" ? to.y : to.x).toBe(cross)
      }
    })

    it("keeps labels on the plate, below its label, and clear of the cells and each other", () => {
      for (const at of labels) {
        const rect = box(at)
        expect(inside(rect, { ...plateRect, y: 30, height: plateRect.height - 30 })).toBe(true)
        for (const cell of cells) expect(overlaps(rect, cell)).toBe(false)
      }
      for (let a = 0; a < labels.length; a += 1)
        for (let b = a + 1; b < labels.length; b += 1) expect(overlaps(box(labels[a]), box(labels[b]))).toBe(false)
    })
  })
}

describe("race-mirror static render", () => {
  const stale = (drawing: string) => drawing.match(/class="chart-box[^"]*is-warn/g)?.length ?? 0
  const loaded = (drawing: string) => drawing.match(/class="chart-box[^"]*is-on/g)?.length ?? 0

  it("ends without a barrier on warp 0's stale loads and the pair the check reports", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = RaceMirrorFigure.renderStatic(layout, { order: "none" })
      expect(render.readout).toMatch(/^Nothing orders thread 63's store before thread 0's load/)
      expect(stale(render.drawing)).toBe(THREADS / 2)
      expect(loaded(render.drawing)).toBe(THREADS / 2)
      expect(render.drawing).toContain("t63 writes")
    }
  })

  it("ends with a barrier on every load seeing a stored value", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = RaceMirrorFigure.renderStatic(layout, { order: "barrier" })
      expect(render.readout).toMatch(/no race/)
      expect(render.drawing).not.toContain("is-warn")
      expect(loaded(render.drawing)).toBe(THREADS)
    }
  })
})
