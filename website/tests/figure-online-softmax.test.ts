import { describe, expect, it } from "vitest"
import OnlineSoftmaxFigure from "../src/components/figures/online-softmax/online-softmax.figure"
import { BLOCKS, onlineSoftmaxGeometry } from "../src/components/figures/online-softmax/geometry"
import { boundsOf, inside, overlaps, textRect } from "../src/components/inference/figure-geometry"
import { libraryFunction } from "../src/lib/stdlib/sources"

describe("online softmax figure", () => {
  it("names steps the library's online_softmax takes", () => {
    const source = libraryFunction("online_softmax").source
    for (const piece of ["rowmax(s)", "tmax(m, mnew)", "exp(s - mnew)", "exp(m - mnew)", "rowsum(s)", "dot(s, v)"])
      expect(source).toContain(piece)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`online softmax geometry, ${layout}`, () => {
    const g = onlineSoftmaxGeometry(layout)
    const plateRect = { x: 0, y: 0, width: g.widthUnits, height: g.heightUnits }
    const groups = [g.blocks, g.scores.flat(), g.m, g.l, g.acc.flat()].map(boundsOf)

    it("keeps every part on the plate and apart", () => {
      for (const rect of groups) expect(inside(rect, plateRect)).toBe(true)
      for (const [i, a] of groups.entries()) for (const b of groups.slice(i + 1)) expect(overlaps(a, b)).toBe(false)
    })

    it("keeps labels on the plate and off the parts", () => {
      const { kv, scores, m, l, acc } = g.labels
      const labels = [kv, scores, m, l, acc].map(textRect)
      for (const rect of labels) {
        expect(inside(rect, plateRect)).toBe(true)
        for (const part of groups) expect(overlaps(rect, part)).toBe(false)
      }
      for (const [i, a] of labels.entries()) for (const b of labels.slice(i + 1)) expect(overlaps(a, b)).toBe(false)
    })

    it("routes each block's lines on the plate, around the scores", () => {
      for (let block = 0; block < BLOCKS; block += 1) {
        for (const line of [g.keyLine(block), g.valueLine(block)]) {
          for (const point of line.points) expect(inside({ ...point, width: 0, height: 0 }, plateRect)).toBe(true)
        }
        for (const point of g.valueLine(block).points)
          expect(inside({ ...point, width: 0, height: 0 }, boundsOf(g.scores.flat()))).toBe(false)
      }
    })
  })
}

describe("online softmax static render", () => {
  it("ends on the normalization with every block taken", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = OnlineSoftmaxFigure.renderStatic(layout, {})
      expect(render.readout).toMatch(/^After the last block/)
      expect(render.drawing.match(/chart-box is-on/g)?.length).toBe(BLOCKS)
    }
  })
})
