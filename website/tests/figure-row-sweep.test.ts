import { describe, expect, it } from "vitest"
import RowSweepFigure from "../src/components/figures/row-sweep/row-sweep.figure"
import { FUNCTIONS, programOf, rowSweepGeometry } from "../src/components/figures/row-sweep/geometry"
import { boundsOf, inside, overlaps, textRect } from "../src/components/inference/figure-geometry"

describe("row-sweep program", () => {
  it("reads each function's statements from the library", () => {
    expect(programOf("softmax").steps.map(s => [s.kind, s.column ?? ""])).toEqual([
      ["fold", "m"],
      ["sweep", ""],
      ["fold", "r"],
      ["sweep", ""],
    ])
    expect(programOf("rms_norm").steps.map(s => s.kind)).toEqual(["fold", "sweep"])
    expect(programOf("rms_norm").steps[1].readsWeight).toBe(true)
    expect(programOf("layer_norm").tile).toBe("x")
    expect(programOf("softmax").steps.some(s => s.readsWeight)).toBe(false)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`row-sweep geometry, ${layout}`, () => {
    const g = rowSweepGeometry(layout)
    const plateRect = { x: 0, y: 0, width: g.widthUnits, height: g.heightUnits }
    const parts = [g.cells.flat(), g.column, g.weight].map(boundsOf)

    it("keeps the tile, the column and the weight row on the plate and apart", () => {
      for (const rect of parts) expect(inside(rect, plateRect)).toBe(true)
      for (const [i, a] of parts.entries()) for (const b of parts.slice(i + 1)) expect(overlaps(a, b)).toBe(false)
    })

    it("fits every statement and label on the plate, off the cells", () => {
      const statements = FUNCTIONS.flatMap(fn => programOf(fn).steps.map(step => step.statement))
      const longest = statements.reduce((a, b) => (a.length >= b.length ? a : b))
      const rects = [g.labels.tile, g.labels.column, g.labels.weight, { ...g.labels.statement, text: longest }].map(textRect)
      for (const rect of rects) {
        expect(inside(rect, plateRect)).toBe(true)
        for (const part of parts) expect(overlaps(rect, part)).toBe(false)
      }
    })
  })
}

describe("row-sweep static render", () => {
  it("ends on the function's last statement", () => {
    for (const fn of FUNCTIONS) {
      const render = RowSweepFigure.renderStatic("wide", { fn })
      expect(render.readout.startsWith(`${fn}: ${programOf(fn).steps.length} statements`)).toBe(true)
      expect(render.drawing).toContain(programOf(fn).steps.at(-1)!.statement.replace(/&/g, "&amp;"))
    }
  })
})
