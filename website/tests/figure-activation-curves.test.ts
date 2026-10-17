import { describe, expect, it } from "vitest"
import ActivationCurvesFigure from "../src/components/figures/activation-curves/activation-curves.figure"
import { activationGeometry, FAMILIES } from "../src/components/figures/activation-curves/geometry"
import { inside, overlaps, textRect } from "../src/components/inference/figure-geometry"
import { FUNCTIONS } from "../src/lib/stdlib/sources"

describe("activation families", () => {
  it("plot functions the library defines, each once", () => {
    const names = FAMILIES.flatMap(family => family.members.map(member => member.name))
    expect(new Set(names).size).toBe(names.length)
    for (const name of names) expect(FUNCTIONS.some(fn => fn.name === name && fn.kind === "expression"), name).toBe(true)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`activation geometry, ${layout}`, () => {
    const geometry = activationGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps every curve inside the plot", () => {
      for (const family of geometry.families) {
        for (const curve of family.curves) {
          expect(curve.segments.length, curve.call).toBeGreaterThan(0)
          for (const point of curve.segments.flat())
            expect(inside({ ...point, width: 0, height: 0 }, geometry.plot), curve.call).toBe(true)
        }
        expect(family.zeroY).toBeGreaterThanOrEqual(geometry.plot.y)
        expect(family.zeroY).toBeLessThanOrEqual(geometry.plot.y + geometry.plot.height)
      }
    })

    it("keeps labels on the plate, off the plot and apart", () => {
      for (const family of geometry.families) {
        const legends = family.curves.map(curve => textRect(curve.legend))
        const ticks = [...family.yTicks.map(t => t.label), ...geometry.xTicks.map(t => t.label)].map(textRect)
        for (const rect of [...legends, ...ticks]) expect(inside(rect, plateRect)).toBe(true)
        for (const rect of legends) expect(overlaps(rect, geometry.plot)).toBe(false)
        for (const [i, a] of legends.entries())
          for (const b of legends.slice(i + 1)) expect(overlaps(a, b), family.key).toBe(false)
      }
    })
  })
}

describe("activation static render", () => {
  it("rests on every curve of the family with its definitions read from the source", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = ActivationCurvesFigure.renderStatic(layout, { family: "gelu" })
      expect(render.readout).toMatch(/^GELU and its kin: gelu\(x\), gelu_tanh\(x\)/)
      expect(render.drawing.match(/chart-wire-group is-on/g)?.length).toBe(6)
    }
  })
})
