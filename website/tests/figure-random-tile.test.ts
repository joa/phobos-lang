import { describe, expect, it } from "vitest"
import RandomTileFigure from "../src/components/figures/random-tile/random-tile.figure"
import { callOf, draw, FUNCTIONS, randomTileGeometry, SEEDS } from "../src/components/figures/random-tile/geometry"
import { boundsOf, inside, overlaps, textRect } from "../src/components/inference/figure-geometry"

describe("random-tile values", () => {
  it("draws the same tile for a seed and another for the next", () => {
    for (const fn of FUNCTIONS) {
      expect(draw(fn, 1, 256).values).toEqual(draw(fn, 1, 256).values)
      expect(draw(fn, 1, 256).values).not.toEqual(draw(fn, 2, 256).values)
    }
  })

  it("shades every cell within [0, 1]", () => {
    for (const fn of FUNCTIONS)
      for (const seed of SEEDS) for (const shade of draw(fn, seed, 256).shades) expect(shade >= 0 && shade <= 1).toBe(true)
  })

  it("keeps dropout's survivors near 1 - p and scaled", () => {
    const { values } = draw("dropout", 1, 4096)
    const kept = values.filter(v => v !== 0)
    expect(Math.abs(kept.length / 4096 - 0.75)).toBeLessThan(0.03)
    for (const v of kept) expect(v).toBeCloseTo(1 / 0.75, 12)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`random-tile geometry, ${layout}`, () => {
    const g = randomTileGeometry(layout)
    const plateRect = { x: 0, y: 0, width: g.widthUnits, height: g.heightUnits }
    const tile = boundsOf(g.cells.flat())

    it("keeps the tile and every call on the plate, apart", () => {
      expect(inside(tile, plateRect)).toBe(true)
      const calls = FUNCTIONS.map(fn => ({ ...g.labels.call, text: callOf(fn) }))
      const texts = [g.labels.tile, g.labels.counter, ...calls].map(textRect)
      for (const rect of texts) {
        expect(inside(rect, plateRect)).toBe(true)
        expect(overlaps(rect, tile)).toBe(false)
      }
    })
  })
}

describe("random-tile static render", () => {
  it("rests on the whole tile with its statistics", () => {
    for (const fn of FUNCTIONS) {
      const render = RandomTileFigure.renderStatic("wide", { fn, seed: "1" })
      expect(render.readout.startsWith(`${fn}`)).toBe(true)
      const lit = draw(fn, 1, 256).shades.filter(shade => shade > 0).length
      expect(render.drawing.match(/chart-cell is-hot/g)?.length).toBe(lit)
      expect(render.drawing).toContain(callOf(fn))
    }
  })
})
