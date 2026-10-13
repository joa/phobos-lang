import { describe, expect, it } from "vitest"
import CtxIndicesFigure from "../src/components/figures/ctx-indices/ctx-indices.figure"
import {
  boundsOf,
  formulaOf,
  indicesGeometry,
  indicesOf,
  ITERATIONS,
  K,
  THREADS,
} from "../src/components/figures/ctx-indices/geometry"
import { textWidthUnits } from "../src/components/figures/ctx-hierarchy/text"

function overlaps(a: { x: number; y: number; width: number; height: number }, b: typeof a): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

describe("ctx-indices model", () => {
  it("matches the page's examples", () => {
    expect(formulaOf(indicesOf(113))).toBe("113 = 3 * 32 + 17")
    expect(indicesOf(160)).toMatchObject({ warp: 5, lane: 0 })
    expect(indicesOf(113).reads).toEqual([17, 49, 81, 113])
  })

  it("lets the warp's lanes cover the row once", () => {
    for (let warp = 0; warp < THREADS / 32; warp += 1) {
      const seen = new Set<number>()
      for (let lane = 0; lane < 32; lane += 1) for (const k of indicesOf(warp * 32 + lane).reads) seen.add(k)
      expect(seen.size).toBe(K)
    }
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`ctx-indices geometry, ${layout}`, () => {
    const geometry = indicesGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps the CTA, the strip and the panel apart and on the plate", () => {
      const { cta, strip } = geometry.bounds
      expect(overlaps(cta, strip)).toBe(false)
      expect(strip.y + strip.height).toBeLessThan(geometry.heightUnits - 8)
      expect(cta.x + cta.width).toBeLessThan(geometry.widthUnits - 8)
      expect(geometry.strip).toHaveLength(ITERATIONS)
      const lastPanelY = geometry.panel.loopYs[geometry.panel.loopYs.length - 1]
      expect(lastPanelY).toBeLessThan(geometry.heightUnits - 8)
      if (layout === "wide") expect(geometry.panel.x).toBeGreaterThan(cta.x + cta.width + 16)
      else expect(geometry.panel.formulaY).toBeGreaterThan(strip.y + strip.height + 16)
      expect(boundsOf([cta, strip]).y).toBeGreaterThan(geometry.laneAxis.y)
      expect(plateRect.width).toBeGreaterThan(0)
    })

    it("fits the panel's longest lines", () => {
      const longest = ["thread = warp * lanes + lane", "ctx.lane = 255 - 7 * 32 = 31"]
      expect(geometry.panel.x + textWidthUnits(longest[0])).toBeLessThanOrEqual(geometry.widthUnits - 8)
      expect(geometry.panel.x + textWidthUnits(longest[1], 11)).toBeLessThanOrEqual(geometry.widthUnits - 8)
      const title = geometry.stripTitle
      const fontUnits = layout === "wide" ? 12 : 11
      expect(title.x + textWidthUnits(title.text, fontUnits)).toBeLessThanOrEqual(geometry.widthUnits - 8)
    })

    it("keeps the warp labels left of the cells and on the plate", () => {
      for (const at of geometry.warpLabels) expect(at.x - textWidthUnits(at.text, 11)).toBeGreaterThan(4)
    })
  })
}

describe("ctx-indices static render", () => {
  it("names thread 113 in both modes", () => {
    const indices = CtxIndicesFigure.renderStatic("wide", { thread: 113, mode: "indices" })
    expect(indices.readout).toBe("Thread 113 is lane 17 of warp 3: 113 = 3 * 32 + 17.")
    const loop = CtxIndicesFigure.renderStatic("narrow", { thread: 113, mode: "loop" })
    expect(loop.readout).toMatch(/^Lane 17 of warp 3 reads k = 17, 49, 81, 113\./)
    expect(loop.drawing.match(/is-hot sp-reg/g)?.length).toBe(1 + ITERATIONS)
    expect(indices.drawing.match(/aria-pressed="true"/g)?.length).toBe(1)
  })
})
