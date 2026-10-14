import { describe, expect, it } from "vitest"
import errors from "../src/data/expected-errors.json"
import CtxCountsFigure from "../src/components/figures/ctx-counts/ctx-counts.figure"
import {
  countsGeometry,
  countsModel,
  ERROR_KEY,
  labeledRows,
  LAUNCH_ERROR,
  MAX_ROWS,
} from "../src/components/figures/ctx-counts/geometry"
import {
  gridExtent,
  inBounds,
  isMasked,
  isWholeWarps,
  laneOf,
  strided,
  threadOf,
  warpOf,
  warpsOf,
  describeThread,
} from "../src/components/figures/ctx-hierarchy/rules"
import { textWidthUnits, wrapText } from "../src/components/figures/ctx-hierarchy/text"

describe("ctx rules", () => {
  it("relates thread, warp and lane as SPEC does", () => {
    for (let thread = 0; thread < 1024; thread += 1) {
      expect(threadOf(warpOf(thread), laneOf(thread))).toBe(thread)
      expect(laneOf(thread)).toBeLessThan(32)
    }
    expect([warpOf(113), laneOf(113)]).toEqual([3, 17])
    expect(threadOf(5, 0)).toBe(160)
  })

  it("accepts only positive multiples of 32 as thread counts", () => {
    expect(isWholeWarps(256)).toBe(true)
    expect(isWholeWarps(48)).toBe(false)
    expect(isWholeWarps(0)).toBe(false)
    expect(warpsOf(256)).toBe(8)
    expect(() => warpsOf(48)).toThrow()
  })

  it("rounds grid extents up and walks strided loops", () => {
    expect(gridExtent(10000, 1024)).toBe(10)
    expect(gridExtent(288, 64)).toBe(5)
    expect(strided(17, 128, 32)).toEqual([17, 49, 81, 113])
  })

  it("masks only past the extent and never under @aligned", () => {
    expect(inBounds(16, 8, 20)).toBe(4)
    expect(inBounds(24, 8, 20)).toBe(0)
    expect(isMasked(16, 3, 20, false)).toBe(false)
    expect(isMasked(16, 4, 20, false)).toBe(true)
    expect(isMasked(16, 4, 20, true)).toBe(false)
  })

  it("names a thread the way the hierarchy page does", () => {
    expect(describeThread({ block: { x: 2, y: 1 }, warp: 3, lane: 17, thread: 113 })).toBe(
      "block (2, 1), warp 3, lane 17, thread 113"
    )
  })

  it("wraps text at spaces", () => {
    expect(wrapText("a bb ccc dddd", 6)).toEqual(["a bb", "ccc", "dddd"])
    expect(textWidthUnits("abcd", 12)).toBeCloseTo(28.8)
  })
})

describe("ctx-counts model", () => {
  it("uses the compiler's error from the docs fence", () => {
    expect(LAUNCH_ERROR).toBe((errors as Record<string, string>)[ERROR_KEY].replaceAll("`", ""))
    expect(LAUNCH_ERROR).not.toContain("`")
    expect(LAUNCH_ERROR).toContain("(got 48)")
  })

  it("groups whole warps and marks the missing half of 48", () => {
    const whole = countsModel(256, false)
    expect([whole.rows, whole.warps, whole.error]).toEqual([8, 8, null])
    const half = countsModel(256, true)
    expect([half.threads, half.rows, half.warps]).toEqual([48, 2, null])
    expect(half.cellState(1, 15)).toBe("on")
    expect(half.cellState(1, 16)).toBe("warn")
  })

  it("labels every fourth warp and an uncrowded last one", () => {
    expect(labeledRows(8)).toEqual([0, 4, 7])
    expect(labeledRows(5)).toEqual([0, 4])
    expect(labeledRows(1)).toEqual([0])
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`ctx-counts geometry, ${layout}`, () => {
    const geometry = countsGeometry(layout)

    it("fits 32 warps and the panel on the plate", () => {
      expect(geometry.areaTop).toBeGreaterThan(40)
      expect(geometry.areaBottom).toBeLessThan(geometry.heightUnits)
      expect(geometry.blockTop(MAX_ROWS)).toBeCloseTo(geometry.areaTop)
      expect(geometry.bracketX + 10 + textWidthUnits("ctx.warps")).toBeLessThan(geometry.widthUnits - 8)
      const last = geometry.errorLines[geometry.errorLines.length - 1]
      expect(last.y).toBeLessThan(geometry.heightUnits - 8)
      for (const line of wrapText(LAUNCH_ERROR, geometry.errorMaxChars))
        expect(geometry.errorLabel.x + textWidthUnits(line, 11)).toBeLessThanOrEqual(geometry.widthUnits - 14)
    })

    it("keeps the panel clear of the warps", () => {
      if (layout === "wide")
        expect(geometry.launch.x).toBeGreaterThan(geometry.bracketX + 10 + textWidthUnits("ctx.warps"))
      else expect(geometry.launch.y).toBeGreaterThan(geometry.areaBottom + 16)
    })
  })
}

describe("ctx-counts static render", () => {
  it("settles on the default count", () => {
    const render = CtxCountsFigure.renderStatic("wide", { threads: 256, launch: "ok" })
    expect(render.readout).toBe(
      "@launch(256): 256 threads in 8 warps of 32 lanes. All three counts are compile-time constants."
    )
    expect(render.drawing).not.toContain("is-warn")
  })

  it("shows the compiler's error for 48 threads", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = CtxCountsFigure.renderStatic(layout, { threads: 256, launch: "48" })
      expect(render.readout).toContain(LAUNCH_ERROR)
      expect(render.drawing.match(/chart-cell is-warn/g)?.length).toBe(16)
    }
  })
})
