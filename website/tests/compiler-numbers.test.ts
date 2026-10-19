import { describe, expect, it } from "vitest"
import { barriers, configCount, placement, searchChoices, searchSpace, supportedChips } from "../src/components/home/compiler-numbers/facts"
import {
  barrierCue,
  chipCue,
  CUE_HEIGHT,
  CUE_WIDTH,
  pipelineCue,
  placementCue,
  placementRects,
  searchCue,
  tensorCoreCue,
} from "../src/components/home/compiler-numbers/cues"

// Every x/cx and y/cy the markup places, with widths and heights added, stays inside the viewBox.
function inside(markup: string) {
  const number = (tag: string, name: string) => Number(tag.match(new RegExp(`\s${name}="([-\d.]+)"`))?.[1] ?? 0)
  for (const [tag] of markup.matchAll(/<(rect|circle|line|text)\b[^>]*>/g)) {
    const x = number(tag, "x") || number(tag, "cx") || number(tag, "x1")
    const y = number(tag, "y") || number(tag, "cy") || number(tag, "y1")
    expect(x, tag).toBeGreaterThanOrEqual(0)
    expect(y, tag).toBeGreaterThanOrEqual(0)
    expect(x + number(tag, "width"), tag).toBeLessThanOrEqual(CUE_WIDTH)
    expect(y + number(tag, "height"), tag).toBeLessThanOrEqual(CUE_HEIGHT)
  }
}

describe("compiler fact cues", () => {
  it("draw inside their viewBox", () => {
    const place = placement()
    for (const cue of [
      placementCue("cn-place", place),
      barrierCue(barriers()),
      pipelineCue(),
      tensorCoreCue(),
      searchCue(configCount(searchSpace(place.trace.source.text))),
      chipCue(supportedChips()),
    ]) {
      inside(cue.markup)
    }
  })

  it("draws one box per tile and one mark per barrier site and configuration", () => {
    const place = placement()
    expect(placementRects(place)).toHaveLength(place.buffers)
    expect(barrierCue(barriers()).markup.match(/cn-barrier/g)).toHaveLength(barriers().sites.length)
    expect(searchCue(64).markup.match(/cn-config/g)).toHaveLength(64)
  })

  it("reads an @autotune list the way the compiler does", () => {
    expect(searchChoices([32, 256])).toEqual([32, 64, 128, 256])
    // The top bound is searched even off the doubling.
    expect(searchChoices([3, 10])).toEqual([3, 6, 10])
    // Two decreasing values, or a pair from zero, are a list.
    expect(searchChoices([256, 128])).toEqual([256, 128])
    expect(searchChoices([0, 4])).toEqual([0, 4])
    expect(searchChoices([1, 2, 4])).toEqual([1, 2, 4])
  })

  it("counts the traced gemm's search, which the gemm page states as 512", () => {
    expect(configCount(searchSpace(placement().trace.source.text))).toBeGreaterThan(0)
    const gemm = searchSpace("@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32], U in [1, 8], T in [256, 128])")
    expect(configCount(gemm)).toBe(512)
  })

  it("keeps a large search's dots apart", () => {
    const radii = [...searchCue(512).markup.matchAll(/cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)"/g)].map(m => m.slice(1).map(Number))
    expect(radii).toHaveLength(512)
    const [[x0, , r], [x1]] = radii
    expect(x1 - x0).toBeGreaterThan(2 * r)
  })

  it("lights each tile while the sweep is inside it", () => {
    const { css } = placementCue("cn-place", placement())
    expect(css.match(/@keyframes cn-place-\d+/g)).toHaveLength(placement().buffers)
  })
})
