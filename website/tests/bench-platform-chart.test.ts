import { describe, expect, it } from "vitest"
import { chartGeometry, chartSeries, spread, testOrder } from "../src/lib/bench/platformChart"
import { loadPlatforms, testOf } from "../src/lib/bench/platforms"

const platforms = loadPlatforms()
const series = chartSeries(platforms)
const tests = testOrder(series)

describe("platform chart", () => {
  it("draws every card with a bench.py run, in chip order, each keeping its slot", () => {
    const measured = platforms.filter(platform => platform.model)
    expect(series.map(line => line.card)).toEqual(measured.map(platform => platform.card))
    for (const line of series) {
      expect(line.slot).toBe(platforms.findIndex(platform => platform.card === line.card) + 1)
    }
  })

  it("orders prompt sizes before decode sizes, each by size", () => {
    const prompts = tests.filter(test => test.startsWith("pp"))
    expect(tests.slice(0, prompts.length)).toEqual(prompts)
    const sizes = (kind: string) => tests.filter(test => test.startsWith(kind)).map(test => Number(test.slice(2)))
    expect(sizes("pp")).toEqual([...sizes("pp")].sort((a, b) => a - b))
    expect(sizes("tg")).toEqual([...sizes("tg")].sort((a, b) => a - b))
  })

  it("plots the ratio of the two means the tables show", () => {
    for (const line of series) {
      const platform = platforms.find(entry => entry.card === line.card)!
      for (const point of line.points) {
        const comparison = testOf(platform, point.test)!
        expect(point.ratio).toBeCloseTo(comparison.phobos!.mean / comparison.other!.mean, 10)
      }
    }
  })

  it("never joins a prompt size to a decode size", () => {
    for (const line of series) {
      for (const segment of line.segments) {
        expect(new Set(segment.map(point => point.test.slice(0, 2))).size).toBe(1)
      }
    }
  })

  it("keeps every point inside the plot and 1.00x on the axis", () => {
    const geometry = chartGeometry(series, tests)
    const { plot } = geometry
    for (const line of series) {
      for (const point of line.points) {
        expect(geometry.x(point.test)).toBeGreaterThan(plot.left)
        expect(geometry.x(point.test)).toBeLessThan(plot.right)
        expect(geometry.y(point.ratio)).toBeGreaterThanOrEqual(plot.top)
        expect(geometry.y(point.ratio)).toBeLessThanOrEqual(plot.bottom)
      }
    }
    expect(geometry.yTicks).toContain(1)
  })

  it("spreads labels that would overlap", () => {
    const labels = spread([{ y: 50 }, { y: 52 }, { y: 51 }, { y: 200 }], 0, 300)
    const ys = labels.map(label => label.y)
    for (let index = 1; index < ys.length; index++) expect(ys[index] - ys[index - 1]).toBeGreaterThanOrEqual(14)
    expect(ys[ys.length - 1]).toBe(200)
  })
})
