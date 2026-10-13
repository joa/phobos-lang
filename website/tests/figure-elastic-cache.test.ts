import { describe, expect, it } from "vitest"
import ElasticCacheFigure, {
  emissions,
  interpolate,
  perBlockAt,
  phaseAt,
  readoutFor,
  RUN,
  schedule,
} from "../src/components/figures/elastic-cache/elastic-cache.figure"
import { cacheText, columnCount, elasticCacheGeometry } from "../src/components/figures/elastic-cache/geometry"
import { boundsOf, inside, overlaps, textRect, type Rect, type TextAt } from "../src/components/inference/figure-geometry"

describe("elastic-cache run", () => {
  const plan = schedule()

  it("walks the cache through the sizes the run logged", () => {
    expect(perBlockAt(0)).toBe(RUN.cache.before)
    for (const [index, shrink] of plan.shrinks.entries()) {
      expect(shrink.to).toBe(RUN.cache.shrinks[index])
      // Each shrink has settled before the next starts.
      expect(perBlockAt(shrink.startMs + 600)).toBeCloseTo(shrink.to)
    }
    expect(perBlockAt(plan.totalMs)).toBe(RUN.cache.after)
  })

  it("lands every miss before the run ends and sends more while the cache is small", () => {
    const flights = emissions(plan)
    expect(flights.length).toBeGreaterThan(0)
    const during = flights.filter(flight => phaseAt(flight.atMs, plan) === "during").length
    const before = flights.filter(flight => phaseAt(flight.atMs, plan) === "before").length
    expect(during).toBeGreaterThan(before * 3)
    for (const flight of flights) expect(flight.atMs + 900).toBeLessThanOrEqual(plan.totalMs)
  })

  it("interpolates piecewise and clamps", () => {
    expect(interpolate(-1, [0, 10, 20], [0, 5, 30])).toBe(0)
    expect(interpolate(5, [0, 10, 20], [0, 5, 30])).toBe(2.5)
    expect(interpolate(15, [0, 10, 20], [0, 5, 30])).toBe(17.5)
    expect(interpolate(99, [0, 10, 20], [0, 5, 30])).toBe(30)
  })

  it("writes ASCII readouts with the measured rates", () => {
    for (const phase of ["before", "during", "after"] as const) expect(readoutFor(phase)).toMatch(/^[\x20-\x7e]+$/)
    expect(readoutFor("during")).toContain(RUN.engines[1].decode[1].toFixed(1))
  })
})

function textRects(geometry: ReturnType<typeof elasticCacheGeometry>, cacheLabel: TextAt): Rect[] {
  const spark = geometry.spark
  const labels: TextAt[] = [
    geometry.vramLabel,
    cacheLabel,
    geometry.otherLabel,
    geometry.cpu.label,
    geometry.cpu.note,
    geometry.wall.title,
    ...geometry.wall.rows,
    spark.title,
    ...spark.legend.map(entry => entry.label),
    ...spark.phases,
    ...spark.series.flatMap(series => series.values),
  ]
  return labels.map(textRect)
}

for (const layout of ["wide", "narrow"] as const) {
  describe(`elastic-cache geometry, ${layout}`, () => {
    const geometry = elasticCacheGeometry(layout, RUN)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }
    const plateLabel = { x: 0, y: 8, width: geometry.widthUnits, height: 24 }
    const cacheAt = (perBlock: number) => boundsOf(geometry.columns.slice(0, perBlock).flat())
    const sparkBounds = boundsOf([geometry.spark.band, ...geometry.spark.series.map(series => boundsOf(series.line.points.map(p => ({ ...p, width: 0, height: 0 }))))])

    it("stays within the size budget", () => {
      expect(geometry.heightUnits).toBeLessThanOrEqual(layout === "wide" ? 340 : 540)
    })

    it("draws the card to scale and the cache inside it", () => {
      expect(geometry.columns).toHaveLength(columnCount(RUN))
      for (const cell of geometry.columns.flat()) expect(inside(cell, geometry.bar)).toBe(true)
      expect(inside(geometry.rest, geometry.bar)).toBe(true)
      expect(inside(geometry.other, geometry.bar)).toBe(true)
      expect(overlaps(geometry.rest, cacheAt(1))).toBe(false)
      // The smallest cache leaves the other process its room.
      expect(overlaps(cacheAt(Math.min(...RUN.cache.shrinks)), geometry.other)).toBe(false)
      expect(geometry.other.width / geometry.bar.width).toBeCloseTo(RUN.squeezeMiB / 1024 / RUN.cardGiB)
    })

    it("keeps every label on the plate and apart", () => {
      for (const cacheLabel of [geometry.cacheLabel, { ...geometry.cacheLabel, text: cacheText(RUN.cache.before) }]) {
        const rects = textRects(geometry, cacheLabel)
        for (const rect of rects) {
          expect(inside(rect, plateRect), JSON.stringify(rect)).toBe(true)
          expect(overlaps(rect, plateLabel), JSON.stringify(rect)).toBe(false)
          expect(overlaps(rect, geometry.bar), JSON.stringify(rect)).toBe(false)
          expect(overlaps(rect, geometry.cpu.box) && rect !== rects[3] && rect !== rects[4]).toBe(false)
        }
        for (let a = 0; a < rects.length; a += 1) {
          for (let b = a + 1; b < rects.length; b += 1)
            expect(overlaps(rects[a], rects[b]), `${JSON.stringify(rects[a])} ${JSON.stringify(rects[b])}`).toBe(false)
        }
      }
      expect(inside(textRect(geometry.restLabel), geometry.rest)).toBe(true)
      expect(inside(textRect(geometry.cpu.label), geometry.cpu.box)).toBe(true)
      expect(inside(textRect(geometry.cpu.note), geometry.cpu.box)).toBe(true)
    })

    it("keeps the parts apart", () => {
      expect(inside(geometry.cpu.box, plateRect)).toBe(true)
      expect(inside(sparkBounds, plateRect)).toBe(true)
      expect(overlaps(geometry.cpu.box, sparkBounds)).toBe(false)
      for (const lane of geometry.lanes) {
        const [top, bottom] = lane.points
        expect(top.x).toBeGreaterThan(geometry.columns[0][0].x)
        expect(top.x).toBeLessThan(cacheAt(Math.min(...RUN.cache.shrinks)).x + cacheAt(Math.min(...RUN.cache.shrinks)).width)
        expect(bottom.y).toBe(geometry.cpu.box.y)
        // A packet's cloud and glow stay clear of the labels beside its lane.
        const swath = { x: top.x - 12, y: top.y, width: 24, height: bottom.y - top.y }
        for (const rect of textRects(geometry, geometry.cacheLabel)) expect(overlaps(swath, rect), JSON.stringify(rect)).toBe(false)
      }
    })

    it("renders statically on the end state", () => {
      const render = ElasticCacheFigure.renderStatic(layout, {})
      expect(render.readout).toBe(readoutFor("after"))
      expect(render.drawing).not.toMatch(/NaN|undefined/)
      expect(render.drawing).toContain(`cache ${RUN.cache.before}, ${RUN.cache.shrinks.join(", ")}, ${RUN.cache.after} a block`)
      expect(render.drawing).toContain('stroke-dashoffset="0"')
    })
  })
}
