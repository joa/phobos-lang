// Layout of the elastic-cache figure, pure so the tests can check it: the card's memory as a bar, the
// rest of the model and its context, the expert cache as columns of slots (one column per expert a
// block), the other process that takes part of the card, the lanes misses take to the CPU, the wall
// times, and decode rates for both engines before, during and after the squeeze.
//
// The bar is drawn to scale from the run: the card's size, the memory in use before the squeeze, what
// one more expert a block costs over all blocks, and the 2 GiB the other process holds.

import type { Layout } from "../../../lib/figure/layout"
import { Polyline } from "../../../lib/figure/polyline"
import type { ElasticCacheData } from "../../../lib/bench/agent"
import type { Rect, TextAt } from "../../inference/figure-geometry"

export const LABEL_UNITS = 13.5
export const SMALL_UNITS = 12.5
const CELL_ROWS = 3

export interface SeriesGeometry {
  engine: string
  line: Polyline
  values: TextAt[]
}

export interface ElasticCacheGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  vramLabel: TextAt
  bar: Rect
  rest: Rect
  restLabel: TextAt
  // columns[column][row]: column n is the (n + 1)th expert a block.
  columns: Rect[][]
  columnPitchUnits: number
  cacheLabel: TextAt
  other: Rect
  otherLabel: TextAt
  lanes: Polyline[]
  cpu: { box: Rect; label: TextAt; note: TextAt }
  wall: { title: TextAt; rows: TextAt[] }
  spark: {
    title: TextAt
    legend: { swatch: Polyline; label: TextAt }[]
    band: Rect
    phases: TextAt[]
    baseline: Polyline
    series: SeriesGeometry[]
  }
}

export function formatRate(rate: number): string {
  return rate.toFixed(1)
}

export function formatGiB(gib: number): string {
  return `${Number.isInteger(gib) ? gib : gib.toFixed(1)} GiB`
}

export function cacheText(perBlock: number): string {
  return `expert cache, ${perBlock} a block`
}

export function cacheEndText(data: ElasticCacheData): string {
  const sizes = [data.cache.before, ...data.cache.shrinks, data.cache.after]
  return `cache ${sizes.join(", ")} a block`
}

export function wallText(engine: { label: string; wallSecs: number }): string {
  return `${engine.label} ${engine.wallSecs.toFixed(1)} s`
}

export function columnCount(data: ElasticCacheData): number {
  return Math.max(data.cache.before, data.cache.after, ...data.cache.shrinks)
}

interface Frame {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  bar: Rect
  labelY: number
  otherLabelY: number
  laneGapUnits: number
  cpu: Rect
  wallTitle: TextAt
  wallRow: (index: number, text: string) => TextAt
  spark: { x: number; width: number; titleY: number; legendY: number; topY: number; baselineY: number; phaseY: number }
}

function frameFor(layout: Layout): Frame {
  if (layout === "wide") {
    return {
      widthUnits: 720,
      heightUnits: 306,
      plateLabel: "FIG / MOE / ELASTIC CACHE",
      bar: { x: 30, y: 58, width: 660, height: 38 },
      labelY: 44,
      otherLabelY: 114,
      laneGapUnits: 8,
      cpu: { x: 240, y: 196, width: 150, height: 48 },
      wallTitle: { x: 30, y: 150, text: "wall time, one replay", sizeUnits: SMALL_UNITS },
      wallRow: (index, text) => ({ x: 30, y: 172 + index * 20, text, sizeUnits: SMALL_UNITS }),
      spark: { x: 420, width: 270, titleY: 150, legendY: 170, topY: 194, baselineY: 270, phaseY: 290 },
    }
  }
  return {
    widthUnits: 400,
    heightUnits: 408,
    plateLabel: "FIG / ELASTIC CACHE",
    bar: { x: 20, y: 58, width: 360, height: 34 },
    labelY: 44,
    otherLabelY: 110,
    laneGapUnits: 7,
    cpu: { x: 106, y: 150, width: 150, height: 46 },
    wallTitle: { x: 20, y: 222, text: "wall", sizeUnits: SMALL_UNITS },
    wallRow: (index, text) => ({ x: [74, 196][index] ?? 74, y: 222, text, sizeUnits: SMALL_UNITS }),
    spark: { x: 20, width: 360, titleY: 252, legendY: 272, topY: 296, baselineY: 372, phaseY: 392 },
  }
}

export function elasticCacheGeometry(layout: Layout, data: ElasticCacheData): ElasticCacheGeometry {
  const frame = frameFor(layout)
  const { bar } = frame
  const unitsPerGiB = bar.width / data.cardGiB
  const columnGiB = (data.cache.blocks * data.cache.slotMiB) / 1024
  const columnPitchUnits = columnGiB * unitsPerGiB
  const restGiB = data.usedBeforeGiB - data.cache.before * columnGiB
  const rest = { x: bar.x, y: bar.y, width: restGiB * unitsPerGiB, height: bar.height }
  const cacheX = rest.x + rest.width + 2
  const rowHeight = (bar.height - 6 - (CELL_ROWS - 1) * 1.5) / CELL_ROWS
  const columns = Array.from({ length: columnCount(data) }, (_, column) =>
    Array.from({ length: CELL_ROWS }, (_, row) => ({
      x: cacheX + column * columnPitchUnits,
      y: bar.y + 3 + row * (rowHeight + 1.5),
      width: Math.max(1, columnPitchUnits - 1.2),
      height: rowHeight,
    }))
  )
  const otherWidth = (data.squeezeMiB / 1024) * unitsPerGiB
  const other = { x: bar.x + bar.width - otherWidth, y: bar.y, width: otherWidth, height: bar.height }

  const cpu = frame.cpu
  const lanes = [0, 1, 2].map(index => {
    const x = cacheX + 6 + index * frame.laneGapUnits
    return new Polyline([
      { x, y: bar.y + bar.height },
      { x, y: cpu.y },
    ])
  })

  const { spark } = frame
  const phaseWidth = spark.width / 3
  const scaleMax = Math.ceil(Math.max(...data.engines.flatMap(engine => engine.decode)) / 10) * 10
  const yOf = (rate: number) => spark.baselineY - (rate / scaleMax) * (spark.baselineY - spark.topY)
  const series = data.engines.map((engine, index) => {
    const ys = engine.decode.map(yOf)
    const points = ys.flatMap((y, phase) => [
      { x: spark.x + phase * phaseWidth, y },
      { x: spark.x + (phase + 1) * phaseWidth, y },
    ])
    // The faster engine's values sit above its line, the other's below, unless that runs into the
    // baseline.
    const values = engine.decode.map((rate, phase) => {
      const y = ys[phase]
      const below = index > 0 && y + 12 + SMALL_UNITS < spark.baselineY
      return {
        x: spark.x + phase * phaseWidth + 4,
        y: below ? y + 12 : y - 11,
        text: formatRate(rate),
        sizeUnits: SMALL_UNITS,
      }
    })
    return { engine: engine.engine, line: new Polyline(points), values }
  })

  let legendX = spark.x
  const legend = data.engines.map(engine => {
    const swatch = new Polyline([
      { x: legendX, y: spark.legendY },
      { x: legendX + 14, y: spark.legendY },
    ])
    const label = { x: legendX + 20, y: spark.legendY, text: engine.label, sizeUnits: SMALL_UNITS }
    legendX += 20 + engine.label.length * 0.6 * SMALL_UNITS + 16
    return { swatch, label }
  })

  return {
    widthUnits: frame.widthUnits,
    heightUnits: frame.heightUnits,
    plateLabel: frame.plateLabel,
    vramLabel: { x: bar.x, y: frame.labelY, text: `card memory, ${formatGiB(data.cardGiB)}`, sizeUnits: SMALL_UNITS },
    bar,
    rest,
    restLabel: {
      x: rest.x + rest.width / 2,
      y: rest.y + rest.height / 2,
      text: "model, context",
      anchor: "middle",
      sizeUnits: SMALL_UNITS,
    },
    columns,
    columnPitchUnits,
    cacheLabel: { x: cacheX, y: frame.labelY, text: cacheEndText(data), sizeUnits: SMALL_UNITS },
    other,
    otherLabel: {
      x: other.x + other.width,
      y: frame.otherLabelY,
      text: `another process, ${formatGiB(data.squeezeMiB / 1024)}`,
      anchor: "end",
      sizeUnits: SMALL_UNITS,
    },
    lanes,
    cpu: {
      box: cpu,
      label: { x: cpu.x + cpu.width / 2, y: cpu.y + 17, text: "CPU threads", anchor: "middle", sizeUnits: LABEL_UNITS },
      note: {
        x: cpu.x + cpu.width / 2,
        y: cpu.y + cpu.height - 13,
        text: "compute misses",
        anchor: "middle",
        sizeUnits: SMALL_UNITS,
      },
    },
    wall: { title: frame.wallTitle, rows: data.engines.map((engine, index) => frame.wallRow(index, wallText(engine))) },
    spark: {
      title: { x: spark.x, y: spark.titleY, text: "decode, tokens/s", sizeUnits: SMALL_UNITS },
      legend,
      band: { x: spark.x + phaseWidth, y: spark.topY - 8, width: phaseWidth, height: spark.baselineY - spark.topY + 8 },
      phases: ["before", `${formatGiB(data.squeezeMiB / 1024)} taken`, "after"].map((text, phase) => ({
        x: spark.x + (phase + 0.5) * phaseWidth,
        y: spark.phaseY,
        text,
        anchor: "middle" as const,
        sizeUnits: SMALL_UNITS,
      })),
      baseline: new Polyline([
        { x: spark.x, y: spark.baselineY },
        { x: spark.x + spark.width, y: spark.baselineY },
      ]),
      series,
    },
  }
}
