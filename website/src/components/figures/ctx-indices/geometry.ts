// Model and geometry of the ctx-indices figure, pure so the tests can check them. One CTA of
// @launch(256) on NVIDIA as 8 rows of 32 lanes, and one row W[col, 0:K] of K = 128 elements as 4
// rows of 32 under the lane columns, the elements a lane-strided loop reads.

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { DEFAULT_THREADS, LANES, laneOf, strided, warpOf, warpsOf } from "../ctx-hierarchy/rules"

export const THREADS = DEFAULT_THREADS
export const WARPS = warpsOf(THREADS)
export const K = 128
export const ITERATIONS = K / LANES
export const DEFAULT_THREAD = 113

export interface ThreadIndices {
  thread: number
  warp: number
  lane: number
  // The k values `for k in range(ctx.lane, K, ctx.lanes)` gives this thread.
  reads: number[]
}

export function indicesOf(thread: number): ThreadIndices {
  const lane = laneOf(thread)
  return { thread, warp: warpOf(thread), lane, reads: strided(lane, K, LANES) }
}

export function formulaOf({ thread, warp, lane }: ThreadIndices): string {
  return `${thread} = ${warp} * ${LANES} + ${lane}`
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface TextAt {
  x: number
  y: number
  text: string
  anchor?: "start" | "middle" | "end"
}

export interface IndicesGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  // cells[warp][lane]
  cells: Rect[][]
  // strip[iteration][lane]: element k = iteration * 32 + lane
  strip: Rect[][]
  warpLabels: TextAt[]
  laneTicks: TextAt[]
  laneAxis: TextAt
  stripLabels: TextAt[]
  stripTitle: TextAt
  // The panel's lines, top to bottom: formula, numbers, then the three fields, then the loop note.
  panel: { x: number; formulaY: number; numbersY: number; fieldYs: number[]; loopYs: number[] }
  bounds: { cta: Rect; strip: Rect }
}

interface Spec {
  heightUnits: number
  cellsX: number
  cellUnits: number
  pitchUnits: number
  rowPitchUnits: number
  ctaY: number
  stripY: number
  panelX: number
  panelY: number
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    heightUnits: 316,
    cellsX: 72,
    cellUnits: 11,
    pitchUnits: 12,
    rowPitchUnits: 15,
    ctaY: 82,
    stripY: 238,
    panelX: 476,
    panelY: 82,
  },
  narrow: {
    heightUnits: 492,
    cellsX: 64,
    cellUnits: 9,
    pitchUnits: 10,
    rowPitchUnits: 13,
    ctaY: 76,
    stripY: 226,
    panelX: 18,
    panelY: 316,
  },
}

const LABEL_GAP_UNITS = 8
const TICK_RISE_UNITS = 10
const STRIP_TITLE_RISE_UNITS = 18

export function indicesGeometry(layout: Layout): IndicesGeometry {
  const spec = SPECS[layout]
  const widthUnits = VIEW_WIDTH_UNITS[layout]
  const rect = (x: number, y: number): Rect => ({ x, y, width: spec.cellUnits, height: spec.cellUnits })
  const colX = (lane: number) => spec.cellsX + lane * spec.pitchUnits
  const cells = Array.from({ length: WARPS }, (_, warp) =>
    Array.from({ length: LANES }, (_, lane) => rect(colX(lane), spec.ctaY + warp * spec.rowPitchUnits))
  )
  const strip = Array.from({ length: ITERATIONS }, (_, iteration) =>
    Array.from({ length: LANES }, (_, lane) => rect(colX(lane), spec.stripY + iteration * spec.rowPitchUnits))
  )
  const labelX = spec.cellsX - LABEL_GAP_UNITS
  const mid = (r: Rect) => r.y + r.height / 2
  const isWide = layout === "wide"
  const lineUnits = 18
  const panelTop = spec.panelY + 4
  return {
    widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: "FIG / CTX / INDICES",
    cells,
    strip,
    warpLabels: cells.map((row, warp) => ({ x: labelX, y: mid(row[0]), text: `warp ${warp}`, anchor: "end" })),
    laneTicks: Array.from({ length: LANES / 4 }, (_, index) => {
      const cell = cells[0][index * 4]
      return { x: cell.x + cell.width / 2, y: cell.y - TICK_RISE_UNITS, text: String(index * 4), anchor: "middle" }
    }),
    laneAxis: { x: labelX, y: cells[0][0].y - TICK_RISE_UNITS, text: "lane", anchor: "end" },
    stripLabels: strip.map((row, iteration) => ({
      x: labelX,
      y: mid(row[0]),
      text: `k ${iteration * LANES}`,
      anchor: "end",
    })),
    stripTitle: {
      x: isWide ? spec.cellsX : 18,
      y: spec.stripY - STRIP_TITLE_RISE_UNITS,
      // The narrow plate has room for the loop only.
      text: `${isWide ? `W[col, 0:${K}]  ` : ""}for k in range(ctx.lane, K, ctx.lanes)`,
    },
    panel: {
      x: spec.panelX,
      formulaY: panelTop,
      numbersY: panelTop + lineUnits,
      fieldYs: [0, 1, 2].map(index => panelTop + 2.5 * lineUnits + index * lineUnits),
      loopYs: [0, 1, 2].map(index => panelTop + (isWide ? 7.5 : 6.5) * lineUnits + index * lineUnits),
    },
    bounds: { cta: boundsOf(cells.flat()), strip: boundsOf(strip.flat()) },
  }
}

export function boundsOf(rects: Rect[]): Rect {
  const minX = Math.min(...rects.map(r => r.x))
  const minY = Math.min(...rects.map(r => r.y))
  const maxX = Math.max(...rects.map(r => r.x + r.width))
  const maxY = Math.max(...rects.map(r => r.y + r.height))
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
}
