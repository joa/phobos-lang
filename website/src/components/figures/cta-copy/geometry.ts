// Geometry of the cta-copy figure, pure so the tests can check it. Everything here is schematic:
// a 24 x 32 f32 tensor A, 8 x 8 tiles, one CTA of 32 threads that loads two adjacent floats each,
// the way gemm_fp32.ph's `var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]` stages a tile.

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { Polyline, type Point } from "../../../lib/figure/polyline"

export const TENSOR_ROWS = 24
export const TENSOR_COLS = 32
export const TILE = 8
export const THREADS = 32
export const PER_THREAD = 2
export const THREADS_PER_ROW = TILE / PER_THREAD
export const BLOCK_ROW = 1
export const K_STEPS = TENSOR_COLS / TILE
export const F32_BYTES = 4

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

export interface CtaCopyGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  global: { cells: Rect[][]; bounds: Rect; label: TextAt }
  registers: { cells: Rect[][]; threads: Rect[]; bounds: Rect; label: TextAt; note: TextAt; threadLabels: TextAt[] }
  shared: { cells: Rect[][]; bounds: Rect; label: TextAt }
  barrier: { from: Point; to: Point; label: TextAt }
  wireLoad: Polyline
  wireStore: Polyline
  tileRect(kStep: number): Rect
  // The short lead inside the tensor, from the tile's edge to where the load wire leaves the grid.
  lead(kStep: number): Polyline
}

// The thread that moves the element at (row, col) of the tile: row-major, PER_THREAD adjacent
// columns per thread.
export function threadOf(row: number, col: number): number {
  return row * THREADS_PER_ROW + Math.floor(col / PER_THREAD)
}

export function elementsOf(thread: number): [number, number][] {
  const row = Math.floor(thread / THREADS_PER_ROW)
  const firstCol = (thread % THREADS_PER_ROW) * PER_THREAD
  return Array.from({ length: PER_THREAD }, (_, index) => [row, firstCol + index] as [number, number])
}

// The slice the CTA copies at a k step, as gemm writes it: A[pm * TILE :+ TILE, kt :+ TILE].
export function sliceOf(kStep: number): { rows: [number, number]; cols: [number, number]; kt: number } {
  const kt = kStep * TILE
  return { rows: [BLOCK_ROW * TILE, BLOCK_ROW * TILE + TILE], cols: [kt, kt + TILE], kt }
}

export const TILE_BYTES = TILE * TILE * F32_BYTES

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  global: { x: number; y: number; cellUnits: number; gapUnits: number; tileGapUnits: number }
  // y 0 centers the block on the tile's row band (wide) or below the tensor (narrow).
  registers: { x: number; y: number; cellUnits: number }
  shared: { x: number; y: number; cellUnits: number }
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: VIEW_WIDTH_UNITS.wide,
    heightUnits: 300,
    plateLabel: "FIG / TILE COPY / GLOBAL TO SHARED",
    global: { x: 36, y: 78, cellUnits: 6, gapUnits: 1, tileGapUnits: 3 },
    registers: { x: 382, y: 0, cellUnits: 11 },
    shared: { x: 584, y: 0, cellUnits: 12 },
  },
  narrow: {
    widthUnits: VIEW_WIDTH_UNITS.narrow,
    heightUnits: 520,
    plateLabel: "FIG / TILE COPY",
    global: { x: 52, y: 78, cellUnits: 8, gapUnits: 1, tileGapUnits: 3 },
    registers: { x: 62, y: 0, cellUnits: 10 },
    shared: { x: 252, y: 0, cellUnits: 11 },
  },
}

const REG_PAIR_GAP_UNITS = 1
const REG_THREAD_GAP_UNITS = 5
const REG_ROW_PITCH_EXTRA_UNITS = 3
const SHARED_GAP_UNITS = 1
const LABEL_RISE_UNITS = 14
const THREAD_LABEL_RISE_UNITS = 8
const BLOCK_LABEL_RISE_UNITS = 26
const NOTE_DROP_UNITS = 18
const WIRE_STANDOFF_UNITS = 8
const TILE_OUTLINE_PAD_UNITS = 2

export function ctaCopyGeometry(layout: Layout): CtaCopyGeometry {
  const spec = SPECS[layout]

  const g = spec.global
  const pitch = g.cellUnits + g.gapUnits
  const globalCellX = (col: number) => g.x + col * pitch + Math.floor(col / TILE) * g.tileGapUnits
  const globalCellY = (row: number) => g.y + row * pitch + Math.floor(row / TILE) * g.tileGapUnits
  const globalCells = grid(TENSOR_ROWS, TENSOR_COLS, (row, col) => ({
    x: globalCellX(col),
    y: globalCellY(row),
    width: g.cellUnits,
    height: g.cellUnits,
  }))
  const globalBounds = boundsOf(globalCells.flat())

  const isWide = layout === "wide"
  const blockHeight = (cellUnits: number, pitchUnits: number) => (TILE - 1) * pitchUnits + cellUnits
  const bandRect = boundsOf([globalCells[BLOCK_ROW * TILE][0], globalCells[BLOCK_ROW * TILE + TILE - 1][0]])
  const blockCenterY = isWide ? midY(bandRect) : globalBounds.y + globalBounds.height + 110
  const r = spec.registers
  const threadWidth = 2 * r.cellUnits + REG_PAIR_GAP_UNITS
  const regRowPitch = r.cellUnits + REG_ROW_PITCH_EXTRA_UNITS
  const regCells = grid(TILE, TILE, (row, col) => {
    const threadInRow = Math.floor(col / PER_THREAD)
    return {
      x:
        r.x +
        threadInRow * (threadWidth + REG_THREAD_GAP_UNITS) +
        (col % PER_THREAD) * (r.cellUnits + REG_PAIR_GAP_UNITS),
      y: blockCenterY - blockHeight(r.cellUnits, regRowPitch) / 2 + row * regRowPitch,
      width: r.cellUnits,
      height: r.cellUnits,
    }
  })
  const threadRects = Array.from({ length: THREADS }, (_, thread) => {
    const [[row, col]] = elementsOf(thread)
    const first = regCells[row][col]
    return { x: first.x - 1.5, y: first.y - 1.5, width: threadWidth + 3, height: r.cellUnits + 3 }
  })
  const regBounds = boundsOf(threadRects)

  const s = spec.shared
  const sharedCells = grid(TILE, TILE, (row, col) => ({
    x: s.x + col * (s.cellUnits + SHARED_GAP_UNITS),
    y:
      blockCenterY -
      blockHeight(s.cellUnits, s.cellUnits + SHARED_GAP_UNITS) / 2 +
      row * (s.cellUnits + SHARED_GAP_UNITS),
    width: s.cellUnits,
    height: s.cellUnits,
  }))
  const sharedBounds = boundsOf(sharedCells.flat())

  const tileRect = (kStep: number): Rect => {
    const { rows, cols } = sliceOf(kStep)
    const tile = boundsOf([globalCells[rows[0]][cols[0]], globalCells[rows[1] - 1][cols[1] - 1]])
    return pad(tile, TILE_OUTLINE_PAD_UNITS)
  }
  const bandMidY = midY(tileRect(0))
  const regMidY = midY(regBounds)

  let wireLoad: Polyline
  let lead: (kStep: number) => Polyline
  if (layout === "wide") {
    const exitX = globalBounds.x + globalBounds.width + WIRE_STANDOFF_UNITS
    const enterX = regBounds.x - WIRE_STANDOFF_UNITS
    const turnX = (exitX + enterX) / 2
    wireLoad = new Polyline([
      { x: exitX, y: bandMidY },
      { x: turnX, y: bandMidY },
      { x: turnX, y: regMidY },
      { x: enterX, y: regMidY },
    ]).simplified()
    lead = kStep => {
      const tile = tileRect(kStep)
      return new Polyline([
        { x: tile.x + tile.width, y: bandMidY },
        { x: exitX, y: bandMidY },
      ])
    }
  } else {
    const exitX = globalBounds.x - WIRE_STANDOFF_UNITS
    const railX = exitX - 14
    wireLoad = new Polyline([
      { x: exitX, y: bandMidY },
      { x: railX, y: bandMidY },
      { x: railX, y: regMidY },
      { x: regBounds.x - WIRE_STANDOFF_UNITS, y: regMidY },
    ])
    lead = kStep => {
      const tile = tileRect(kStep)
      return new Polyline([
        { x: tile.x, y: bandMidY },
        { x: exitX, y: bandMidY },
      ])
    }
  }
  const wireStore = new Polyline([
    { x: regBounds.x + regBounds.width + WIRE_STANDOFF_UNITS, y: regMidY },
    { x: sharedBounds.x - WIRE_STANDOFF_UNITS, y: midY(sharedBounds) },
  ])

  const barrierY = sharedBounds.y + sharedBounds.height + 14
  // Wide: one header row across the plate. Narrow: each block's label sits just above it.
  const blockLabelY = isWide ? globalBounds.y - LABEL_RISE_UNITS : regBounds.y - BLOCK_LABEL_RISE_UNITS
  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    global: {
      cells: globalCells,
      bounds: globalBounds,
      label: {
        x: globalBounds.x,
        y: globalBounds.y - LABEL_RISE_UNITS,
        text: `global  A  ${TENSOR_ROWS} x ${TENSOR_COLS} f32`,
      },
    },
    registers: {
      cells: regCells,
      threads: threadRects,
      bounds: regBounds,
      label: { x: regBounds.x, y: blockLabelY, text: "registers" },
      note: {
        x: regBounds.x,
        y: regBounds.y + regBounds.height + NOTE_DROP_UNITS,
        text: isWide ? "32 threads, 2 x f32" : "32 threads",
      },
      threadLabels: Array.from({ length: THREADS_PER_ROW }, (_, thread) => ({
        x: threadRects[thread].x + threadRects[thread].width / 2,
        y: regBounds.y - THREAD_LABEL_RISE_UNITS,
        text: `t${thread}`,
        anchor: "middle" as const,
      })),
    },
    shared: {
      cells: sharedCells,
      bounds: sharedBounds,
      label: { x: sharedBounds.x, y: blockLabelY, text: `shared  a  ${TILE_BYTES} B` },
    },
    barrier: {
      from: { x: sharedBounds.x, y: barrierY },
      to: { x: sharedBounds.x + sharedBounds.width, y: barrierY },
      label: { x: sharedBounds.x, y: barrierY + NOTE_DROP_UNITS - 2, text: "barrier" },
    },
    wireLoad,
    wireStore,
    tileRect,
    lead,
  }
}

function grid(rows: number, cols: number, at: (row: number, col: number) => Rect): Rect[][] {
  return Array.from({ length: rows }, (_, row) => Array.from({ length: cols }, (_, col) => at(row, col)))
}

export function boundsOf(rects: Rect[]): Rect {
  const minX = Math.min(...rects.map(rect => rect.x))
  const minY = Math.min(...rects.map(rect => rect.y))
  const maxX = Math.max(...rects.map(rect => rect.x + rect.width))
  const maxY = Math.max(...rects.map(rect => rect.y + rect.height))
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
}

function pad(rect: Rect, padUnits: number): Rect {
  return {
    x: rect.x - padUnits,
    y: rect.y - padUnits,
    width: rect.width + 2 * padUnits,
    height: rect.height + 2 * padUnits,
  }
}

function midY(rect: Rect): number {
  return rect.y + rect.height / 2
}
