// Model and geometry of the tile-copy-sweep figure, pure so the tests can check them. Schematic: a
// 16 x 32 f32 tensor A, the slice A[8 :+ 8, 16 :+ 16] staged into a shared tile a by a CTA drawn as
// 32 threads, each moving one run of 4 adjacent f32 (one 16-byte load). A three-line program steps
// through a's lifetime on a ruler of shared-memory bytes: allocated and filled, live until its last
// use, released, and its bytes taken by the next tile. With let, a is a view and takes no bytes.

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { elbow, type Polyline } from "../../../lib/figure/polyline"

export const TENSOR_ROWS = 16
export const TENSOR_COLS = 32
export const SLICE_ROW = 8
export const SLICE_COL = 16
export const TILE_ROWS = 8
export const TILE_COLS = 16
export const THREADS = 32
export const RUN = 4
export const RUNS_PER_ROW = TILE_COLS / RUN
export const F32_BYTES = 4
export const TILE_BYTES = TILE_ROWS * TILE_COLS * F32_BYTES
export const S_BYTES = TILE_ROWS * F32_BYTES
export const RULER_BYTES = 1024

export type Bind = "var" | "let"
export type Step = "copy" | "read" | "reuse"
export const STEPS: readonly Step[] = ["copy", "read", "reuse"]

export interface Run {
  row: number
  col: number
}

// Thread t moves the run of RUN elements starting at a[row, col].
export function runOf(thread: number): Run {
  return { row: Math.floor(thread / RUNS_PER_ROW), col: (thread % RUNS_PER_ROW) * RUN }
}

export function threadOf(row: number, col: number): number {
  return row * RUNS_PER_ROW + Math.floor(col / RUN)
}

export const PROGRAM: Record<Bind, readonly string[]> = {
  var: [
    `var a = A[${SLICE_ROW} :+ ${TILE_ROWS}, ${SLICE_COL} :+ ${TILE_COLS}]`,
    "var s: tile<f32>[8, 1] = rowsum(a)",
    `var b = B[${SLICE_ROW} :+ ${TILE_ROWS}, 0 :+ ${TILE_COLS}]`,
  ],
  let: [
    `let a = A[${SLICE_ROW} :+ ${TILE_ROWS}, ${SLICE_COL} :+ ${TILE_COLS}]`,
    `Y[${SLICE_ROW} :+ ${TILE_ROWS}, ${SLICE_COL} :+ ${TILE_COLS}] = a * 2.0`,
    `var b = B[${SLICE_ROW} :+ ${TILE_ROWS}, 0 :+ ${TILE_COLS}]`,
  ],
}

export interface Allocation {
  name: "a" | "s" | "b"
  offsetBytes: number
  sizeBytes: number
}

// The tiles alive after each step. Lifetimes that do not overlap share bytes: b takes a's bytes once
// a's last use has passed, while s, still live, keeps its own.
export function liveAfter(bind: Bind, step: Step): Allocation[] {
  const a: Allocation = { name: "a", offsetBytes: 0, sizeBytes: TILE_BYTES }
  const s: Allocation = { name: "s", offsetBytes: TILE_BYTES, sizeBytes: S_BYTES }
  const b: Allocation = { name: "b", offsetBytes: 0, sizeBytes: TILE_BYTES }
  if (bind === "let") return step === "reuse" ? [b] : []
  if (step === "copy") return [a]
  if (step === "read") return [a, s]
  return [b, s]
}

export function footprintBytes(bind: Bind): number {
  return Math.max(
    ...STEPS.map(step =>
      liveAfter(bind, step).reduce((end, alloc) => Math.max(end, alloc.offsetBytes + alloc.sizeBytes), 0)
    )
  )
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
}

export interface TileCopyGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  tensorCells: Rect[][]
  tensorBounds: Rect
  tensorLabel: TextAt
  slice: Rect
  sliceLabel: TextAt
  tileCells: Rect[][]
  tileBounds: Rect
  tileLabel: TextAt
  runHits: Rect[]
  sCells: Rect[]
  sLabel: TextAt
  wire: Polyline
  program: { marker: TextAt; text: TextAt }[]
  ruler: {
    x: number
    width: number
    label: TextAt
    segmentY: number
    segmentHeight: number
    axisY: number
    tickY: number
  }
  rulerX(bytes: number): number
}

interface Spec {
  heightUnits: number
  tensor: { x: number; y: number; cell: number; pitch: number }
  tile: { x: number; y: number; cell: number; pitch: number }
  programY: number
  programX: number
  programPitch: number
  ruler: { x: number; width: number; y: number }
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    heightUnits: 300,
    tensor: { x: 36, y: 78, cell: 6, pitch: 7 },
    tile: { x: 316, y: 78, cell: 12, pitch: 13 },
    programX: 36,
    programY: 232,
    programPitch: 20,
    ruler: { x: 316, width: 320, y: 222 },
  },
  narrow: {
    heightUnits: 540,
    tensor: { x: 56, y: 78, cell: 8, pitch: 9 },
    tile: { x: 56, y: 272, cell: 14, pitch: 15 },
    programX: 24,
    programY: 420,
    programPitch: 20,
    ruler: { x: 40, width: 320, y: 484 },
  },
}

const LABEL_RISE_UNITS = 16
const S_GAP_UNITS = 16
const MARKER_GAP_UNITS = 10

export function tileCopyGeometry(layout: Layout): TileCopyGeometry {
  const spec = SPECS[layout]
  const isWide = layout === "wide"
  const t = spec.tensor
  const tensorCells = grid(TENSOR_ROWS, TENSOR_COLS, (row, col) => ({
    x: t.x + col * t.pitch,
    y: t.y + row * t.pitch,
    width: t.cell,
    height: t.cell,
  }))
  const tensorBounds = boundsOf(tensorCells.flat())
  const slice = pad(
    boundsOf([tensorCells[SLICE_ROW][SLICE_COL], tensorCells[SLICE_ROW + TILE_ROWS - 1][SLICE_COL + TILE_COLS - 1]]),
    1.5
  )
  const p = spec.tile
  const tileCells = grid(TILE_ROWS, TILE_COLS, (row, col) => ({
    x: p.x + col * p.pitch,
    y: p.y + row * p.pitch,
    width: p.cell,
    height: p.cell,
  }))
  const tileBounds = boundsOf(tileCells.flat())
  const runHits = Array.from({ length: THREADS }, (_, thread) => {
    const { row, col } = runOf(thread)
    return pad(boundsOf([tileCells[row][col], tileCells[row][col + RUN - 1]]), 0.5)
  })
  const sX = tileBounds.x + tileBounds.width + S_GAP_UNITS
  const sCells = Array.from({ length: TILE_ROWS }, (_, row) => ({
    x: sX,
    y: p.y + row * p.pitch,
    width: p.cell,
    height: p.cell,
  }))

  const sliceMid = { x: slice.x + slice.width / 2, y: slice.y + slice.height / 2 }
  const tileMidY = tileBounds.y + tileBounds.height / 2
  const wire = isWide
    ? elbow(
        { x: tensorBounds.x + tensorBounds.width + 8, y: sliceMid.y },
        { x: tileBounds.x - 8, y: tileMidY }
      ).simplified()
    : // The slice sits over the tile's right end, so the wire drops straight down past the tile's label.
      elbow({ x: sliceMid.x, y: tensorBounds.y + tensorBounds.height + 8 }, { x: sliceMid.x, y: tileBounds.y - 8 })

  const r = spec.ruler
  return {
    widthUnits: VIEW_WIDTH_UNITS[layout],
    heightUnits: spec.heightUnits,
    plateLabel: isWide ? "FIG / TILES / VAR, LET AND LIFETIME" : "FIG / TILES / LIFETIME",
    tensorCells,
    tensorBounds,
    tensorLabel: { x: tensorBounds.x, y: tensorBounds.y - LABEL_RISE_UNITS },
    slice,
    sliceLabel: isWide
      ? { x: tensorBounds.x, y: tensorBounds.y + tensorBounds.height + 16 }
      : { x: 24, y: tensorBounds.y + tensorBounds.height + 15 },
    tileCells,
    tileBounds,
    tileLabel: { x: tileBounds.x, y: tileBounds.y - LABEL_RISE_UNITS },
    runHits,
    sCells,
    sLabel: { x: sX + p.cell / 2, y: tileBounds.y - LABEL_RISE_UNITS },
    wire,
    program: [0, 1, 2].map(line => {
      const y = spec.programY + line * spec.programPitch
      return { marker: { x: spec.programX, y }, text: { x: spec.programX + MARKER_GAP_UNITS, y } }
    }),
    ruler: {
      x: r.x,
      width: r.width,
      label: { x: r.x, y: r.y },
      segmentY: r.y + 12,
      segmentHeight: 14,
      axisY: r.y + 30,
      tickY: r.y + 44,
    },
    rulerX: bytes => r.x + (bytes / RULER_BYTES) * r.width,
  }
}

function grid<T>(rows: number, cols: number, at: (row: number, col: number) => T): T[][] {
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
