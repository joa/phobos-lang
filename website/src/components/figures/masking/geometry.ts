// Model and geometry of the masking figure, pure so the tests can check them. Schematic: a tensor
// A[M, N] with M = 20 and N = 28 (multiples of 4, as SPEC assumes) under 8 x 8 tiles, one CTA per
// tile as gemm addresses them (ctx.block.x picks the row of tiles, ctx.block.y the column). The picked
// tile is staged into a shared tile a; elements past the extent are masked. Under @aligned the host
// promises whole tiles, so the extents are drawn as 24 x 32 and nothing is masked.

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { elbow, Polyline } from "../../../lib/figure/polyline"
import { gridExtent, inBounds, isMasked, type Block } from "../ctx-hierarchy/rules"

export const TILE = 8
export const M = 20
export const N = 28
export const GRID_ROWS = gridExtent(M, TILE)
export const GRID_COLS = gridExtent(N, TILE)
export const PADDED_ROWS = GRID_ROWS * TILE
export const PADDED_COLS = GRID_COLS * TILE
export const PROMISE = "@aligned(M = 8, N = 8)"
// Elements per 16-byte vector of f32.
export const VECTOR = 4

export interface Extents {
  m: number
  n: number
}

// Unaligned, the extents are the tensor's own. Aligned, the drawing shows extents that keep the
// promise: whole tiles.
export function extentsOf(aligned: boolean): Extents {
  return aligned ? { m: PADDED_ROWS, n: PADDED_COLS } : { m: M, n: N }
}

export function elementMasked(block: Block, row: number, col: number, aligned: boolean): boolean {
  const { m, n } = extentsOf(aligned)
  return isMasked(block.x * TILE, row, m, aligned) || isMasked(block.y * TILE, col, n, aligned)
}

export interface TileStats {
  rows: number
  cols: number
  inBounds: number
  masked: number
}

export function tileStats(block: Block, aligned: boolean): TileStats {
  const { m, n } = extentsOf(aligned)
  const rows = inBounds(block.x * TILE, TILE, m)
  const cols = inBounds(block.y * TILE, TILE, n)
  return { rows, cols, inBounds: rows * cols, masked: TILE * TILE - rows * cols }
}

export function parseBlock(value: string): Block {
  const [x, y] = value.split(",").map(Number)
  const ok = (v: number, extent: number) => Number.isInteger(v) && v >= 0 && v < extent
  return ok(x, GRID_ROWS) && ok(y, GRID_COLS) ? { x, y } : { x: GRID_ROWS - 1, y: GRID_COLS - 1 }
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

export interface MaskingGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  // PADDED_ROWS x PADDED_COLS: the tensor's cells and the positions past its extent.
  tensorCells: Rect[][]
  tensorBounds: Rect
  tensorLabel: TextAt
  sharedCells: Rect[][]
  sharedBounds: Rect
  sharedLabel: TextAt
  vectors: Rect[][]
  code: TextAt
  promise: TextAt
  notes: { x: number; y: number }[]
  tileHit(block: Block): Rect
  outline(block: Block): Rect
  extentBox(extents: Extents): Rect
  wire(block: Block): Polyline
  lead(block: Block): Polyline
}

interface Spec {
  heightUnits: number
  tensor: { x: number; y: number; cellUnits: number; pitchUnits: number; tileGapUnits: number }
  shared: { x: number; y: number }
  notes: { x: number; y: number; pitchUnits: number }
  code: { x: number; y: number }
}

const SHARED_CELL_UNITS = 16
const SHARED_PITCH_UNITS = 18
const LABEL_RISE_UNITS = 16
const WIRE_STANDOFF_UNITS = 8
const OUTLINE_PAD_UNITS = 2

const SPECS: Record<Layout, Spec> = {
  wide: {
    heightUnits: 320,
    tensor: { x: 36, y: 78, cellUnits: 7, pitchUnits: 8, tileGapUnits: 3 },
    shared: { x: 356, y: 78 },
    notes: { x: 530, y: 86, pitchUnits: 20 },
    code: { x: 356, y: 262 },
  },
  narrow: {
    heightUnits: 544,
    tensor: { x: 36, y: 78, cellUnits: 9, pitchUnits: 10, tileGapUnits: 3 },
    shared: { x: 36, y: 372 },
    notes: { x: 202, y: 380, pitchUnits: 20 },
    code: { x: 24, y: 528 },
  },
}

export function maskingGeometry(layout: Layout): MaskingGeometry {
  const spec = SPECS[layout]
  const t = spec.tensor
  const cellX = (col: number) => t.x + col * t.pitchUnits + Math.floor(col / TILE) * t.tileGapUnits
  const cellY = (row: number) => t.y + row * t.pitchUnits + Math.floor(row / TILE) * t.tileGapUnits
  const tensorCells = grid(PADDED_ROWS, PADDED_COLS, (row, col) => ({
    x: cellX(col),
    y: cellY(row),
    width: t.cellUnits,
    height: t.cellUnits,
  }))
  const tensorBounds = boundsOf(tensorCells.flat())
  const sharedCells = grid(TILE, TILE, (row, col) => ({
    x: spec.shared.x + col * SHARED_PITCH_UNITS,
    y: spec.shared.y + row * SHARED_PITCH_UNITS,
    width: SHARED_CELL_UNITS,
    height: SHARED_CELL_UNITS,
  }))
  const sharedBounds = boundsOf(sharedCells.flat())
  const vectors = grid(TILE, TILE / VECTOR, (row, group) =>
    pad(boundsOf([sharedCells[row][group * VECTOR], sharedCells[row][group * VECTOR + VECTOR - 1]]), 1)
  )

  const tileBounds = (block: Block) =>
    boundsOf([
      tensorCells[block.x * TILE][block.y * TILE],
      tensorCells[block.x * TILE + TILE - 1][block.y * TILE + TILE - 1],
    ])
  const extentBox = ({ m, n }: Extents) =>
    pad(boundsOf([tensorCells[0][0], tensorCells[m - 1][n - 1]]), (t.pitchUnits - t.cellUnits) / 2 + 1)

  const isWide = layout === "wide"
  const sharedMid = { x: sharedBounds.x + sharedBounds.width / 2, y: sharedBounds.y + sharedBounds.height / 2 }
  const exitX = tensorBounds.x + tensorBounds.width + WIRE_STANDOFF_UNITS
  const exitY = tensorBounds.y + tensorBounds.height + WIRE_STANDOFF_UNITS
  const wire = (block: Block): Polyline => {
    const tile = tileBounds(block)
    if (isWide) {
      return elbow(
        { x: exitX, y: tile.y + tile.height / 2 },
        { x: sharedBounds.x - WIRE_STANDOFF_UNITS, y: sharedMid.y }
      ).simplified()
    }
    return elbow(
      { x: tile.x + tile.width / 2, y: exitY },
      { x: sharedMid.x, y: sharedBounds.y - LABEL_RISE_UNITS - 12 },
      "v"
    ).simplified()
  }
  const lead = (block: Block): Polyline => {
    const tile = pad(tileBounds(block), OUTLINE_PAD_UNITS)
    return isWide
      ? new Polyline([
          { x: tile.x + tile.width, y: tile.y + tile.height / 2 },
          { x: exitX, y: tile.y + tile.height / 2 },
        ])
      : new Polyline([
          { x: tile.x + tile.width / 2, y: tile.y + tile.height },
          { x: tile.x + tile.width / 2, y: exitY },
        ])
  }

  return {
    widthUnits: VIEW_WIDTH_UNITS[layout],
    heightUnits: spec.heightUnits,
    plateLabel: isWide ? "FIG / MASKING / TILE AT THE EDGE" : "FIG / MASKING",
    tensorCells,
    tensorBounds,
    tensorLabel: { x: tensorBounds.x, y: tensorBounds.y - LABEL_RISE_UNITS, text: "" },
    sharedCells,
    sharedBounds,
    sharedLabel: { x: sharedBounds.x, y: sharedBounds.y - LABEL_RISE_UNITS, text: "shared  a  8 x 8 f32" },
    vectors,
    code: { ...spec.code, text: "var a = A[pm * 8 :+ 8, pn * 8 :+ 8]" },
    promise: isWide
      ? { x: spec.code.x, y: spec.code.y + 22, text: PROMISE }
      : { x: spec.notes.x, y: spec.notes.y + 5 * spec.notes.pitchUnits + 8, text: PROMISE },
    notes: Array.from({ length: 5 }, (_, index) => ({
      x: spec.notes.x,
      y: spec.notes.y + index * spec.notes.pitchUnits,
    })),
    tileHit: block => pad(tileBounds(block), 1),
    outline: block => pad(tileBounds(block), OUTLINE_PAD_UNITS),
    extentBox,
    wire,
    lead,
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
