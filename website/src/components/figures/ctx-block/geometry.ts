// Model and geometry of the ctx-block figure, pure so the tests can check them. An output tensor
// C[M, N] cut into 64 x 64 tiles, one per CTA, as gemm addresses it: ctx.block.x picks a row of
// tiles (along M) and ctx.block.y a column (along N). The grid is M and N over the tile, rounded
// up, so the last row and column of tiles can overhang C; that overhang is masked. Schematic.

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { gridExtent, inBounds, type Block } from "../ctx-hierarchy/rules"

export const TILE = 64
export const MIN_EXTENT = 64
export const MAX_EXTENT = 512
export const EXTENT_STEP = 32
export const BATCH = 3
export const MAX_BLOCKS = MAX_EXTENT / TILE
export const DEFAULT_M = 288
export const DEFAULT_N = 416
export const DEFAULT_PICK = "1,2"

export interface BlockModel {
  m: number
  n: number
  blocksX: number
  blocksY: number
  blocksZ: number
  batched: boolean
  pick: Block
  rows: [number, number]
  cols: [number, number]
  rowsInside: number
  colsInside: number
}

export function parsePick(value: string): Block {
  const [x, y] = value.split(",").map(Number)
  return { x: Number.isFinite(x) ? x : 0, y: Number.isFinite(y) ? y : 0 }
}

export function pickValue(block: Block): string {
  return `${block.x},${block.y}`
}

export function blockModel(m: number, n: number, pick: Block, batched: boolean): BlockModel {
  const blocksX = gridExtent(m, TILE)
  const blocksY = gridExtent(n, TILE)
  const clamped = { x: Math.min(Math.max(0, pick.x), blocksX - 1), y: Math.min(Math.max(0, pick.y), blocksY - 1) }
  const rowStart = clamped.x * TILE
  const colStart = clamped.y * TILE
  return {
    m,
    n,
    blocksX,
    blocksY,
    blocksZ: batched ? BATCH : 1,
    batched,
    pick: clamped,
    rows: [rowStart, rowStart + TILE],
    cols: [colStart, colStart + TILE],
    rowsInside: inBounds(rowStart, TILE, m),
    colsInside: inBounds(colStart, TILE, n),
  }
}

export function isWholeTile(model: BlockModel): boolean {
  return model.rowsInside === TILE && model.colsInside === TILE
}

export function sliceText(model: BlockModel): string {
  return `C[${model.rows[0]}:${model.rows[1]}, ${model.cols[0]}:${model.cols[1]}]`
}

export function readoutFor(model: BlockModel): string {
  const { m, n, blocksX, blocksY, pick } = model
  const grid = model.batched
    ? `a grid of ${blocksX} x ${blocksY} x ${BATCH} CTAs, one ${blocksX} x ${blocksY} grid per batch along z`
    : `a grid of ${blocksX} x ${blocksY} CTAs`
  const cta = model.batched ? `CTA (${pick.x}, ${pick.y}, 0)` : `CTA (${pick.x}, ${pick.y})`
  const owns = model.batched ? `${sliceText(model)} of batch 0` : sliceText(model)
  const fit = isWholeTile(model)
    ? "a whole tile"
    : `of which only ${model.rowsInside} x ${model.colsInside} lies inside C; the rest is masked`
  return `C[${m}, ${n}] in ${TILE} x ${TILE} tiles needs ${grid}. ${cta} owns ${owns}, ${fit}.`
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
  text?: string
  anchor?: "start" | "middle" | "end"
}

export interface BlockGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  originX: number
  originY: number
  tileUnits: number
  unitsPerElement: number
  ghostStepUnits: number
  tileRect(bx: number, by: number): Rect
  rowIndex(bx: number): TextAt
  colIndex(by: number): TextAt
  axisRows: TextAt & { rotate: number }
  axisCols: TextAt
  panel: TextAt[]
  // Panel lines that stay empty, to separate the grid's lines from the picked CTA's.
  panelGapAfter: number
}

export const PANEL_LINES = 10

interface Spec {
  heightUnits: number
  originX: number
  panelX: number
  panelY: number
  panelPitch: number
}

const SPECS: Record<Layout, Spec> = {
  wide: { heightUnits: 336, originX: 120, panelX: 400, panelY: 84, panelPitch: 21 },
  narrow: { heightUnits: 540, originX: 72, panelX: 24, panelY: 344, panelPitch: 17 },
}

const TILE_UNITS = 28
const ORIGIN_Y = 96
const GHOST_STEP_UNITS = 8
const PANEL_GAP_UNITS = 12
const GAP_AFTER = 4

export function blockGeometry(layout: Layout): BlockGeometry {
  const spec = SPECS[layout]
  const gridUnits = MAX_BLOCKS * TILE_UNITS
  const panel = Array.from({ length: PANEL_LINES }, (_, index) => ({
    x: spec.panelX,
    y: spec.panelY + index * spec.panelPitch + (index > GAP_AFTER ? PANEL_GAP_UNITS : 0),
  }))
  return {
    widthUnits: VIEW_WIDTH_UNITS[layout],
    heightUnits: spec.heightUnits,
    plateLabel: "FIG / CTX / BLOCKS",
    originX: spec.originX,
    originY: ORIGIN_Y,
    tileUnits: TILE_UNITS,
    unitsPerElement: TILE_UNITS / TILE,
    ghostStepUnits: GHOST_STEP_UNITS,
    tileRect: (bx, by) => ({
      x: spec.originX + by * TILE_UNITS,
      y: ORIGIN_Y + bx * TILE_UNITS,
      width: TILE_UNITS,
      height: TILE_UNITS,
    }),
    rowIndex: bx => ({ x: spec.originX - 8, y: ORIGIN_Y + (bx + 0.5) * TILE_UNITS, text: String(bx), anchor: "end" }),
    colIndex: by => ({
      x: spec.originX + (by + 0.5) * TILE_UNITS,
      y: ORIGIN_Y - 2 * GHOST_STEP_UNITS - 12,
      text: String(by),
      anchor: "middle",
    }),
    axisRows: {
      x: spec.originX - 30,
      y: ORIGIN_Y + gridUnits / 2,
      text: "ctx.block.x: rows of C (M)",
      anchor: "middle",
      rotate: -90,
    },
    axisCols: { x: spec.originX, y: 48, text: "ctx.block.y: columns of C (N)", anchor: "start" },
    panel,
    panelGapAfter: GAP_AFTER,
  }
}
