// Model and geometry of the ctx-counts figure, pure so the tests can check them. A CTA of
// @launch(threads) threads drawn as warps, one row of ctx.lanes cells per warp. A count that is not
// a whole number of warps (the page's @launch(48)) is drawn with its missing half warp as cost, and
// its error is the compiler's own text from the docs' expect-error fence.

import errors from "../../../data/expected-errors.json"
import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { isWholeWarps, LANES, MAX_THREADS, warpsOf } from "../ctx-hierarchy/rules"
import { wrapText } from "../ctx-hierarchy/text"

export const ERROR_KEY = "website/src/content/docs/docs/ctx/counts.mdx#1"
export const INVALID_THREADS = 48
// Drawn as plain text, so without the backticks that mark code.
export const LAUNCH_ERROR: string = (errors as Record<string, string>)[ERROR_KEY].replaceAll("`", "")
export const MAX_ROWS = MAX_THREADS / LANES

export type CellState = "on" | "warn" | "off"

export interface CountsModel {
  threads: number
  rows: number
  // null when the count is not a whole number of warps.
  warps: number | null
  error: string | null
  cellState(row: number, lane: number): CellState
}

export function countsModel(threads: number, invalid: boolean): CountsModel {
  const count = invalid ? INVALID_THREADS : threads
  const whole = isWholeWarps(count)
  const rows = Math.ceil(count / LANES)
  return {
    threads: count,
    rows,
    warps: whole ? warpsOf(count) : null,
    error: whole ? null : LAUNCH_ERROR,
    cellState: (row, lane) => (row * LANES + lane < count ? "on" : row < rows ? "warn" : "off"),
  }
}

export const CELL_UNITS = 5
export const CELL_PITCH_UNITS = 6
export const ROW_PITCH_UNITS = 8

export type TextAt = { x: number; y: number }

export interface CountsGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  cellsX: number
  warpLabelX: number
  bracketX: number
  areaTop: number
  areaBottom: number
  // The block's top edge for a number of rows: the rows stay centered in the area.
  blockTop(rows: number): number
  launch: TextAt
  fields: { name: TextAt; note: TextAt }[]
  errorLabel: TextAt
  errorLines: TextAt[]
  errorMaxChars: number
}

interface Spec {
  heightUnits: number
  plateLabel: string
  cellsX: number
  areaTop: number
  panelX: number
  panelY: number
  fieldPitch: number
  noteDx: number
  noteDy: number
  errorGap: number
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    heightUnits: 340,
    plateLabel: "FIG / CTX / COUNTS",
    cellsX: 124,
    areaTop: 72,
    panelX: 470,
    panelY: 72,
    fieldPitch: 46,
    noteDx: 0,
    noteDy: 16,
    errorGap: 54,
  },
  narrow: {
    heightUnits: 532,
    plateLabel: "FIG / CTX / COUNTS",
    cellsX: 96,
    areaTop: 64,
    panelX: 24,
    panelY: 342,
    fieldPitch: 32,
    noteDx: 0,
    noteDy: 15,
    errorGap: 30,
  },
}

export const ERROR_LINE_UNITS = 16
const SMALL_ADVANCE_UNITS = 6.6

export function countsGeometry(layout: Layout): CountsGeometry {
  const spec = SPECS[layout]
  const widthUnits = VIEW_WIDTH_UNITS[layout]
  const blockWidth = (LANES - 1) * CELL_PITCH_UNITS + CELL_UNITS
  const blockHeight = (rows: number) => (rows - 1) * ROW_PITCH_UNITS + CELL_UNITS
  const areaBottom = spec.areaTop + blockHeight(MAX_ROWS)
  const areaMid = (spec.areaTop + areaBottom) / 2
  const fieldsTop = spec.panelY + (layout === "wide" ? 36 : 28)
  const fields = [0, 1, 2].map(index => {
    const y = fieldsTop + index * spec.fieldPitch
    return { name: { x: spec.panelX, y }, note: { x: spec.panelX + spec.noteDx, y: y + spec.noteDy } }
  })
  const errorLabelY = fields[2].note.y + spec.errorGap
  const errorMaxChars = Math.floor((widthUnits - 14 - spec.panelX) / SMALL_ADVANCE_UNITS)
  const lines = wrapText(LAUNCH_ERROR ?? "", errorMaxChars)
  return {
    widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    cellsX: spec.cellsX,
    warpLabelX: spec.cellsX - 8,
    bracketX: spec.cellsX + blockWidth + 10,
    areaTop: spec.areaTop,
    areaBottom,
    blockTop: rows => areaMid - blockHeight(rows) / 2,
    launch: { x: spec.panelX, y: spec.panelY },
    fields,
    errorLabel: { x: spec.panelX, y: errorLabelY },
    errorLines: lines.map((_, index) => ({ x: spec.panelX, y: errorLabelY + 18 + index * ERROR_LINE_UNITS })),
    errorMaxChars,
  }
}

// Which warp rows get a label: every fourth, and the last when it is not crowding the one before.
export function labeledRows(rows: number): number[] {
  const labeled: number[] = []
  for (let row = 0; row < rows; row += 1) if (row % 4 === 0 || (row === rows - 1 && row % 4 >= 2)) labeled.push(row)
  return labeled
}
