// Layout and program of the row-sweep figure, pure so the tests can check them. The program is a
// library function's own statements: one that declares a value from a row reduction folds every
// row into a `[R, 1]` column; any other is a sweep over the tile that broadcasts the column back,
// and reads the weight row where it names `w` or `b`.

import type { Layout } from "../../../lib/figure/layout"
import { libraryFunction } from "../../../lib/stdlib/sources"
import type { Rect, TextAt } from "../../inference/figure-geometry"

export const ROWS = 5
export const COLS = 8
export const FUNCTIONS = ["softmax", "rms_norm", "layer_norm"] as const
export type RowFunction = (typeof FUNCTIONS)[number]

export interface Step {
  statement: string
  kind: "fold" | "sweep"
  // The column a fold writes, by its name in the source.
  column?: string
  readsWeight: boolean
}

// The statements of `name`'s body, each a step.
export function programOf(name: RowFunction): { tile: string; steps: Step[] } {
  const fn = libraryFunction(name)
  const lines = fn.source
    .split("\n")
    .slice(1, -1)
    .map(line => line.trim())
    .filter(Boolean)
  const steps = lines.map((statement): Step => {
    const fold = /^var\s+([A-Za-z_]\w*)\s*=.*\brow(max|sum|mean)\(/.exec(statement)
    return {
      statement,
      kind: fold ? "fold" : "sweep",
      column: fold?.[1],
      readsWeight: !fold && /\b[wb]\b/.test(statement),
    }
  })
  return { tile: fn.params[0], steps }
}

export interface RowSweepGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  cells: Rect[][]
  column: Rect[]
  weight: Rect[]
  labels: { tile: TextAt; column: TextAt; weight: TextAt; statement: TextAt }
}

const CELL_UNITS = 18
const PITCH_UNITS = 22

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  tile: { x: number; y: number }
  columnGapUnits: number
  statement: { x: number; y: number }
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: 720,
    heightUnits: 268,
    plateLabel: "FIG / STDLIB / ROW-WISE",
    tile: { x: 256, y: 116 },
    columnGapUnits: 48,
    statement: { x: 360, y: 48 },
  },
  narrow: {
    widthUnits: 400,
    heightUnits: 290,
    plateLabel: "FIG / ROW-WISE",
    tile: { x: 76, y: 128 },
    columnGapUnits: 44,
    statement: { x: 200, y: 52 },
  },
}

export function rowSweepGeometry(layout: Layout): RowSweepGeometry {
  const spec = SPECS[layout]
  const { x, y } = spec.tile
  const at = (row: number, col: number): Rect => ({
    x: x + col * PITCH_UNITS,
    y: y + row * PITCH_UNITS,
    width: CELL_UNITS,
    height: CELL_UNITS,
  })
  const cells = Array.from({ length: ROWS }, (_, row) => Array.from({ length: COLS }, (_, col) => at(row, col)))
  const tileRight = x + (COLS - 1) * PITCH_UNITS + CELL_UNITS
  const columnX = tileRight + spec.columnGapUnits
  const column = Array.from({ length: ROWS }, (_, row) => ({ ...at(row, 0), x: columnX }))
  const weightY = y - 2 * PITCH_UNITS
  const weight = Array.from({ length: COLS }, (_, col) => ({ ...at(0, col), y: weightY }))
  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    cells,
    column,
    weight,
    labels: {
      tile: { x: x - 14, y: y + ((ROWS - 1) * PITCH_UNITS + CELL_UNITS) / 2, text: "t", anchor: "end", sizeUnits: 13.5 },
      column: { x: columnX + CELL_UNITS / 2, y: y - 16, text: "m", anchor: "middle", sizeUnits: 13.5 },
      weight: { x: x - 14, y: weightY + CELL_UNITS / 2, text: "w", anchor: "end", sizeUnits: 13.5 },
      statement: { x: spec.statement.x, y: spec.statement.y, text: "", anchor: "middle", sizeUnits: 13.5 },
    },
  }
}
