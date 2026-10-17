// Layout and values of the random-tile figure, pure so the tests can check them. A tile of
// counters, the flat index `(r + iota(0)) * C + iota(1)` of each element, each shaded by what the
// library's `uniform`, `randn` or `dropout` gives it, evaluated from the shipped source.

import type { Layout } from "../../../lib/figure/layout"
import { evaluator, int } from "../../../lib/stdlib/library"
import { LIBRARY } from "../../../lib/stdlib/sources"
import type { Rect, TextAt } from "../../inference/figure-geometry"

export const FUNCTIONS = ["uniform", "randn", "dropout"] as const
export type RandomFunction = (typeof FUNCTIONS)[number]
export const SEEDS = [1, 2, 3] as const
export const DROP_P = 0.25

// Each function as the figure calls it, on the counter `c`.
export function callOf(fn: RandomFunction): string {
  return fn === "dropout" ? `dropout(x, ${DROP_P}, seed, c)` : `${fn}(seed, c)`
}

// The counter of a tile `cols` wide: each element's flat index.
export function counterOf(cols: number): string {
  return `c = (r + iota(0)) * ${cols} + iota(1)`
}

export interface Draw {
  // Row-major, one a counter.
  values: number[]
  // The shade of each cell, in [0, 1].
  shades: number[]
  readout: string
}

// What `fn` gives each of `count` counters under `seed`, with `x` at 1 for dropout.
export function draw(fn: RandomFunction, seed: number, count: number): Draw {
  const f = evaluator(LIBRARY)
  const counters = Array.from({ length: count }, (_, c) => c)
  if (fn === "uniform") {
    const values = counters.map(c => f("uniform", int(seed), int(c)))
    const mean = values.reduce((a, b) => a + b, 0) / count
    return {
      values,
      shades: values,
      readout: `uniform: ${count} counters, mean ${mean.toFixed(3)}; each a multiple of 2^-23 in [0, 1).`,
    }
  }
  if (fn === "randn") {
    const values = counters.map(c => f("randn", int(seed), int(c)))
    const mean = values.reduce((a, b) => a + b, 0) / count
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / count
    return {
      values,
      shades: values.map(z => Math.min(1, Math.max(0, (z + 3) / 6))),
      readout: `randn: ${count} counters, mean ${mean.toFixed(3)}, variance ${variance.toFixed(3)}; shaded from -3 to 3.`,
    }
  }
  const values = counters.map(c => f("dropout", 1, DROP_P, int(seed), int(c)))
  const kept = values.filter(v => v !== 0).length
  return {
    values,
    shades: values.map(v => (v === 0 ? 0 : 1)),
    readout: `dropout(x, ${DROP_P}): ${kept} of ${count} kept, each scaled by 1 / ${1 - DROP_P}; the rest are 0.`,
  }
}

export interface RandomTileGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  rows: number
  cols: number
  cells: Rect[][]
  labels: { tile: TextAt; call: TextAt; counter: TextAt }
}

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  rows: number
  cols: number
  tile: { x: number; y: number }
  call: { x: number; y: number }
}

const CELL_UNITS = 13
const PITCH_UNITS = 16

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: 720,
    heightUnits: 236,
    plateLabel: "FIG / STDLIB / RANDOM",
    rows: 8,
    cols: 32,
    tile: { x: 106, y: 84 },
    call: { x: 360, y: 40 },
  },
  narrow: {
    widthUnits: 400,
    heightUnits: 300,
    plateLabel: "FIG / RANDOM",
    rows: 12,
    cols: 16,
    tile: { x: 74, y: 92 },
    call: { x: 200, y: 44 },
  },
}

export function randomTileGeometry(layout: Layout): RandomTileGeometry {
  const spec = SPECS[layout]
  const { x, y } = spec.tile
  const cells = Array.from({ length: spec.rows }, (_, row) =>
    Array.from({ length: spec.cols }, (_, col) => ({
      x: x + col * PITCH_UNITS,
      y: y + row * PITCH_UNITS,
      width: CELL_UNITS,
      height: CELL_UNITS,
    }))
  )
  const tileMiddle = y + ((spec.rows - 1) * PITCH_UNITS + CELL_UNITS) / 2
  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    rows: spec.rows,
    cols: spec.cols,
    cells,
    labels: {
      tile: { x: x - 14, y: tileMiddle, text: "Y", anchor: "end", sizeUnits: 13.5 },
      call: { x: spec.call.x, y: spec.call.y, text: "", anchor: "middle", sizeUnits: 13.5 },
      counter: { x: spec.call.x, y: spec.call.y + 20, text: counterOf(spec.cols), anchor: "middle", sizeUnits: 12.5 },
    },
  }
}
