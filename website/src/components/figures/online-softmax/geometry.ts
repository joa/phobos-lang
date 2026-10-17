// Layout of the online-softmax figure, pure so the tests can check it: the blocks of keys and
// values in global memory, the block's scores `s`, and the running state, the row maximum `m`, the
// denominator `l` and the output `acc`, with the two lines a block travels on.

import type { Layout } from "../../../lib/figure/layout"
import { Polyline } from "../../../lib/figure/polyline"
import type { Rect, TextAt } from "../../inference/figure-geometry"

// Schematic sizes: BR query rows, BC keys a block, D output columns, BLOCKS blocks of keys.
export const ROWS = 4
export const BC = 6
export const D = 5
export const BLOCKS = 4

export interface OnlineSoftmaxGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  blocks: Rect[]
  blockLabels: TextAt[]
  scores: Rect[][]
  m: Rect[]
  l: Rect[]
  acc: Rect[][]
  labels: { kv: TextAt; scores: TextAt; m: TextAt; l: TextAt; acc: TextAt; step: TextAt }
  // From block i to the scores, as k, and to acc, as v.
  keyLine(block: number): Polyline
  valueLine(block: number): Polyline
}

const CELL_UNITS = 14
const PITCH_UNITS = 17

function grid(x: number, y: number, rows: number, cols: number): Rect[][] {
  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: cols }, (_, col) => ({
      x: x + col * PITCH_UNITS,
      y: y + row * PITCH_UNITS,
      width: CELL_UNITS,
      height: CELL_UNITS,
    }))
  )
}

const extent = (cells: number) => (cells - 1) * PITCH_UNITS + CELL_UNITS

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  // Blocks stacked in a column (wide) or laid in a row (narrow).
  blocks: { x: number; y: number; width: number; height: number; stepX: number; stepY: number }
  scores: { x: number; y: number }
  state: { x: number; y: number }
  step: { x: number; y: number }
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: 720,
    heightUnits: 300,
    plateLabel: "FIG / STDLIB / ONLINE SOFTMAX",
    blocks: { x: 32, y: 78, width: 104, height: 34, stepX: 0, stepY: 46 },
    scores: { x: 236, y: 130 },
    state: { x: 452, y: 130 },
    step: { x: 360, y: 52 },
  },
  narrow: {
    widthUnits: 400,
    heightUnits: 420,
    plateLabel: "FIG / ONLINE SOFTMAX",
    blocks: { x: 20, y: 96, width: 82, height: 30, stepX: 92, stepY: 0 },
    scores: { x: 148, y: 196 },
    state: { x: 96, y: 318 },
    step: { x: 200, y: 52 },
  },
}

export function onlineSoftmaxGeometry(layout: Layout): OnlineSoftmaxGeometry {
  const spec = SPECS[layout]
  const b = spec.blocks
  const blocks = Array.from({ length: BLOCKS }, (_, i) => ({
    x: b.x + i * b.stepX,
    y: b.y + i * b.stepY,
    width: b.width,
    height: b.height,
  }))
  const blockLabels = blocks.map((rect, i) => ({
    x: rect.x + rect.width / 2,
    y: rect.y + rect.height / 2,
    text: `k, v ${i}`,
    anchor: "middle" as const,
    sizeUnits: 12.5,
  }))

  const scores = grid(spec.scores.x, spec.scores.y, ROWS, BC)
  const sx = spec.state.x
  const sy = spec.state.y
  const m = grid(sx, sy, ROWS, 1).map(row => row[0])
  const l = grid(sx + PITCH_UNITS + 14, sy, ROWS, 1).map(row => row[0])
  const acc = grid(sx + 2 * (PITCH_UNITS + 14), sy, ROWS, D)
  const above = (x: number, y: number, text: string, anchor: "start" | "middle" = "start"): TextAt => ({
    x,
    y: y - 16,
    text,
    anchor,
    sizeUnits: 13.5,
  })

  const scoresMidY = spec.scores.y + extent(ROWS) / 2
  const accX = acc[0][0].x
  const keyLine = (i: number): Polyline => {
    const block = blocks[i]
    if (layout === "wide") {
      const y = block.y + block.height / 2
      const midX = (block.x + block.width + spec.scores.x) / 2
      return new Polyline([
        { x: block.x + block.width, y },
        { x: midX, y },
        { x: midX, y: scoresMidY },
        { x: spec.scores.x - 8, y: scoresMidY },
      ])
    }
    const x = block.x + block.width / 2
    const midY = (block.y + block.height + spec.scores.y - 30) / 2
    return new Polyline([
      { x, y: block.y + block.height },
      { x, y: midY },
      { x: spec.scores.x + extent(BC) / 2, y: midY },
      { x: spec.scores.x + extent(BC) / 2, y: spec.scores.y - 30 },
    ])
  }
  const valueLine = (i: number): Polyline => {
    const block = blocks[i]
    if (layout === "wide") {
      // Out of the block beside its key line, under the scores, up into acc.
      const y = block.y + block.height / 2 + 8
      const railX = (block.x + block.width + spec.scores.x) / 2 - 12
      const underY = spec.scores.y + extent(ROWS) + 26
      return new Polyline([
        { x: block.x + block.width, y },
        { x: railX, y },
        { x: railX, y: underY },
        { x: accX + extent(D) / 2, y: underY },
        { x: accX + extent(D) / 2, y: sy + extent(ROWS) + 8 },
      ])
    }
    // Down out of the block, along the right edge, into acc from the side.
    const x = block.x + block.width / 2 + 8
    const railX = spec.widthUnits - 14
    const midY = (block.y + block.height + spec.scores.y - 30) / 2 + 8
    return new Polyline([
      { x, y: block.y + block.height },
      { x, y: midY },
      { x: railX, y: midY },
      { x: railX, y: sy + extent(ROWS) / 2 },
      { x: accX + extent(D) + 8, y: sy + extent(ROWS) / 2 },
    ])
  }

  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    blocks,
    blockLabels,
    scores,
    m,
    l,
    acc,
    labels: {
      kv: above(b.x, b.y, "keys, values"),
      scores: above(spec.scores.x, spec.scores.y, "s, this block"),
      m: above(m[0].x, sy, "m"),
      l: above(l[0].x, sy, "l"),
      acc: above(accX, sy, "acc"),
      step: { x: spec.step.x, y: spec.step.y, text: "", anchor: "middle", sizeUnits: 13.5 },
    },
    keyLine,
    valueLine,
  }
}
