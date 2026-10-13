// Model and geometry of the dot figure, pure so the tests can check them. One CTA through gemm's k
// loop, schematic sizes: TILE_M = TILE_N = 64, TILE_K = 16, K = 64, so four k steps. Panel kt of the
// global row band of A and of the column band of B is copied into shared a and b, and
// acc += dot(a, b) adds their product. The tensor-core overlay is the grid of 16 x 16 fragments,
// tiled by the CTA's 8 warps; the FMA overlay is a block of acc per thread.

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { Polyline } from "../../../lib/figure/polyline"
import { DEFAULT_THREADS, warpsOf } from "../ctx-hierarchy/rules"

export const TILE = 64
export const TILE_K = 16
export const K = 64
export const K_STEPS = K / TILE_K
export const FRAGMENT = 16
export const FRAGMENTS = TILE / FRAGMENT
export const WARPS = warpsOf(DEFAULT_THREADS)
export const FRAGMENTS_PER_WARP = (FRAGMENTS * FRAGMENTS) / WARPS
// Schematic: the FMA overlay splits acc into one block per thread, 16 x 16 blocks for 256 threads.
export const THREAD_BLOCKS = Math.sqrt(DEFAULT_THREADS)

export type Path = "fma" | "tc"

// The warp that multiplies acc's fragment (row, col): each warp takes FRAGMENTS_PER_WARP neighbors
// of one fragment row.
export function fragmentWarp(row: number, col: number): number {
  return row * (FRAGMENTS / FRAGMENTS_PER_WARP) + Math.floor(col / FRAGMENTS_PER_WARP)
}

// The slices gemm copies at k step kStep for block (0, 0).
export function slicesOf(kStep: number): { a: string; b: string; kt: number } {
  const kt = kStep * TILE_K
  return { a: `A[0:${TILE}, ${kt}:${kt + TILE_K}]`, b: `B[${kt}:${kt + TILE_K}, 0:${TILE}]`, kt }
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

export const CODE_LINES = [
  "var acc: tile<f32>[64, 64] = 0.0",
  "for kt in range(0, K, 16) {",
  "  var a = A[pm * 64 :+ 64, kt :+ 16]",
  "  var b = B[kt :+ 16, pn * 64 :+ 64]",
  "  acc += dot(a, b)",
  "}",
]
export const COPY_LINES = [2, 3]
export const DOT_LINE = 4

export interface DotGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  globalA: { panels: Rect[]; bounds: Rect; label: TextAt }
  globalB: { panels: Rect[]; bounds: Rect; label: TextAt }
  sharedA: { parts: Rect[]; bounds: Rect; label: TextAt }
  sharedB: { parts: Rect[]; bounds: Rect; label: TextAt }
  acc: { bounds: Rect; label: TextAt; fragmentPath: string; threadPath: string; threadBlock: Rect }
  warpLabels: TextAt[]
  ruler: { segments: Rect[]; label: TextAt }
  joinA: Polyline
  joinB: Polyline
  code: TextAt[]
  tensorcore: TextAt
  notes: TextAt[]
  wireA(kStep: number): Polyline
  wireB(kStep: number): Polyline
}

interface Spec {
  heightUnits: number
  plateLabel: string
  b: { x: number; y: number }
  a: { x: number }
  acc: { x: number; y: number }
  code: { x: number; y: number; pitch: number }
  notesY: number
  // Wide puts the ruler's count beside the ruler, narrow under it.
  rulerLabelBeside: boolean
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    heightUnits: 340,
    plateLabel: "FIG / DOT / K LOOP",
    b: { x: 36, y: 60 },
    a: { x: 230 },
    acc: { x: 270, y: 206 },
    code: { x: 430, y: 74, pitch: 20 },
    notesY: 216,
    rulerLabelBeside: true,
  },
  narrow: {
    heightUnits: 540,
    plateLabel: "FIG / DOT",
    b: { x: 24, y: 60 },
    a: { x: 182 },
    acc: { x: 222, y: 210 },
    code: { x: 24, y: 376, pitch: 18 },
    notesY: 490,
    rulerLabelBeside: false,
  },
}

export const UNITS_PER_ELEMENT = 1.25
const PANEL_GAP_UNITS = 3
const PART_GAP_UNITS = 1
const WIRE_STANDOFF_UNITS = 8
const LABEL_RISE_UNITS = 13
// The advance of a 12-unit Noto Sans Mono label.
export const CODE_ADVANCE_UNITS = 7.2

export function dotGeometry(layout: Layout): DotGeometry {
  const spec = SPECS[layout]
  const u = UNITS_PER_ELEMENT
  const tileUnits = TILE * u
  const kUnits = TILE_K * u

  const acc: Rect = { x: spec.acc.x, y: spec.acc.y, width: tileUnits, height: tileUnits }

  const panelsB = Array.from({ length: K_STEPS }, (_, kStep) => ({
    x: spec.b.x,
    y: spec.b.y + kStep * (kUnits + PANEL_GAP_UNITS),
    width: tileUnits,
    height: kUnits,
  }))
  const globalBBounds = boundsOf(panelsB)
  const aX = layout === "wide" ? 36 : 24
  const panelsA = Array.from({ length: K_STEPS }, (_, kStep) => ({
    x: aX + kStep * (kUnits + PANEL_GAP_UNITS),
    y: acc.y,
    width: kUnits,
    height: tileUnits,
  }))
  const globalABounds = boundsOf(panelsA)

  const partUnits = FRAGMENT * u
  const sharedAParts = Array.from({ length: FRAGMENTS }, (_, index) => ({
    x: spec.a.x,
    y: acc.y + index * partUnits + PART_GAP_UNITS / 2,
    width: kUnits,
    height: partUnits - PART_GAP_UNITS,
  }))
  const sharedA: Rect = { x: spec.a.x, y: acc.y, width: kUnits, height: tileUnits }
  const bY = acc.y - 26 - kUnits
  const sharedBParts = Array.from({ length: FRAGMENTS }, (_, index) => ({
    x: acc.x + index * partUnits + PART_GAP_UNITS / 2,
    y: bY,
    width: partUnits - PART_GAP_UNITS,
    height: kUnits,
  }))
  const sharedB: Rect = { x: acc.x, y: bY, width: tileUnits, height: kUnits }

  const gridPath = (count: number) => {
    const step = tileUnits / count
    const parts: string[] = []
    for (let index = 1; index < count; index += 1) {
      const offset = round(index * step)
      parts.push(`M${acc.x + offset} ${acc.y}V${acc.y + tileUnits}`, `M${acc.x} ${acc.y + offset}H${acc.x + tileUnits}`)
    }
    return parts.join("")
  }
  const blockUnits = tileUnits / THREAD_BLOCKS
  const warpLabels: TextAt[] = []
  for (let row = 0; row < FRAGMENTS; row += 1) {
    for (let col = 0; col < FRAGMENTS; col += 1) {
      warpLabels.push({
        x: acc.x + (col + 0.5) * partUnits,
        y: acc.y + (row + 0.5) * partUnits,
        text: `w${fragmentWarp(row, col)}`,
        anchor: "middle",
      })
    }
  }

  const labelBelowY = acc.y + tileUnits + 14
  const rulerY = labelBelowY + 12
  const segmentUnits = (tileUnits - (K_STEPS - 1) * 3) / K_STEPS
  const segments = Array.from({ length: K_STEPS }, (_, kStep) => ({
    x: acc.x + kStep * (segmentUnits + 3),
    y: rulerY,
    width: segmentUnits,
    height: 5,
  }))
  const rulerLabel: TextAt = spec.rulerLabelBeside
    ? { x: acc.x + tileUnits + 10, y: rulerY + 2.5, text: "" }
    : { x: acc.x, y: rulerY + 19, text: "" }

  const midA = acc.y + tileUnits / 2
  const sharedBCenterX = sharedB.x + sharedB.width / 2
  const wireA = (kStep: number) =>
    new Polyline([
      { x: panelsA[kStep].x + panelsA[kStep].width, y: midA },
      { x: sharedA.x - WIRE_STANDOFF_UNITS, y: midA },
    ])
  const wireB = (kStep: number) => {
    const y = panelsB[kStep].y + kUnits / 2
    return new Polyline([
      { x: globalBBounds.x + globalBBounds.width, y },
      { x: sharedBCenterX, y },
      { x: sharedBCenterX, y: sharedB.y - WIRE_STANDOFF_UNITS },
    ])
  }

  // SVG collapses leading spaces, so indentation becomes an x offset.
  const code = CODE_LINES.map((line, index) => {
    const text = line.trimStart()
    return {
      x: spec.code.x + (line.length - text.length) * CODE_ADVANCE_UNITS,
      y: spec.code.y + index * spec.code.pitch,
      text,
    }
  })
  return {
    widthUnits: VIEW_WIDTH_UNITS[layout],
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    globalA: {
      panels: panelsA,
      bounds: globalABounds,
      label: { x: globalABounds.x, y: globalABounds.y - LABEL_RISE_UNITS, text: "global  A  64 x K" },
    },
    globalB: {
      panels: panelsB,
      bounds: globalBBounds,
      label: { x: globalBBounds.x, y: globalBBounds.y - LABEL_RISE_UNITS, text: "global  B  K x 64" },
    },
    sharedA: {
      parts: sharedAParts,
      bounds: sharedA,
      label: { x: sharedA.x + sharedA.width, y: labelBelowY, text: "shared  a", anchor: "end" },
    },
    sharedB: {
      parts: sharedBParts,
      bounds: sharedB,
      label: { x: sharedB.x + sharedB.width + 8, y: sharedB.y + sharedB.height / 2, text: "shared  b" },
    },
    acc: {
      bounds: acc,
      label: { x: acc.x, y: labelBelowY, text: "acc  64 x 64 f32" },
      fragmentPath: gridPath(FRAGMENTS),
      threadPath: gridPath(THREAD_BLOCKS),
      threadBlock: { x: acc.x, y: acc.y, width: blockUnits, height: blockUnits },
    },
    warpLabels,
    ruler: { segments, label: rulerLabel },
    joinA: new Polyline([
      { x: sharedA.x + sharedA.width + 3, y: midA },
      { x: acc.x - 3, y: midA },
    ]),
    joinB: new Polyline([
      { x: sharedBCenterX, y: sharedB.y + sharedB.height + 3 },
      { x: sharedBCenterX, y: acc.y - 3 },
    ]),
    code,
    tensorcore: { x: spec.code.x, y: spec.code.y - spec.code.pitch, text: "@tensorcore" },
    notes: [0, 1, 2].map(index => ({ x: spec.code.x, y: spec.notesY + index * 16, text: "" })),
    wireA,
    wireB,
  }
}

export const NOTES: Record<Path, string[]> = {
  fma: ["f32 FMA path", "each thread owns a block of acc", "and adds into it in registers"],
  tc: ["tensor cores, 16 x 16 fragments", "8 warps, 2 fragments of acc each", "f32 operands rounded to f16"],
}

export function boundsOf(rects: Rect[]): Rect {
  const minX = Math.min(...rects.map(rect => rect.x))
  const minY = Math.min(...rects.map(rect => rect.y))
  const maxX = Math.max(...rects.map(rect => rect.x + rect.width))
  const maxY = Math.max(...rects.map(rect => rect.y + rect.height))
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
