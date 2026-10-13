// Layout of the bandwidth-bound figure, pure so the tests can check it: weights in VRAM, the memory bus,
// the SMs, and under them a log axis of operations per byte with the card's balance point.

import type { Layout } from "../../../lib/figure/layout"
import { Polyline } from "../../../lib/figure/polyline"
import { textWidthUnits, type Rect, type TextAt } from "../../inference/figure-geometry"
import {
  BALANCE_OPS_PER_BYTE,
  BANDWIDTH_LABEL,
  CARD,
  DECODE_OPS_PER_BYTE,
  PEAK_TFLOPS_LABEL,
  PROMPT_OPS_PER_BYTE,
  approx,
  logX,
  type Phase,
} from "./model"

export const WEIGHT_ROWS = 12

export interface BandwidthGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  weights: { box: Rect; rows: Rect[]; label: TextAt; note: Point }
  bus: { upper: Polyline; lower: Polyline; path: Polyline; label: TextAt; note: Point }
  sms: { box: Rect; cells: Rect[]; columns: number; label: TextAt; note: Point }
  axis: {
    x0: number
    x1: number
    y: number
    ticks: { x: number; label: TextAt }[]
    balanceX: number
    balanceLabel: TextAt
    memoryLabel: TextAt
    computeLabel: TextAt
    title: TextAt
    pointLabelY: number
  }
  pointX(phase: Phase): number
  pointLabel(phase: Phase): TextAt
}

interface Point {
  x: number
  y: number
}

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  top: number
  boxHeight: number
  weights: [number, number]
  bus: [number, number]
  sms: [number, number]
  smColumns: number
  busLabel: string
  smLabel: string
  axis: { x0: number; x1: number; y: number; balanceLabelY: number; titleY: number }
}

const CELL_UNITS = 12
const CELL_PITCH_UNITS = 16
const BUS_GAP_UNITS = 16
const LABEL_RISE_UNITS = 12
const NOTE_DROP_UNITS = 16
const ROW_PITCH_UNITS = 7
const ROW_HEIGHT_UNITS = 5

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: 720,
    heightUnits: 288,
    plateLabel: "FIG / DECODE / BYTES VS MATH",
    top: 58,
    boxHeight: 96,
    weights: [24, 164],
    bus: [172, 468],
    sms: [476, 696],
    smColumns: 12,
    busLabel: `memory bus, ${BANDWIDTH_LABEL}`,
    smLabel: `${CARD.smCount} SMs, ${PEAK_TFLOPS_LABEL} TFLOP/s`,
    axis: { x0: 60, x1: 660, y: 230, balanceLabelY: 248, titleY: 270 },
  },
  narrow: {
    widthUnits: 400,
    heightUnits: 318,
    plateLabel: "FIG / BYTES VS MATH",
    top: 58,
    boxHeight: 108,
    weights: [16, 96],
    bus: [104, 232],
    sms: [240, 384],
    smColumns: 8,
    busLabel: `bus, ${BANDWIDTH_LABEL}`,
    smLabel: `${CARD.smCount} SMs`,
    axis: { x0: 36, x1: 364, y: 240, balanceLabelY: 274, titleY: 298 },
  },
}

export function bandwidthGeometry(layout: Layout): BandwidthGeometry {
  const spec = SPECS[layout]
  const { top, boxHeight } = spec
  const midY = top + boxHeight / 2
  const noteY = top + boxHeight + NOTE_DROP_UNITS
  const labelY = top - LABEL_RISE_UNITS

  const weightsBox = { x: spec.weights[0], y: top, width: spec.weights[1] - spec.weights[0], height: boxHeight }
  const rowsHeight = (WEIGHT_ROWS - 1) * ROW_PITCH_UNITS + ROW_HEIGHT_UNITS
  const rowsTop = midY - rowsHeight / 2
  const rows = Array.from({ length: WEIGHT_ROWS }, (_, row) => ({
    x: weightsBox.x + 8,
    y: rowsTop + row * ROW_PITCH_UNITS,
    width: weightsBox.width - 16,
    height: ROW_HEIGHT_UNITS,
  }))

  const [busX0, busX1] = spec.bus
  const upper = new Polyline([
    { x: busX0, y: midY - BUS_GAP_UNITS / 2 },
    { x: busX1, y: midY - BUS_GAP_UNITS / 2 },
  ])
  const lower = new Polyline([
    { x: busX0, y: midY + BUS_GAP_UNITS / 2 },
    { x: busX1, y: midY + BUS_GAP_UNITS / 2 },
  ])
  const path = new Polyline([
    { x: busX0 + 6, y: midY },
    { x: busX1 - 6, y: midY },
  ])

  const smBox = { x: spec.sms[0], y: top, width: spec.sms[1] - spec.sms[0], height: boxHeight }
  const columns = spec.smColumns
  const smRows = CARD.smCount / columns
  const gridWidth = (columns - 1) * CELL_PITCH_UNITS + CELL_UNITS
  const gridHeight = (smRows - 1) * CELL_PITCH_UNITS + CELL_UNITS
  const gridX = smBox.x + (smBox.width - gridWidth) / 2
  const gridY = midY - gridHeight / 2
  const cells = Array.from({ length: CARD.smCount }, (_, index) => ({
    x: gridX + (index % columns) * CELL_PITCH_UNITS,
    y: gridY + Math.floor(index / columns) * CELL_PITCH_UNITS,
    width: CELL_UNITS,
    height: CELL_UNITS,
  }))

  const { x0, x1, y: axisY } = spec.axis
  const ticks = [1, 10, 100, 1000].map(value => {
    const x = logX(value, x0, x1)
    return {
      x,
      label: { x, y: axisY + 18, text: value.toLocaleString("en-US"), anchor: "middle" as const, sizeUnits: 11 },
    }
  })
  const balanceX = logX(BALANCE_OPS_PER_BYTE, x0, x1)
  const pointLabelY = axisY - 38
  const pointX = (phase: Phase) => logX(phase === "decode" ? DECODE_OPS_PER_BYTE : PROMPT_OPS_PER_BYTE, x0, x1)
  const pointLabel = (phase: Phase): TextAt => {
    const text =
      phase === "decode" ? `decode, ${approx(DECODE_OPS_PER_BYTE)}` : `prompt, ${approx(PROMPT_OPS_PER_BYTE)}`
    const width = textWidthUnits(text)
    // Centered on the point, but never past either end of the axis.
    const x = Math.min(Math.max(pointX(phase) - width / 2, x0 - 8), x1 + 8 - width)
    return { x, y: pointLabelY, text }
  }

  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    weights: {
      box: weightsBox,
      rows,
      label: { x: weightsBox.x, y: labelY, text: layout === "wide" ? "weights in VRAM" : "weights" },
      note: { x: weightsBox.x, y: noteY },
    },
    bus: {
      upper,
      lower,
      path,
      label: { x: (busX0 + busX1) / 2, y: labelY, text: spec.busLabel, anchor: "middle" },
      note: { x: (busX0 + busX1) / 2, y: noteY },
    },
    sms: {
      box: smBox,
      cells,
      columns,
      label: { x: smBox.x, y: labelY, text: spec.smLabel },
      note: { x: smBox.x, y: noteY },
    },
    axis: {
      x0,
      x1,
      y: axisY,
      ticks,
      balanceX,
      balanceLabel: {
        x: balanceX,
        y: spec.axis.balanceLabelY,
        text: `balance, ${approx(BALANCE_OPS_PER_BYTE)}`,
        anchor: "middle",
        sizeUnits: 11,
      },
      memoryLabel: { x: (x0 + balanceX) / 2, y: axisY - 14, text: "memory-bound", anchor: "middle", sizeUnits: 11 },
      computeLabel: { x: (balanceX + x1) / 2, y: axisY - 14, text: "compute-bound", anchor: "middle", sizeUnits: 11 },
      title: {
        x: (x0 + x1) / 2,
        y: spec.axis.titleY,
        text: "operations per byte, log scale",
        anchor: "middle",
        sizeUnits: 11,
      },
      pointLabelY,
    },
    pointX,
    pointLabel,
  }
}

// Notes under the three blocks, per phase.
export function notesFor(
  phase: Phase,
  busBusy: string,
  mathBusy: string
): { weights: string; bus: string; sms: string } {
  return phase === "decode"
    ? { weights: "once a token", bus: "busy 100%", sms: `busy ${mathBusy}` }
    : { weights: "once a prompt", bus: `busy ${busBusy}`, sms: "busy 100%" }
}
