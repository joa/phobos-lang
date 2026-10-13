// Layout of the hadamard-rotation figure, pure so the tests can check it: a row of 16 weights on the
// left, the rotation network in the middle (sign flips, then four butterfly stages), and the row
// rounded to -1, 0, +1 on the right.

import type { Layout } from "../../../lib/figure/layout"
import type { Rect, TextAt } from "../../inference/figure-geometry"
import { LANE_COUNT, MODEL_BLOCK, MODEL_STAGES, OUTLIER_VALUE, STAGE_COUNT } from "./model"

export interface HadamardGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  laneY: number[]
  barHeight: number
  leftAxis: number
  rightAxis: number
  unitsPerValue: number
  network: { left: number; right: number; signX: number; stageX: number[]; span: number }
  labels: { left: TextAt; network: TextAt; right: TextAt; signs: TextAt; stages: TextAt; note: TextAt[] }
  codeLabelY: number
  top: number
  bottom: number
  // A bar from the axis to value v, as a rect.
  bar(axisX: number, lane: number, value: number): Rect
  // The crossing lines of one butterfly stage: lanes 2^stage apart.
  butterfly(stage: number): string
}

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  laneTop: number
  lanePitch: number
  barHeight: number
  leftAxis: number
  rightAxis: number
  halfWidth: number
  network: { left: number; right: number; signX: number; stageX: number[]; span: number }
  labelY: number
  subLabelY: number
  codeLabelOffset: number
  noteY: number
  text: { left: [number, string]; network: string; right: string; signs: string; stages: string; note: string[] }
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: 720,
    heightUnits: 324,
    plateLabel: "FIG / TERNARY / ROTATE, THEN ROUND",
    laneTop: 72,
    lanePitch: 13,
    barHeight: 8,
    leftAxis: 98,
    rightAxis: 612,
    halfWidth: 72,
    network: { left: 180, right: 534, signX: 206, stageX: [272, 336, 400, 464], span: 24 },
    labelY: 42,
    subLabelY: 58,
    codeLabelOffset: 16,
    noteY: 306,
    text: {
      left: [26, "a row of weights"],
      network: "Hadamard rotation",
      right: "rounded to -1, 0, +1",
      signs: "signs",
      stages: `${STAGE_COUNT} butterfly stages`,
      note: [
        `adds and subtracts only, then / ${Math.sqrt(LANE_COUNT)}; the model's blocks are ${MODEL_BLOCK.toLocaleString("en-US")} wide, ${MODEL_STAGES} stages`,
      ],
    },
  },
  narrow: {
    widthUnits: 400,
    heightUnits: 336,
    plateLabel: "FIG / ROTATE, THEN ROUND",
    laneTop: 72,
    lanePitch: 11.5,
    barHeight: 7,
    leftAxis: 62,
    rightAxis: 338,
    halfWidth: 46,
    network: { left: 112, right: 288, signX: 130, stageX: [168, 202, 236, 270], span: 13 },
    labelY: 42,
    subLabelY: 58,
    codeLabelOffset: 14,
    noteY: 296,
    text: {
      left: [16, "weights"],
      network: "rotation",
      right: "rounded",
      signs: "signs",
      stages: `${STAGE_COUNT} stages`,
      note: [
        `adds and subtracts, then / ${Math.sqrt(LANE_COUNT)}`,
        `the model's blocks: ${MODEL_BLOCK.toLocaleString("en-US")} wide`,
      ],
    },
  },
}

export function hadamardGeometry(layout: Layout): HadamardGeometry {
  const spec = SPECS[layout]
  const laneY = Array.from({ length: LANE_COUNT }, (_, lane) => spec.laneTop + lane * spec.lanePitch)
  const unitsPerValue = spec.halfWidth / OUTLIER_VALUE
  const bottom = laneY[LANE_COUNT - 1] + spec.barHeight / 2
  const network = spec.network
  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    laneY,
    barHeight: spec.barHeight,
    leftAxis: spec.leftAxis,
    rightAxis: spec.rightAxis,
    unitsPerValue,
    network,
    labels: {
      left: { x: spec.text.left[0], y: spec.labelY, text: spec.text.left[1] },
      network: { x: (network.left + network.right) / 2, y: spec.labelY, text: spec.text.network, anchor: "middle" },
      right: { x: spec.rightAxis, y: spec.labelY, text: spec.text.right, anchor: "middle" },
      signs: { x: network.signX, y: spec.subLabelY, text: spec.text.signs, anchor: "middle", sizeUnits: 11 },
      stages: {
        x: (network.stageX[0] + network.stageX[network.stageX.length - 1]) / 2,
        y: spec.subLabelY,
        text: spec.text.stages,
        anchor: "middle",
        sizeUnits: 11,
      },
      note: spec.text.note.map((text, index) => ({
        x: (network.left + network.right) / 2,
        y: spec.noteY + index * 15,
        text,
        anchor: "middle" as const,
        sizeUnits: 11,
      })),
    },
    codeLabelY: bottom + spec.codeLabelOffset,
    top: laneY[0] - spec.barHeight / 2,
    bottom,
    bar(axisX, lane, value) {
      const length = value * unitsPerValue
      return {
        x: Math.min(axisX, axisX + length),
        y: laneY[lane] - spec.barHeight / 2,
        width: Math.abs(length),
        height: spec.barHeight,
      }
    },
    butterfly(stage) {
      const x = network.stageX[stage]
      const half = 2 ** stage
      const parts: string[] = []
      for (let start = 0; start < LANE_COUNT; start += 2 * half) {
        for (let lane = start; lane < start + half; lane += 1) {
          const [a, b] = [laneY[lane], laneY[lane + half]]
          parts.push(
            `M${x - network.span} ${a}L${x + network.span} ${b}M${x - network.span} ${b}L${x + network.span} ${a}`
          )
        }
      }
      return parts.join("")
    },
  }
}
