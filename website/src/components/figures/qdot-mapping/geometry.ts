// Layout of the qdot-mapping figure, pure so the tests can check it. A decode matrix-vector product with
// 32 outputs and a row of 32 Q8_0 blocks (1,024 weights), mapped onto one warp of 32 lanes two ways:
// a thread per output, or a warp per output as the lanes matvecs do it, each lane reading 16 bytes a
// step and sum<over ctx.lane> folding the partial sums in five xor-shuffle stages.

import type { Layout } from "../../../lib/figure/layout"
import type { Rect, TextAt } from "../../inference/figure-geometry"

export const LANES = 32
export const ROWS = 32
export const BLOCKS = 32
export const BLOCK_WEIGHTS = 32
// Each lane reads 16 int8 weights a step, so a step of the warp covers 512 contiguous bytes.
export const BYTES_PER_LANE = 16
export const STEP_BYTES = LANES * BYTES_PER_LANE
export const STEPS = (BLOCKS * BLOCK_WEIGHTS) / STEP_BYTES
export const BLOCKS_PER_STEP = STEP_BYTES / BLOCK_WEIGHTS
// The xor masks of the butterfly sum<over ctx.lane> lowers to: 16, 8, 4, 2, 1.
export const SHUFFLE_MASKS = Array.from({ length: Math.log2(LANES) }, (_, stage) => LANES / 2 ** (stage + 1))

// The lane a lane exchanges with at a stage.
export function partnerOf(lane: number, mask: number): number {
  return lane ^ mask
}

// After every stage of the butterfly, which lanes' partial sums each lane holds.
export function butterfly(): Set<number>[][] {
  let held = Array.from({ length: LANES }, (_, lane) => new Set([lane]))
  const stages: Set<number>[][] = []
  for (const mask of SHUFFLE_MASKS) {
    held = held.map((own, lane) => new Set([...own, ...held[partnerOf(lane, mask)]]))
    stages.push(held)
  }
  return stages
}

export interface Arc {
  from: number
  to: number
  d: string
}

export interface QdotGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  activation: { blocks: Rect[]; label: TextAt }
  matrix: { rows: Rect[]; bounds: Rect; boundaries: string; label: TextAt }
  output: { cells: Rect[]; label: TextAt }
  lanes: { cells: Rect[]; label: TextAt }
  stageLabel: TextAt
  arcs: Arc[][]
  // The column of the matrix that block b covers.
  blockRect(block: number): Rect
  // The band one warp step reads in row 0, and lane l's 16 bytes inside it.
  stepRect(step: number): Rect
  laneTick(step: number, lane: number): { x: number; y1: number; y2: number }
}

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  x: number
  blockPitch: number
  activationY: number
  activationHeight: number
  matrixY: number
  rowPitch: number
  rowHeight: number
  outputX: number
  outputWidth: number
  lanesY: number
  laneCell: number
  lanePitch: number
  arcDepthBase: number
  arcDepthPerStage: number
  stageLabel: TextAt
  labels: { activation: string; matrix: string; output: string; lanes: string }
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: 720,
    heightUnits: 346,
    plateLabel: "FIG / DECODE / MATVEC MAPPING",
    x: 40,
    blockPitch: 18,
    activationY: 50,
    activationHeight: 8,
    matrixY: 82,
    rowPitch: 5,
    rowHeight: 4,
    outputX: 644,
    outputWidth: 12,
    lanesY: 262,
    laneCell: 12,
    lanePitch: 18,
    arcDepthBase: 8,
    arcDepthPerStage: 5,
    stageLabel: { x: 40, y: 330, text: "", sizeUnits: 11 },
    labels: {
      activation: "activation, int8",
      matrix: "weights, Q8_0: 32 int8 and a scale per block",
      output: "output",
      lanes: "one warp, 32 lanes",
    },
  },
  narrow: {
    widthUnits: 400,
    heightUnits: 340,
    plateLabel: "FIG / MATVEC MAPPING",
    x: 20,
    blockPitch: 10,
    activationY: 50,
    activationHeight: 6,
    matrixY: 80,
    rowPitch: 5,
    rowHeight: 4,
    outputX: 352,
    outputWidth: 10,
    lanesY: 262,
    laneCell: 8,
    lanePitch: 10,
    arcDepthBase: 6,
    arcDepthPerStage: 5,
    stageLabel: { x: 20, y: 318, text: "", sizeUnits: 11 },
    labels: { activation: "activation", matrix: "weights, Q8_0", output: "out", lanes: "one warp, 32 lanes" },
  },
}

const LABEL_RISE_UNITS = 10

export function qdotGeometry(layout: Layout): QdotGeometry {
  const spec = SPECS[layout]
  const blockWidth = spec.blockPitch - 1
  const matrixWidth = BLOCKS * spec.blockPitch - 1
  const blocks = Array.from({ length: BLOCKS }, (_, block) => ({
    x: spec.x + block * spec.blockPitch,
    y: spec.activationY,
    width: blockWidth,
    height: spec.activationHeight,
  }))
  const rows = Array.from({ length: ROWS }, (_, row) => ({
    x: spec.x,
    y: spec.matrixY + row * spec.rowPitch,
    width: matrixWidth,
    height: spec.rowHeight,
  }))
  const matrixBottom = spec.matrixY + (ROWS - 1) * spec.rowPitch + spec.rowHeight
  const bounds = { x: spec.x, y: spec.matrixY, width: matrixWidth, height: matrixBottom - spec.matrixY }
  const boundaries = Array.from({ length: BLOCKS - 1 }, (_, index) => {
    const x = spec.x + (index + 1) * spec.blockPitch - 0.5
    return `M${x} ${spec.matrixY}V${matrixBottom}`
  }).join("")
  const outputCells = Array.from({ length: ROWS }, (_, row) => ({
    x: spec.outputX,
    y: spec.matrixY + row * spec.rowPitch,
    width: spec.outputWidth,
    height: spec.rowHeight,
  }))
  const laneInset = (spec.lanePitch - spec.laneCell) / 2
  const lanes = Array.from({ length: LANES }, (_, lane) => ({
    x: spec.x + lane * spec.lanePitch + laneInset - 0.5,
    y: spec.lanesY,
    width: spec.laneCell,
    height: spec.laneCell,
  }))
  const laneCenter = (lane: number) => lanes[lane].x + spec.laneCell / 2
  const arcTop = spec.lanesY + spec.laneCell + 2
  const arcs = SHUFFLE_MASKS.map((mask, stage) => {
    const depth = spec.arcDepthBase + spec.arcDepthPerStage * (SHUFFLE_MASKS.length - 1 - stage) * 1.4
    const pairs: Arc[] = []
    for (let lane = 0; lane < LANES; lane += 1) {
      const partner = partnerOf(lane, mask)
      if (partner < lane) continue
      const [x1, x2] = [laneCenter(lane), laneCenter(partner)]
      pairs.push({
        from: lane,
        to: partner,
        d: `M${x1} ${arcTop}C${x1} ${arcTop + depth * 1.33} ${x2} ${arcTop + depth * 1.33} ${x2} ${arcTop}`,
      })
    }
    return pairs
  })

  const blockRect = (block: number): Rect => ({
    x: spec.x + block * spec.blockPitch - 1,
    y: spec.matrixY - 2,
    width: spec.blockPitch + 1,
    height: matrixBottom - spec.matrixY + 4,
  })
  const stepRect = (step: number): Rect => ({
    x: spec.x + step * BLOCKS_PER_STEP * spec.blockPitch - 1,
    y: spec.matrixY - 2,
    width: BLOCKS_PER_STEP * spec.blockPitch + 1,
    height: spec.rowHeight + 4,
  })
  const laneTick = (step: number, lane: number) => {
    const weightsPerUnit = BLOCK_WEIGHTS / spec.blockPitch
    const x = spec.x + step * BLOCKS_PER_STEP * spec.blockPitch + (lane * BYTES_PER_LANE) / weightsPerUnit
    return { x, y1: spec.matrixY - 5, y2: spec.matrixY - 1 }
  }

  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    activation: { blocks, label: { x: spec.x, y: spec.activationY - LABEL_RISE_UNITS, text: spec.labels.activation } },
    matrix: {
      rows,
      bounds,
      boundaries,
      label: { x: spec.x, y: spec.matrixY - LABEL_RISE_UNITS - 6, text: spec.labels.matrix, sizeUnits: 11 },
    },
    output: {
      cells: outputCells,
      label: { x: spec.outputX, y: spec.matrixY - LABEL_RISE_UNITS - 6, text: spec.labels.output, sizeUnits: 11 },
    },
    lanes: {
      cells: lanes,
      label: { x: spec.x, y: spec.lanesY - LABEL_RISE_UNITS, text: spec.labels.lanes, sizeUnits: 11 },
    },
    stageLabel: spec.stageLabel,
    arcs,
    blockRect,
    stepRect,
    laneTick,
  }
}
