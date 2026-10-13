// A 16-wide toy row with one large outlier, rounded to ternary with the best possible scale, once as it
// is and once after a randomized Hadamard rotation: random sign flips, then the fast Walsh-Hadamard
// transform, log2(n) stages of butterflies that only add and subtract, scaled by 1/sqrt(n) so lengths
// do not change. phobos-gguf rotates in blocks of HADAMARD_BLOCK = 1,024, ten stages. Pure.

import { mulberry32 } from "../../../lib/figure/math"

export const LANE_COUNT = 16
export const STAGE_COUNT = Math.log2(LANE_COUNT)
export const MODEL_BLOCK = 1024
export const MODEL_STAGES = Math.log2(MODEL_BLOCK)
// PTQ1_0's file block, phobos-gguf/src/quant/ptq1_0.rs: 128 ternary weights and an f16 scale in 28 bytes.
export const PTQ1_0_BLOCK_WEIGHTS = 128
export const PTQ1_0_BLOCK_BYTES = 28
export const OUTLIER_LANE = 5
export const OUTLIER_VALUE = 6
const ROW_SEED = 1489
const SCALE_SEARCH_STEPS = 1000

export interface ToyRow {
  weights: number[]
  signs: number[]
}

export function exampleRow(): ToyRow {
  const random = mulberry32(ROW_SEED)
  // Box-Muller: two uniform draws per normal value.
  const gaussian = () => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random())
  const weights = Array.from({ length: LANE_COUNT }, gaussian)
  weights[OUTLIER_LANE] = OUTLIER_VALUE
  const signs = Array.from({ length: LANE_COUNT }, () => (random() < 0.5 ? -1 : 1))
  return { weights, signs }
}

// The values after each butterfly stage of the unnormalized transform; the last entry is the result
// scaled by 1/sqrt(n).
export function hadamardStages(values: number[]): number[][] {
  const stages: number[][] = []
  let current = values.slice()
  for (let half = 1; half < current.length; half *= 2) {
    const next = current.slice()
    for (let start = 0; start < next.length; start += 2 * half) {
      for (let lane = start; lane < start + half; lane += 1) {
        next[lane] = current[lane] + current[lane + half]
        next[lane + half] = current[lane] - current[lane + half]
      }
    }
    stages.push(next)
    current = next
  }
  const norm = Math.sqrt(values.length)
  stages[stages.length - 1] = current.map(value => value / norm)
  return stages
}

export function hadamard(values: number[]): number[] {
  const stages = hadamardStages(values)
  return stages[stages.length - 1]
}

export function relativeError(values: number[], approximation: number[]): number {
  const error = Math.hypot(...values.map((value, lane) => value - approximation[lane]))
  return error / Math.hypot(...values)
}

export interface TernaryFit {
  scale: number
  codes: number[]
  error: number
}

// Ternary rounding with whichever scale gives the smallest error, by a grid search over (0, max |v|].
export function roundTernary(values: number[]): TernaryFit {
  const maxAbs = Math.max(...values.map(Math.abs))
  let best: TernaryFit | null = null
  for (let step = 1; step <= SCALE_SEARCH_STEPS; step += 1) {
    const scale = (maxAbs * step) / SCALE_SEARCH_STEPS
    const codes = values.map(value => Math.max(-1, Math.min(1, Math.round(value / scale))))
    const error = relativeError(
      values,
      codes.map(code => code * scale)
    )
    if (best === null || error < best.error) best = { scale, codes, error }
  }
  return best!
}

export type Rotation = "off" | "on"

export const ROW = exampleRow()
export const SIGNED = ROW.weights.map((weight, lane) => weight * ROW.signs[lane])
export const STAGES = hadamardStages(SIGNED)
export const ROTATED = STAGES[STAGES.length - 1]

export const FITS: Record<Rotation, { values: number[]; fit: TernaryFit }> = {
  off: { values: ROW.weights, fit: roundTernary(ROW.weights) },
  on: { values: ROTATED, fit: roundTernary(ROTATED) },
}

function percent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`
}

export function readoutFor(rotation: Rotation): string {
  const { fit } = FITS[rotation]
  if (rotation === "off") {
    const zeros = fit.codes.filter(code => code === 0).length
    return `Rounded as is: the best scale fits the outlier, so ${zeros} of ${LANE_COUNT} weights round to 0. Rounding error ${percent(fit.error)}.`
  }
  return `Rotated first: the outlier spreads over all ${LANE_COUNT} values. Rounding error ${percent(fit.error)}, and the activations get the same rotation at run time.`
}
