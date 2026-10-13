// The hero's starfield, generated at build time from a seed so every build draws the same sky.
// Coordinates are in a 1600 x 1000 viewBox that the SVG crops to the hero ("slice").

import { mulberry32 } from "../../lib/figure/math"

export const SKY_WIDTH_UNITS = 1600
export const SKY_HEIGHT_UNITS = 1000

export interface Star {
  x: number
  y: number
  radiusUnits: number
  opacity: number
  // A few stars carry a hue: "b" for the cold blue, "n" for the nebula violet.
  tint: "" | "b" | "n"
  // Twinkling stars only: the animation delay and period, staggered so no two pulse together.
  delayMs?: number
  periodMs?: number
}

// Three depth layers, far to near: each drifts and follows the pointer by its own amount, so the sky
// has parallax. Twinkling stars ride the middle layer.
export interface Starfield {
  far: Star[]
  middle: Star[]
  near: Star[]
  twinkling: Star[]
}

const FAR_COUNT = 320
const MIDDLE_COUNT = 140
const NEAR_COUNT = 42
const TWINKLE_COUNT = 40

function round(value: number, digits = 1): number {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

export function makeStarfield(seed = 0x9a0b05): Starfield {
  const random = mulberry32(seed)
  // Fewer stars low in the sky, where the limb of Mars rises.
  const y = () => round(SKY_HEIGHT_UNITS * random() ** 1.35)
  const x = () => round(SKY_WIDTH_UNITS * random())
  const tint = (): Star["tint"] => {
    const roll = random()
    return roll < 0.06 ? "b" : roll < 0.09 ? "n" : ""
  }

  const far = Array.from({ length: FAR_COUNT }, () => ({
    x: x(),
    y: y(),
    radiusUnits: round(0.4 + random() * 0.5, 2),
    opacity: round(0.2 + random() * 0.4, 2),
    tint: tint(),
  }))
  const middle = Array.from({ length: MIDDLE_COUNT }, () => ({
    x: x(),
    y: y(),
    radiusUnits: round(0.75 + random() * 0.6, 2),
    opacity: round(0.45 + random() * 0.4, 2),
    tint: tint(),
  }))
  const near = Array.from({ length: NEAR_COUNT }, () => ({
    x: x(),
    y: y(),
    radiusUnits: round(1.2 + random() * 0.8, 2),
    opacity: round(0.6 + random() * 0.35, 2),
    tint: tint(),
  }))
  const twinkling = Array.from({ length: TWINKLE_COUNT }, (_, index) => ({
    x: x(),
    y: y(),
    radiusUnits: round(1 + random() * 0.7, 2),
    opacity: round(0.7 + random() * 0.3, 2),
    tint: tint(),
    delayMs: Math.round(-index * 397 - random() * 300),
    periodMs: Math.round(3800 + random() * 3400),
  }))
  return { far, middle, near, twinkling }
}
