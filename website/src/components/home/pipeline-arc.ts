// The arc the compiler scene's stations sit on: one quadratic curve, with the stations spaced evenly by
// arc length so a packet moving at constant offset-distance reaches them at even intervals. Pure.

import type { Point } from "../../lib/figure/polyline"

export const ARC_WIDTH_UNITS = 1000
export const ARC_HEIGHT_UNITS = 250

const START: Point = { x: 70, y: 196 }
const CONTROL: Point = { x: 500, y: -36 }
const END: Point = { x: 930, y: 196 }
const SAMPLES = 512

export const ARC_PATH = `M${START.x} ${START.y}Q${CONTROL.x} ${CONTROL.y} ${END.x} ${END.y}`

function pointAt(t: number): Point {
  const u = 1 - t
  return {
    x: u * u * START.x + 2 * u * t * CONTROL.x + t * t * END.x,
    y: u * u * START.y + 2 * u * t * CONTROL.y + t * t * END.y,
  }
}

// Positions of count stations at equal arc-length fractions from start to end.
export function stationPoints(count: number): { point: Point; fraction: number }[] {
  const cumulative = [0]
  let previous = pointAt(0)
  for (let index = 1; index <= SAMPLES; index += 1) {
    const point = pointAt(index / SAMPLES)
    cumulative.push(cumulative[index - 1] + Math.hypot(point.x - previous.x, point.y - previous.y))
    previous = point
  }
  const total = cumulative[SAMPLES]
  return Array.from({ length: count }, (_, station) => {
    const fraction = count === 1 ? 0 : station / (count - 1)
    const target = fraction * total
    let index = 1
    while (index < SAMPLES && cumulative[index] < target) index += 1
    const segment = cumulative[index] - cumulative[index - 1]
    const local = segment === 0 ? 0 : (target - cumulative[index - 1]) / segment
    const point = pointAt((index - 1 + local) / SAMPLES)
    return { point: { x: Math.round(point.x * 10) / 10, y: Math.round(point.y * 10) / 10 }, fraction }
  })
}

// Keyframe CSS for the scroll-driven scene: a station lights as the packet reaches it, and each
// excerpt shows while the packet is nearest its station.
export function stationKeyframes(count: number): string {
  const rules: string[] = []
  const halfWindow = count > 1 ? 50 / (count - 1) : 50
  const fadePct = 1.5
  const pct = (value: number) => `${Math.min(100, Math.max(0, Math.round(value * 100) / 100))}%`
  for (let station = 0; station < count; station += 1) {
    const at = count > 1 ? (station / (count - 1)) * 100 : 0
    rules.push(
      `@keyframes pipe-lit-${station} { 0%, ${pct(at - fadePct)} { opacity: 0 } ${pct(at)}, 100% { opacity: 1 } }`
    )
    const from = station === 0 ? -1 : at - halfWindow
    const to = station === count - 1 ? 101 : at + halfWindow
    const frames = [
      `0% { opacity: ${station === 0 ? 1 : 0} }`,
      station === 0 ? "" : `${pct(from)} { opacity: 0 } ${pct(from + fadePct)} { opacity: 1 }`,
      station === count - 1 ? "" : `${pct(to - fadePct)} { opacity: 1 } ${pct(to)} { opacity: 0 }`,
      `100% { opacity: ${station === count - 1 ? 1 : 0} }`,
    ]
    rules.push(`@keyframes pipe-ex-${station} { ${frames.filter(Boolean).join(" ")} }`)
  }
  return rules.join("\n")
}
