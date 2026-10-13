// Cubic Bezier routes: one path that SVG draws and the canvas samples, so a stream flows exactly on its
// wire. Pure.

import type { Point } from "../../../lib/figure/polyline"
import { round } from "./types"

/** A route: a start point and cubic segments, each three points (two controls and an end). */
export interface Route {
  start: Point
  segments: [Point, Point, Point][]
}

export function mapRoute(route: Route, map: (point: Point) => Point): Route {
  return {
    start: map(route.start),
    segments: route.segments.map(([a, b, c]) => [map(a), map(b), map(c)]),
  }
}

export function routePath(route: Route): string {
  const p = (point: Point) => `${round(point.x)} ${round(point.y)}`
  return `M${p(route.start)}` + route.segments.map(([a, b, c]) => `C${p(a)} ${p(b)} ${p(c)}`).join("")
}

function cubicAt(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const u = 1 - t
  const a = u * u * u
  const b = 3 * u * u * t
  const c = 3 * u * t * t
  const d = t * t * t
  return { x: a * p0.x + b * p1.x + c * p2.x + d * p3.x, y: a * p0.y + b * p1.y + c * p2.y + d * p3.y }
}

export function sampleRoute(route: Route, samplesPerSegment = 24): Point[] {
  const points: Point[] = [route.start]
  let from = route.start
  for (const [a, b, c] of route.segments) {
    for (let index = 1; index <= samplesPerSegment; index += 1) {
      points.push(cubicAt(from, a, b, c, index / samplesPerSegment))
    }
    from = c
  }
  return points
}

/** An S-curve from a to b leaving and arriving along the x axis, bowed sideways by bowUnits. */
export function sCurve(a: Point, b: Point, bowUnits = 0): Route {
  const dx = (b.x - a.x) * 0.5
  return {
    start: a,
    segments: [
      [
        { x: a.x + dx, y: a.y + bowUnits },
        { x: b.x - dx, y: b.y + bowUnits },
        b,
      ],
    ],
  }
}
