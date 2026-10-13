// A path of straight segments measured by length, so a packet can move along it at constant speed. Pure.

export interface Point {
  x: number
  y: number
}

export class Polyline {
  #points: Point[]
  #cumulativeUnits: number[]

  constructor(points: Point[]) {
    if (points.length < 2) throw new Error("a polyline needs at least two points")
    this.#points = points
    this.#cumulativeUnits = [0]
    for (let index = 1; index < points.length; index += 1) {
      const from = points[index - 1]
      const to = points[index]
      this.#cumulativeUnits.push(this.#cumulativeUnits[index - 1] + Math.hypot(to.x - from.x, to.y - from.y))
    }
  }

  get points(): readonly Point[] {
    return this.#points
  }

  get lengthUnits(): number {
    return this.#cumulativeUnits[this.#cumulativeUnits.length - 1]
  }

  get start(): Point {
    return this.#points[0]
  }

  get end(): Point {
    return this.#points[this.#points.length - 1]
  }

  // The point at a distance from the start, clamped to the ends.
  pointAt(distanceUnits: number): Point {
    if (distanceUnits <= 0) return { ...this.start }
    if (distanceUnits >= this.lengthUnits) return { ...this.end }
    let index = 1
    while (this.#cumulativeUnits[index] < distanceUnits) index += 1
    const segmentStartUnits = this.#cumulativeUnits[index - 1]
    const segmentUnits = this.#cumulativeUnits[index] - segmentStartUnits
    const t = segmentUnits === 0 ? 0 : (distanceUnits - segmentStartUnits) / segmentUnits
    const from = this.#points[index - 1]
    const to = this.#points[index]
    return { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t }
  }

  pointAtFraction(fraction: number): Point {
    return this.pointAt(fraction * this.lengthUnits)
  }

  // The same route without repeated points or points that lie on a straight run, so a wire whose
  // elbow came out straight gets no vertex dots.
  simplified(): Polyline {
    const kept: Point[] = []
    for (const point of this.#points) {
      const last = kept[kept.length - 1]
      if (last && last.x === point.x && last.y === point.y) continue
      const beforeLast = kept[kept.length - 2]
      if (last && beforeLast && isCollinear(beforeLast, last, point)) kept.pop()
      kept.push(point)
    }
    return kept.length >= 2 ? new Polyline(kept) : this
  }

  toPathData(): string {
    return this.#points.map((point, index) => `${index === 0 ? "M" : "L"}${round(point.x)} ${round(point.y)}`).join("")
  }
}

// A right-angled route between two points, horizontal first ("h") or vertical first ("v"), turning
// halfway, so wires between boxes look drafted rather than drawn freehand.
export function elbow(from: Point, to: Point, firstAxis: "h" | "v" = "h"): Polyline {
  if (from.x === to.x || from.y === to.y) return new Polyline([from, to])
  if (firstAxis === "h") {
    const midX = (from.x + to.x) / 2
    return new Polyline([from, { x: midX, y: from.y }, { x: midX, y: to.y }, to])
  }
  const midY = (from.y + to.y) / 2
  return new Polyline([from, { x: from.x, y: midY }, { x: to.x, y: midY }, to])
}

function isCollinear(a: Point, b: Point, c: Point): boolean {
  return Math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) < 1e-6
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
