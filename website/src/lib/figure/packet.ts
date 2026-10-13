// Moving data is a star: a bright core with a four-point flare, a glow, and a short comet trail of
// three trailing copies with decaying opacity. Packets move at constant speed along a Polyline; the
// figure's frame loop calls place() with a distance, so a packet is a pure function of progress.

import { glow, spaceClass, type Space } from "./draw"
import type { Point, Polyline } from "./polyline"
import { svgEl } from "./svg"

const TRAIL_COPIES = 3
const DEFAULT_TRAIL_GAP_UNITS = 4.5
const CORE_RADIUS_UNITS = 2.1
const FLARE_ARM_UNITS = 5
const GLOW_RADIUS_UNITS = 11

export class StarPacket {
  #group: SVGGElement
  #head: SVGGElement
  #trail: SVGCircleElement[] = []
  #trailGapUnits: number

  constructor(
    parent: Element,
    { space = "global", trailGapUnits = DEFAULT_TRAIL_GAP_UNITS }: { space?: Space; trailGapUnits?: number } = {}
  ) {
    this.#trailGapUnits = trailGapUnits
    this.#group = svgEl("g", { class: `chart-packet ${spaceClass(space)} is-hidden` }, parent)
    for (let index = TRAIL_COPIES; index >= 1; index -= 1) {
      this.#trail.unshift(
        svgEl(
          "circle",
          { class: `chart-star-trail is-trail-${index}`, r: CORE_RADIUS_UNITS * (1 - index * 0.18) },
          this.#group
        )
      )
    }
    this.#head = svgEl("g", { class: "chart-star" }, this.#group)
    glow(this.#head, 0, 0, GLOW_RADIUS_UNITS, space)
    const a = FLARE_ARM_UNITS
    svgEl("path", { class: "chart-star-flare", d: `M${-a} 0H${a}M0 ${-a}V${a}` }, this.#head)
    svgEl("circle", { class: "chart-star-core", r: CORE_RADIUS_UNITS }, this.#head)
  }

  get element(): SVGGElement {
    return this.#group
  }

  // Puts the head at a distance along the line and the trail behind it; trail copies that would sit
  // before the start are hidden, so a packet leaving its source grows its tail.
  place(line: Polyline, distanceUnits: number): void {
    this.#group.classList.remove("is-hidden")
    setPosition(this.#head, line.pointAt(distanceUnits))
    this.#trail.forEach((copy, index) => {
      const behindUnits = distanceUnits - (index + 1) * this.#trailGapUnits
      copy.classList.toggle("is-hidden", behindUnits < 0)
      setPosition(copy, line.pointAt(Math.max(0, behindUnits)))
    })
  }

  // Fraction 0 is the start, 1 the end; outside [0, 1] the packet is hidden (not yet sent, or arrived).
  placeAtFraction(line: Polyline, fraction: number): void {
    if (fraction <= 0 || fraction >= 1) {
      this.hide()
      return
    }
    this.place(line, fraction * line.lengthUnits)
  }

  hide(): void {
    this.#group.classList.add("is-hidden")
  }
}

function setPosition(element: Element, point: Point): void {
  element.setAttribute("transform", `translate(${Math.round(point.x * 100) / 100} ${Math.round(point.y * 100) / 100})`)
}
