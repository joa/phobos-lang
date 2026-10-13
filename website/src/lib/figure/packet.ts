// Moving data is a small cloud of points: a swarm of particles on three depth layers, far ones small
// and faint, near ones large and bright, strung out behind a soft glow so the cloud trails off. Each
// particle keeps a fixed lag behind the head and a fixed offset beside the path, and the offset sways
// with the distance travelled, so the swarm shimmers as it moves. Near both ends of the line the cloud
// draws together, so it leaves its source as a point and lands on its target as one.
//
// Packets move at constant speed along a Polyline; the figure's frame loop calls place() with a
// distance, so a packet is a pure function of progress. Only transforms change per frame.

import { glow, spaceClass, type Space } from "./draw"
import { mulberry32 } from "./math"
import type { Point, Polyline } from "./polyline"
import { svgEl } from "./svg"

export const DEFAULT_PARTICLES = 14
export const MAX_PARTICLES = 24
const DEFAULT_TRAIL_GAP_UNITS = 6
const GLOW_RADIUS_UNITS = 9
// Radius and opacity per depth layer, far to near.
const LAYER_RADIUS_UNITS = [0.8, 1.4, 2.2]
const LAYER_OPACITY = [0.4, 0.65, 0.95]
// Share of the particles on the near and middle layers; the rest are far.
const NEAR_SHARE = 0.2
const MIDDLE_SHARE = 0.35
const SPREAD_UNITS = 4.2
const SWAY_UNITS = 1.4
const SWAY_PER_UNIT = 0.14

let nextSeed = 1

export interface Particle {
  layer: 0 | 1 | 2
  // Distance behind the head along the line.
  lagUnits: number
  // Offset beside the path at rest.
  dx: number
  dy: number
  phase: number
  opacity: number
  radiusUnits: number
}

// The particles of one packet, deterministic for a seed, far layers first so near ones paint over
// them. The trail is three trail gaps long; particles further back are fainter.
export function cloudParticles(count: number, seed: number, trailGapUnits = DEFAULT_TRAIL_GAP_UNITS): Particle[] {
  const total = Math.max(1, Math.min(MAX_PARTICLES, Math.round(count)))
  const random = mulberry32(seed)
  const trailUnits = trailGapUnits * 3
  const particles: Particle[] = []
  for (let index = 0; index < total; index += 1) {
    const share = index / total
    const layer = share < NEAR_SHARE ? 2 : share < NEAR_SHARE + MIDDLE_SHARE ? 1 : 0
    // Near particles ride at the head, far ones spread down the trail.
    const reach = [1, 0.7, 0.35][layer]
    const lagUnits = round(random() ** 1.6 * trailUnits * reach)
    const fade = 1 - (lagUnits / trailUnits) * 0.7
    const angle = random() * Math.PI * 2
    const radius = Math.sqrt(random()) * SPREAD_UNITS * (layer === 2 ? 0.5 : 1)
    particles.push({
      layer,
      lagUnits,
      dx: round(Math.cos(angle) * radius),
      dy: round(Math.sin(angle) * radius),
      phase: round(random() * Math.PI * 2),
      opacity: round(LAYER_OPACITY[layer] * fade),
      radiusUnits: LAYER_RADIUS_UNITS[layer],
    })
  }
  return particles.sort((a, b) => a.layer - b.layer)
}

// How much of its spread the cloud has at a distance along a line: none at either end, all of it a
// trail's length in, eased between.
export function cloudSpread(distanceUnits: number, lengthUnits: number, trailUnits: number): number {
  if (trailUnits <= 0) return 1
  const room = Math.max(0, Math.min(1, Math.min(distanceUnits, lengthUnits - distanceUnits) / trailUnits))
  return room * room * (3 - 2 * room)
}

export interface StarPacketOptions {
  space?: Space
  // The trail's spacing; the cloud is three of these long.
  trailGapUnits?: number
  // Particles in the cloud, at most MAX_PARTICLES.
  count?: number
  seed?: number
}

export class StarPacket {
  #group: SVGGElement
  #head: SVGCircleElement
  #particles: { element: SVGCircleElement; spec: Particle }[]
  #trailUnits: number

  constructor(parent: Element, options: StarPacketOptions = {}) {
    const { space = "global", trailGapUnits = DEFAULT_TRAIL_GAP_UNITS, count = DEFAULT_PARTICLES } = options
    const seed = options.seed ?? nextSeed++
    this.#trailUnits = trailGapUnits * 3
    this.#group = svgEl("g", { class: `chart-packet ${spaceClass(space)} is-hidden` }, parent)
    this.#head = glow(this.#group, 0, 0, GLOW_RADIUS_UNITS, space)
    this.#particles = cloudParticles(count, seed, trailGapUnits).map(spec => ({
      spec,
      element: svgEl(
        "circle",
        { class: `chart-particle is-depth-${spec.layer}`, r: spec.radiusUnits, opacity: spec.opacity },
        this.#group
      ),
    }))
  }

  get element(): SVGGElement {
    return this.#group
  }

  get particleCount(): number {
    return this.#particles.length
  }

  // Puts the head at a distance along the line and the cloud behind and around it.
  place(line: Polyline, distanceUnits: number): void {
    this.#group.classList.remove("is-hidden")
    const lengthUnits = line.lengthUnits
    const distance = Math.max(0, Math.min(lengthUnits, distanceUnits))
    const spread = cloudSpread(distance, lengthUnits, this.#trailUnits)
    setPosition(this.#head, line.pointAt(distance))
    for (const { element, spec } of this.#particles) {
      const at = line.pointAt(Math.max(0, distance - spec.lagUnits * spread))
      const swayAngle = distance * SWAY_PER_UNIT + spec.phase
      setPosition(element, {
        x: at.x + (spec.dx + Math.sin(swayAngle) * SWAY_UNITS) * spread,
        y: at.y + (spec.dy + Math.cos(swayAngle) * SWAY_UNITS * 0.6) * spread,
      })
    }
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
  element.setAttribute("transform", `translate(${round(point.x)} ${round(point.y)})`)
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
