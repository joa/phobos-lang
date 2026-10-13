// The station rail: seven stations on a ring seen from above and in front, the current one nearest the
// reader at the bottom. The journey turns the ring one station a move and the last station leads back
// to the first, so the rail shows the loop. Positions are percentages of the rail's box. Pure.

export const RING_CENTER = { x: 50, y: 46 }
export const RING_RADIUS = { x: 41, y: 34 }

export interface RingSpot {
  /** Percent of the rail's width and height. */
  x: number
  y: number
  /** 1 at the front, 0 at the back. */
  depth: number
}

/**
 * Where station `index` of `count` sits when the ring has turned `turn` stations (fractional during a
 * move): the station equal to `turn` is at the front.
 */
export function ringSpot(index: number, count: number, turn: number): RingSpot {
  const angle = ((index - turn) / count) * Math.PI * 2
  const x = RING_CENTER.x + Math.sin(angle) * RING_RADIUS.x
  const y = RING_CENTER.y + Math.cos(angle) * RING_RADIUS.y
  return { x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100, depth: Math.round(((Math.cos(angle) + 1) / 2) * 1000) / 1000 }
}

/** The turn between two stations at a move's progress, the short way round. */
export function ringTurn(from: number, to: number, count: number, progress: number): number {
  let delta = (to - from) % count
  if (delta > count / 2) delta -= count
  if (delta < -count / 2) delta += count
  return from + delta * progress
}

/** The ring's outline in a 100 by 100 box: the back half and the front half, for two strokes. */
export function ringPaths(): { back: string; front: string } {
  const { x: cx, y: cy } = RING_CENTER
  const { x: rx, y: ry } = RING_RADIUS
  return {
    back: `M${cx - rx} ${cy}A${rx} ${ry} 0 0 1 ${cx + rx} ${cy}`,
    front: `M${cx + rx} ${cy}A${rx} ${ry} 0 0 1 ${cx - rx} ${cy}`,
  }
}
