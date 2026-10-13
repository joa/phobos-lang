// Small numeric helpers shared by every figure. Pure, so they run in Node and in tests.

// Mulberry32 (Tommy Ettinger): a tiny 32-bit PRNG, so a seeded figure draws the same on every load.
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let mixed = Math.imul(state ^ (state >>> 15), state | 1)
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61)
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296
  }
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function lerp(from: number, to: number, t: number): number {
  return from + (to - from) * t
}

export type Easing = (t: number) => number

export const linear: Easing = t => t

export function easeInOutSine(t: number): number {
  return (1 - Math.cos(Math.PI * t)) / 2
}

export function easeOutCubic(t: number): number {
  return 1 - (1 - t) ** 3
}

// The CSS cubic-bezier() timing function, so a frame-loop animation eases exactly like the CSS tokens.
// Solves x(s) = t for the curve parameter s by Newton's method, falling back to bisection.
export function cubicBezier(x1: number, y1: number, x2: number, y2: number): Easing {
  const cx = 3 * x1
  const bx = 3 * (x2 - x1) - cx
  const ax = 1 - cx - bx
  const cy = 3 * y1
  const by = 3 * (y2 - y1) - cy
  const ay = 1 - cy - by
  const sampleX = (s: number) => ((ax * s + bx) * s + cx) * s
  const sampleY = (s: number) => ((ay * s + by) * s + cy) * s
  const slopeX = (s: number) => (3 * ax * s + 2 * bx) * s + cx

  const solve = (x: number) => {
    let s = x
    for (let iteration = 0; iteration < 8; iteration += 1) {
      const error = sampleX(s) - x
      if (Math.abs(error) < 1e-7) return s
      const slope = slopeX(s)
      if (Math.abs(slope) < 1e-6) break
      s -= error / slope
    }
    let low = 0
    let high = 1
    s = x
    while (high - low > 1e-7) {
      if (sampleX(s) < x) low = s
      else high = s
      s = (low + high) / 2
    }
    return s
  }

  return t => {
    if (t <= 0) return 0
    if (t >= 1) return 1
    return sampleY(solve(t))
  }
}

// The two easing tokens of tokens.css: --ease-out and --ease-in-out.
export const easeOut: Easing = cubicBezier(0.16, 1, 0.3, 1)
export const easeInOut: Easing = cubicBezier(0.65, 0, 0.35, 1)
