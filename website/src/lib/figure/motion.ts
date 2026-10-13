// The Motion techniques of the site as reusable helpers. Every helper honors prefers-reduced-motion by
// landing on the end state at once, and none writes a color: they animate opacity, transforms, blur
// and stroke dashes, and CSS classes carry the rest.
//
// - drawIn: hairlines draw themselves (stroke-dashoffset over the measured length).
// - enter: staggered entrances from opacity 0, 6 px down and blur(6px).
// - glide: FLIP, for SVG and HTML elements; viewTransition() for HTML layout swaps.
// - pulse: the soft glow on arrival, fading back over 300 ms.
// - breathe and sweep are pure CSS (.is-breathing, .chart-sweep, .sweep-line in chart.css), paused
//   while their figure is off screen; setBreathing() and sweepWire() only add the classes or element.
// - Star packets with comet trails live in packet.ts; they move in the figure's frame loop.

import type { Space } from "./draw"
import { svgEl } from "./svg"

export interface MotionTokens {
  dur1Ms: number
  dur2Ms: number
  dur3Ms: number
  dur4Ms: number
  staggerMs: number
  easeOut: string
  easeInOut: string
}

const DEFAULT_TOKENS: MotionTokens = {
  dur1Ms: 120,
  dur2Ms: 240,
  dur3Ms: 480,
  dur4Ms: 900,
  staggerMs: 30,
  easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
  easeInOut: "cubic-bezier(0.65, 0, 0.35, 1)",
}

export const ARRIVAL_GLOW_MS = 300

export function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches
}

// Parses "240ms" or "0.24s" into milliseconds.
export function parseDurationMs(value: string, fallbackMs: number): number {
  const match = /^\s*(-?[\d.]+)\s*(ms|s)\s*$/.exec(value)
  if (!match) return fallbackMs
  const amount = Number(match[1])
  return match[2] === "s" ? amount * 1000 : amount
}

// The motion tokens as tokens.css defines them right now (they shrink under reduced motion).
export function readMotionTokens(): MotionTokens {
  if (typeof document === "undefined") return DEFAULT_TOKENS
  const style = getComputedStyle(document.documentElement)
  const read = (name: string) => style.getPropertyValue(name).trim()
  return {
    dur1Ms: parseDurationMs(read("--dur-1"), DEFAULT_TOKENS.dur1Ms),
    dur2Ms: parseDurationMs(read("--dur-2"), DEFAULT_TOKENS.dur2Ms),
    dur3Ms: parseDurationMs(read("--dur-3"), DEFAULT_TOKENS.dur3Ms),
    dur4Ms: parseDurationMs(read("--dur-4"), DEFAULT_TOKENS.dur4Ms),
    staggerMs: parseDurationMs(read("--stagger"), DEFAULT_TOKENS.staggerMs),
    easeOut: read("--ease-out") || DEFAULT_TOKENS.easeOut,
    easeInOut: read("--ease-in-out") || DEFAULT_TOKENS.easeInOut,
  }
}

interface StaggerOptions {
  durationMs?: number
  staggerMs?: number
  delayMs?: number
}

// Owns the WAAPI animations of one figure, so settle() can finish them and a rebuild can drop them.
export class Motion {
  #animations = new Set<Animation>()
  #glides = new WeakMap<Element, Animation>()
  #isReduced: () => boolean

  constructor({ isReduced = prefersReducedMotion }: { isReduced?: () => boolean } = {}) {
    this.#isReduced = isReduced
  }

  get isReduced(): boolean {
    return this.#isReduced()
  }

  get tokens(): MotionTokens {
    return readMotionTokens()
  }

  // Jumps every running animation to its end.
  finishAll(): void {
    for (const animation of [...this.#animations]) {
      try {
        animation.finish()
      } catch {
        animation.cancel()
      }
    }
    this.#animations.clear()
  }

  cancelAll(): void {
    for (const animation of this.#animations) animation.cancel()
    this.#animations.clear()
  }

  animate(element: Element, keyframes: Keyframe[], options: KeyframeAnimationOptions): Animation | null {
    if (this.isReduced || typeof element.animate !== "function") return null
    const animation = element.animate(keyframes, options)
    this.#animations.add(animation)
    const forget = () => this.#animations.delete(animation)
    animation.addEventListener("finish", forget)
    animation.addEventListener("cancel", forget)
    return animation
  }

  // Hairlines draw themselves from their start, one after another by the stagger.
  drawIn(elements: Iterable<SVGGeometryElement>, options: StaggerOptions = {}): void {
    const tokens = this.tokens
    const { durationMs = tokens.dur4Ms, staggerMs = tokens.staggerMs, delayMs = 0 } = options
    let index = 0
    for (const element of elements) {
      const lengthUnits = measureLength(element)
      if (lengthUnits > 0) {
        const dash = `${lengthUnits} ${lengthUnits}`
        this.animate(
          element,
          [
            { strokeDasharray: dash, strokeDashoffset: lengthUnits },
            { strokeDasharray: dash, strokeDashoffset: 0 },
          ],
          { duration: durationMs, delay: delayMs + index * staggerMs, easing: tokens.easeOut, fill: "backwards" }
        )
      }
      index += 1
    }
  }

  // Siblings arrive from opacity 0, 6 px below and blurred, one after another by the stagger.
  enter(elements: Iterable<Element>, options: StaggerOptions & { offsetPx?: number; blurPx?: number } = {}): void {
    const tokens = this.tokens
    const { durationMs = tokens.dur4Ms, staggerMs = tokens.staggerMs, delayMs = 0, offsetPx = 6, blurPx = 6 } = options
    let index = 0
    for (const element of elements) {
      this.animate(
        element,
        [
          { opacity: 0, translate: `0px ${offsetPx}px`, filter: `blur(${blurPx}px)` },
          { opacity: 1, translate: "0px 0px", filter: "blur(0px)" },
        ],
        { duration: durationMs, delay: delayMs + index * staggerMs, easing: tokens.easeOut, fill: "backwards" }
      )
      index += 1
    }
  }

  // FLIP: measure, let mutate() move the elements to where they belong, then glide each from where it
  // was. SVG deltas are converted from screen pixels to the parent's user units. A glide that starts
  // while another runs on the same element continues from where that one had got to.
  glide(
    elements: Element[],
    mutate: () => void,
    { durationMs = this.tokens.dur3Ms }: { durationMs?: number } = {}
  ): void {
    if (this.isReduced) {
      mutate()
      return
    }
    const before = elements.map(element => element.getBoundingClientRect())
    for (const element of elements) this.#glides.get(element)?.cancel()
    mutate()
    elements.forEach((element, index) => {
      const after = element.getBoundingClientRect()
      let dx = before[index].left - after.left
      let dy = before[index].top - after.top
      const parent = element.parentNode
      if (element instanceof SVGElement && parent instanceof SVGGraphicsElement) {
        const ctm = parent.getScreenCTM()
        if (ctm && ctm.a !== 0 && ctm.d !== 0) {
          dx /= ctm.a
          dy /= ctm.d
        }
      }
      if (Math.abs(dx) < 0.01 && Math.abs(dy) < 0.01) return
      const animation = this.animate(element, [{ translate: `${dx}px ${dy}px` }, { translate: "0px 0px" }], {
        duration: durationMs,
        easing: this.tokens.easeInOut,
      })
      if (animation) this.#glides.set(element, animation)
    })
  }

  // The glow on arrival: full, then back to nothing over 300 ms.
  pulse(glow: Element, { durationMs = ARRIVAL_GLOW_MS }: { durationMs?: number } = {}): void {
    this.animate(glow, [{ opacity: 1 }, { opacity: 0 }], { duration: durationMs, easing: "ease-out" })
  }
}

// A same-document view transition where the browser has one and motion is wanted, else a plain
// update. For HTML layout swaps only: Chrome does not capture SVG children as transition groups.
export function viewTransition(mutate: () => void, { isReduced = prefersReducedMotion() } = {}): void {
  const doc = document as Document & { startViewTransition?: (update: () => void) => unknown }
  if (doc.startViewTransition && !isReduced) doc.startViewTransition(mutate)
  else mutate()
}

// The current element breathes: glow opacity between 0.85 and 1 over 2.4 s, paused off screen.
export function setBreathing(element: Element, isBreathing: boolean): void {
  element.classList.toggle("is-breathing", isBreathing)
}

// A slow light sweep along a wire: a short bright dash traveling over a copy of the wire's path.
// Toggle "is-on" on the returned element to run it; CSS stops it off screen and under reduced motion.
export function sweepWire(parent: Element, pathData: string, space: Space = "global"): SVGPathElement {
  return svgEl("path", { class: `chart-sweep sp-${space}`, d: pathData, pathLength: 1 }, parent)
}

function measureLength(element: SVGGeometryElement): number {
  try {
    return element.getTotalLength()
  } catch {
    return 0
  }
}
