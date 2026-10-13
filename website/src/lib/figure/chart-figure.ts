// The figure base. Lifecycle: start() reads the state from the markup, builds the current layout and
// settles it. The frame loop runs only while the figure wants to play and at least 30% of it is on
// screen in a visible tab. Subclasses override build, update, settle, the on* hooks and optionally
// get tour(). Nothing here touches the DOM at module level, so figure modules import in Node.
//
// Figures loop: once update() and the tour have finished, the end state holds for LOOP_HOLD_MS, then
// cross-fades into a fresh run started by replay(). Under reduced motion nothing loops and the figure
// keeps its settled end state.

import { clamp } from "./math"
import { layoutFor, type Layout } from "./layout"
import { Motion } from "./motion"
import { svgEl, withElementFactory } from "./svg"
import { StaticElement, staticFactory } from "./static"
import { LOOP_FADE_MS, LoopClock } from "./timeline"

const VISIBLE_RATIO = 0.3
const MAX_FRAME_DELTA_MS = 50
const TOUR_HOLD_MS = 1400

export type StateValue = string | number
export type FigureState = Readonly<Record<string, StateValue>>

export interface StaticRender {
  viewBox: string
  drawing: string
  readout: string
}

export type FigureConstructor = new (root: Element) => ChartFigure

// A range's track fills up to its thumb (--fill) and, with few enough values, is cut into one segment
// per value (--ticks, data-ticked); chart.css draws both.
const MAX_TICKS = 40

export function syncRangeFill(input: HTMLInputElement): void {
  const min = Number(input.min || 0)
  const max = Number(input.max || 100)
  const span = max - min
  const fraction = span > 0 ? (Number(input.value) - min) / span : 0
  input.style.setProperty("--fill", `${Math.round(clamp(fraction, 0, 1) * 1000) / 10}%`)
  const step = Number(input.step || 1)
  const ticks = step > 0 ? Math.round(span / step) : 0
  const isTicked = ticks > 1 && ticks <= MAX_TICKS
  input.toggleAttribute("data-ticked", isTicked)
  if (isTicked) input.style.setProperty("--ticks", String(ticks))
  else input.style.removeProperty("--ticks")
}

export class ChartFigure {
  #root: Element
  #svg: SVGSVGElement
  #readout: Element | null
  #drawing: SVGGElement | null = null
  #layout: Layout | null = null
  #state: FigureState = Object.freeze({})
  #motion: Motion
  #reducedMotionQuery: MediaQueryList | null = null
  #isInView = false
  #isTabVisible = true
  #hasBeenVisible = false
  #hasInteracted = false
  #isPlaying = false
  #isInFrame = false
  #frameId = 0
  #lastFrameMs: number | null = null
  #rebuildFrameId = 0
  #activeTour: { patches: Record<string, StateValue>[]; index: number; heldMs: number } | null = null
  #loopClock = new LoopClock()
  #ghost: SVGGElement | null = null

  constructor(root: Element) {
    this.#root = root
    const svg = root.querySelector(".chart-svg")
    if (!svg) throw new Error(`#${root.id} has no .chart-svg`)
    this.#svg = svg as SVGSVGElement
    this.#readout = root.querySelector(".chart-readout")
    this.#motion = new Motion({ isReduced: () => this.reducedMotion })
  }

  // Runs build() and settle() for one layout and state without a browser, for the page's no-script
  // end state. id prefixes the ids inside the drawing and must be unique on the page. See static.ts
  // for what build() and settle() may use.
  static renderStatic(
    this: FigureConstructor,
    layout: Layout,
    state: Record<string, StateValue>,
    id = "static"
  ): StaticRender {
    return withElementFactory(staticFactory, () => {
      const root = new StaticElement("figure")
      root.setAttribute("id", id)
      const svg = new StaticElement("svg")
      svg.classList.add("chart-svg")
      const readout = new StaticElement("p")
      readout.classList.add("chart-readout")
      root.append(svg, readout)
      const figure = new this(root as unknown as Element)
      figure.#state = Object.freeze({ ...state })
      figure.#rebuild(layout)
      const drawing = figure.#drawing as unknown as StaticElement
      return {
        viewBox: svg.getAttribute("viewBox") ?? "0 0 0 0",
        drawing: drawing.innerHTML,
        readout: readout.textContent,
      }
    })
  }

  get root(): Element {
    return this.#root
  }

  // The root's id, unique on the page, for ids inside the drawing (gradients, patterns).
  get id(): string {
    return this.#root.id || "chart"
  }

  get svg(): SVGSVGElement {
    return this.#svg
  }

  get drawing(): SVGGElement {
    if (!this.#drawing) throw new Error("drawing is only available from build() on")
    return this.#drawing
  }

  get layout(): Layout {
    return this.#layout ?? "wide"
  }

  get state(): FigureState {
    return this.#state
  }

  get motion(): Motion {
    return this.#motion
  }

  get reducedMotion(): boolean {
    return this.#reducedMotionQuery?.matches ?? false
  }

  get isVisible(): boolean {
    return this.#isInView && this.#isTabVisible
  }

  get hasInteracted(): boolean {
    return this.#hasInteracted
  }

  get tour(): Record<string, StateValue>[] {
    return []
  }

  // Whether the figure plays again after it has finished. A figure that runs forever never finishes,
  // so it needs no opt-out; one whose replay would be wrong to repeat returns false.
  get loops(): boolean {
    return true
  }

  // Whether a loop's restart cross-fades from the end state. A figure whose end and start look the
  // same (only packets move) can skip the fade.
  get loopFades(): boolean {
    return true
  }

  start(): void {
    for (const stale of this.#root.querySelectorAll(".chart-static")) stale.remove()
    this.#state = Object.freeze(this.#readMarkupState())
    this.#syncControls(this.#state)
    this.#reducedMotionQuery = matchMedia("(prefers-reduced-motion: reduce)")
    this.#reducedMotionQuery.addEventListener("change", () => this.#onReducedMotionChange())
    this.#isTabVisible = document.visibilityState === "visible"
    document.addEventListener("visibilitychange", () => this.#onTabVisibilityChange())
    this.#root.addEventListener("click", event => this.#onClick(event))
    this.#root.addEventListener("input", event => this.#onInputEvent(event))
    this.#rebuild(layoutFor(this.#root.getBoundingClientRect().width))
    this.#root.setAttribute("data-live", "")
    new ResizeObserver(entries => this.#onResize(entries)).observe(this.#root)
    new IntersectionObserver(entries => this.#onIntersection(entries), {
      threshold: [0, VISIBLE_RATIO],
    }).observe(this.#root)
  }

  // Draws everything for this layout into the fresh drawing group, sets the viewBox and restores the
  // visuals of the current state.
  build(_layout: Layout): void {}

  // Advances animations; returns true to keep the frame loop alive.
  update(_dtMs: number): boolean {
    return false
  }

  // Jumps to the end state of whatever is playing.
  settle(): void {}

  onChange(_group: string, _value: string): void {}

  onAction(_action: string): void {}

  onInput(_param: string, _value: number): void {}

  onFirstVisible(): void {
    this.#startTour()
  }

  // Starts the next run of a loop: the tour again while the reader has not taken over, else the
  // figure's own replay action.
  replay(): void {
    if (this.tour.length > 0 && !this.#hasInteracted) this.#startTour()
    else this.onAction("replay")
  }

  // Cross-fades from what the drawing shows now to whatever it shows next: a copy of the drawing is
  // laid over it and fades out while the drawing fades in. The two blend with plus-lighter inside the
  // isolated svg, so what both show stays at full strength and only what changes fades.
  crossfade({ durationMs = LOOP_FADE_MS }: { durationMs?: number } = {}): void {
    const drawing = this.#drawing
    if (!drawing || this.reducedMotion || typeof drawing.animate !== "function") return
    this.#dropGhost()
    const ghost = drawing.cloneNode(true) as SVGGElement
    ghost.setAttribute("class", "chart-drawing chart-ghost")
    ghost.setAttribute("aria-hidden", "true")
    // The copy is a picture: no ids, no focus stops, nothing a click handler would take for a control.
    for (const element of ghost.querySelectorAll("[id], [tabindex], [data-group], [data-value]")) {
      for (const name of ["id", "tabindex", "data-group", "data-value", "role", "aria-label"])
        element.removeAttribute(name)
    }
    drawing.after(ghost)
    this.#ghost = ghost
    const easing = this.#motion.tokens.easeInOut
    const fadeOut = this.#motion.animate(ghost, [{ opacity: 1 }, { opacity: 0 }], {
      duration: durationMs,
      easing,
      fill: "forwards",
    })
    this.#motion.animate(drawing, [{ opacity: 0 }, { opacity: 1 }], { duration: durationMs, easing })
    if (!fadeOut) {
      this.#dropGhost()
      return
    }
    const drop = () => this.#dropGhost(ghost)
    fadeOut.addEventListener("finish", drop)
    fadeOut.addEventListener("cancel", drop)
  }

  play(): void {
    if (this.reducedMotion) {
      this.settle()
      return
    }
    this.#isPlaying = true
    this.#startLoop()
  }

  pause(): void {
    this.#isPlaying = false
    this.#stopLoop()
  }

  setReadout(text: string): void {
    if (this.#readout && this.#readout.textContent !== text) this.#readout.textContent = text
  }

  setViewBox(widthUnits: number, heightUnits: number): void {
    this.#svg.setAttribute("viewBox", `0 0 ${widthUnits} ${heightUnits}`)
  }

  // Programmatic state change: updates state and the controls, calls no hook.
  setState(patch: Record<string, StateValue>): void {
    this.#state = Object.freeze({ ...this.#state, ...patch })
    this.#syncControls(patch)
  }

  #readMarkupState(): Record<string, StateValue> {
    const state: Record<string, StateValue> = {}
    for (const button of this.#root.querySelectorAll<HTMLElement>("[data-group][data-value]")) {
      const { group, value } = button.dataset
      if (group === undefined || value === undefined) continue
      if (!Object.hasOwn(state, group) || button.getAttribute("aria-pressed") === "true") state[group] = value
    }
    for (const input of this.#root.querySelectorAll<HTMLInputElement>("input[data-param]")) {
      if (input.dataset.param) state[input.dataset.param] = Number(input.value)
      if (input.type === "range") syncRangeFill(input)
    }
    return state
  }

  #syncControls(patch: Record<string, StateValue>): void {
    for (const button of this.#root.querySelectorAll<HTMLElement>("[data-group][data-value]")) {
      const { group, value } = button.dataset
      if (group !== undefined && Object.hasOwn(patch, group))
        button.setAttribute("aria-pressed", String(value === patch[group]))
    }
    for (const input of this.#root.querySelectorAll<HTMLInputElement>("input[data-param]")) {
      const param = input.dataset.param
      if (param !== undefined && Object.hasOwn(patch, param) && Number(input.value) !== patch[param]) {
        input.value = String(patch[param])
        if (input.type === "range") syncRangeFill(input)
      }
    }
  }

  // Tour patches and clicks go through here, so a patch fires its hooks even when a value is unchanged.
  #applyPatch(patch: Record<string, StateValue>): void {
    this.setState(patch)
    for (const [key, value] of Object.entries(patch)) {
      if (typeof value === "number") this.onInput(key, value)
      else this.onChange(key, value)
    }
  }

  #onClick(event: Event): void {
    const target = event.target as Element | null
    const control = target?.closest<HTMLButtonElement>("[data-group][data-value], [data-action]")
    if (!control || !this.#root.contains(control) || control.disabled) return
    this.#noteReaderInput()
    const { action, group, value } = control.dataset
    if (action) this.onAction(action)
    else if (group !== undefined && value !== undefined) this.#applyPatch({ [group]: value })
  }

  #onInputEvent(event: Event): void {
    const input = event.target as HTMLInputElement
    if (!input.matches("input[data-param]") || !input.dataset.param) return
    this.#noteReaderInput()
    if (input.type === "range") syncRangeFill(input)
    const value = Number(input.value)
    this.#state = Object.freeze({ ...this.#state, [input.dataset.param]: value })
    this.onInput(input.dataset.param, value)
  }

  #noteReaderInput(): void {
    this.#hasInteracted = true
    this.#activeTour = null
  }

  #onResize(entries: ResizeObserverEntry[]): void {
    const widthPx = entries[entries.length - 1].contentRect.width
    if (widthPx === 0) return
    cancelAnimationFrame(this.#rebuildFrameId)
    this.#rebuildFrameId = 0
    if (layoutFor(widthPx) === this.#layout) return
    // Rebuilding resizes the root; doing it inside this callback would trip the browser's
    // "ResizeObserver loop completed with undelivered notifications" error, so wait a frame. The width
    // is measured again then: a transient size (a full-page capture resizing the viewport) is often
    // gone by the next frame, and rebuilding for it would draw the figure twice.
    this.#rebuildFrameId = requestAnimationFrame(() => {
      this.#rebuildFrameId = 0
      const settledWidthPx = this.#root.getBoundingClientRect().width
      if (settledWidthPx === 0) return
      const layout = layoutFor(settledWidthPx)
      if (layout !== this.#layout) this.#rebuild(layout)
    })
  }

  #rebuild(layout: Layout): void {
    this.#motion.cancelAll()
    this.#dropGhost()
    this.#loopClock.reset()
    this.#layout = layout
    this.#root.setAttribute("data-layout", layout)
    this.#drawing?.remove()
    for (const stale of this.#svg.querySelectorAll(":scope > .chart-drawing")) stale.remove()
    this.#drawing = svgEl("g", { class: "chart-drawing" }, this.#svg)
    this.build(layout)
    this.settle()
  }

  #onIntersection(entries: IntersectionObserverEntry[]): void {
    const entry = entries[entries.length - 1]
    // The ratio reported at the threshold crossing can round to just below it.
    this.#isInView = entry.isIntersecting && entry.intersectionRatio >= VISIBLE_RATIO - 0.001
    this.#onVisibilityChange()
  }

  #onTabVisibilityChange(): void {
    this.#isTabVisible = document.visibilityState === "visible"
    this.#onVisibilityChange()
  }

  #onVisibilityChange(): void {
    // CSS pauses breathing and sweeps while the figure is off screen.
    this.#root.toggleAttribute("data-visible", this.isVisible)
    if (!this.isVisible) {
      this.#stopLoop()
      return
    }
    if (!this.#hasBeenVisible) {
      this.#hasBeenVisible = true
      this.onFirstVisible()
    }
    this.#startLoop()
  }

  #onReducedMotionChange(): void {
    if (!this.reducedMotion) return
    this.#activeTour = null
    this.pause()
    this.#loopClock.reset()
    this.#motion.finishAll()
    this.#dropGhost()
    this.settle()
  }

  #dropGhost(ghost: SVGGElement | null = this.#ghost): void {
    if (!ghost) return
    ghost.remove()
    if (ghost === this.#ghost) this.#ghost = null
  }

  #startLoop(): void {
    if (this.#frameId !== 0 || this.#isInFrame || !this.#isPlaying || !this.isVisible) return
    this.#lastFrameMs = null
    this.#frameId = requestAnimationFrame(nowMs => this.#onFrame(nowMs))
  }

  #stopLoop(): void {
    cancelAnimationFrame(this.#frameId)
    this.#frameId = 0
  }

  #onFrame(nowMs: number): void {
    this.#frameId = 0
    const dtMs = this.#lastFrameMs === null ? 0 : clamp(nowMs - this.#lastFrameMs, 0, MAX_FRAME_DELTA_MS)
    this.#lastFrameMs = nowMs
    // Re-armed below by update() or the tour, or by a play() call made during this frame.
    this.#isPlaying = false
    this.#isInFrame = true
    try {
      const isRunning = this.update(dtMs) || (this.#activeTour !== null && this.#advanceTour(dtMs))
      if (isRunning) this.#isPlaying = true
      if (this.loops && !this.reducedMotion) {
        const tick = this.#loopClock.tick(isRunning, dtMs)
        if (tick === "hold") this.#isPlaying = true
        else if (tick === "restart") this.#restartLoop()
      }
    } finally {
      this.#isInFrame = false
    }
    if (this.#isPlaying && this.isVisible) this.#frameId = requestAnimationFrame(next => this.#onFrame(next))
  }

  // Called inside a frame; replay() re-arms the loop through play() when it starts something. When it
  // starts nothing, the copy laid over the drawing goes at once, since the two are the same.
  #restartLoop(): void {
    if (this.loopFades) this.crossfade()
    this.replay()
    if (!this.#isPlaying) {
      this.#motion.finishAll()
      this.#dropGhost()
    }
  }

  #startTour(): void {
    const patches = this.tour
    if (patches.length === 0 || this.reducedMotion || this.#hasInteracted) return
    this.#activeTour = { patches, index: 0, heldMs: 0 }
    this.#applyPatch(patches[0])
    this.#isPlaying = true
    this.#startLoop()
  }

  // Runs once the current patch's animation has finished: hold, then apply the next patch.
  #advanceTour(dtMs: number): boolean {
    const tour = this.#activeTour
    if (!tour) return false
    if (tour.index === tour.patches.length - 1) {
      this.#activeTour = null
      return false
    }
    tour.heldMs += dtMs
    if (tour.heldMs < TOUR_HOLD_MS) return true
    tour.index += 1
    tour.heldMs = 0
    this.#applyPatch(tour.patches[tour.index])
    return true
  }
}
