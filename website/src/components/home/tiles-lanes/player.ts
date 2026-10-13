// Plays the tiles-and-lanes scene in the browser: beat 1, 2 and 3, each handed to the next by a
// camera move, forever. The frame loop runs only while 30% of the scene is on screen in a visible
// tab. Under reduced motion nothing moves and the beats show their settled frames side by side, as
// without JavaScript. The pills jump to a beat, which then plays twice before the loop moves on.

import { clamp, easeInOut } from "../../../lib/figure/math"
import { lerpCamera, lerpView, type Camera, type View } from "./camera"
import type { SceneData } from "./data"
import { mixedBeat } from "./mixed"
import { StageCanvas } from "./stage-canvas"
import {
  lanesBeat,
  tilesBeat,
  VIEW_HEIGHT_UNITS,
  VIEW_WIDTH_UNITS,
  type Beat,
  type BeatName,
  type ParticleState,
  type PathState,
} from "./scene"

const VISIBLE_RATIO = 0.3
const MAX_FRAME_DELTA_MS = 50
const TRANSITION_MS = 1500
const REPLAY_MS = 700
const ORDER: BeatName[] = ["tiles", "lanes", "mixed"]

interface Stage {
  beat: Beat
  element: HTMLElement
  paths: Map<string, SVGPathElement>
  particles: SVGCircleElement[]
  glows: SVGCircleElement[]
  canvas: StageCanvas | null
  // The paths a frame restyles, in paint order, and what they show; drawn on the canvas while live.
  moving: { key: string; element: SVGPathElement }[]
  movingKeys: Set<string>
  drawn: Map<string, PathState>
  labels: Map<string, HTMLElement>
  lines: HTMLElement[]
  marks: Record<string, number>
  metas: HTMLElement[]
  tab: HTMLButtonElement | null
  cameraKey: string
  line: number
  metaIndex: number
}

interface Transition {
  from: BeatName
  to: BeatName
  elapsedMs: number
}

export class TilesLanesPlayer {
  #root: HTMLElement
  #stages: Record<BeatName, Stage>
  #reducedMotion = matchMedia("(prefers-reduced-motion: reduce)")
  #current: BeatName = "tiles"
  #timeMs = 0
  #transition: Transition | null = null
  // Extra runs of the current beat before the loop moves on, after a pill was pressed.
  #replays = 0
  #isInView = false
  #frameId = 0
  #lastFrameMs: number | null = null

  constructor(root: HTMLElement, data: SceneData) {
    this.#root = root
    this.#stages = {
      tiles: this.#stage(tilesBeat(data.tiles), data.tilesCode.marks),
      lanes: this.#stage(lanesBeat(data.lanes), data.lanesCode.marks),
      mixed: this.#stage(mixedBeat(data.mixed), data.mixedCode.marks),
    }
  }

  start(): void {
    this.#reducedMotion.addEventListener("change", () => this.#applyMotionPreference())
    document.addEventListener("visibilitychange", () => this.#schedule())
    for (const name of ORDER) {
      this.#stages[name].tab?.addEventListener("click", () => this.#jump(name))
    }
    new IntersectionObserver(
      entries => {
        const entry = entries[entries.length - 1]
        this.#isInView = entry.isIntersecting && entry.intersectionRatio >= VISIBLE_RATIO - 0.001
        this.#schedule()
      },
      { threshold: [0, VISIBLE_RATIO] }
    ).observe(this.#root)
    this.#applyMotionPreference()
  }

  #stage(beat: Beat, marks: Record<string, number>): Stage {
    const element = this.#root.querySelector<HTMLElement>(`.tl-beat[data-beat="${beat.name}"]`)
    if (!element) throw new Error(`no stage for ${beat.name}`)
    const particles = [...element.querySelectorAll<SVGCircleElement>(".tl-p")]
    const glows = [...element.querySelectorAll<SVGCircleElement>(".tl-glow")]
    const canvas = element.querySelector<HTMLCanvasElement>(".tl-canvas")
    const keyed = <T extends Element>(selector: string) =>
      new Map([...element.querySelectorAll<T & HTMLElement>(selector)].map(el => [el.dataset.key ?? "", el as T]))
    const moving = [...element.querySelectorAll<SVGPathElement>(".tl-svg-moving path[data-key]")].map(path => ({
      key: path.dataset.key ?? "",
      element: path,
    }))
    return {
      beat,
      element,
      paths: keyed<SVGPathElement>("path[data-key]"),
      particles,
      glows,
      canvas: canvas ? new StageCanvas(canvas, particles, glows) : null,
      moving,
      movingKeys: new Set(moving.map(path => path.key)),
      drawn: new Map(),
      labels: keyed<HTMLElement>(".tl-label[data-key]"),
      lines: [...element.querySelectorAll<HTMLElement>(".tl-line")],
      marks,
      metas: [...element.querySelectorAll<HTMLElement>("[data-meta]")],
      tab: this.#root.querySelector<HTMLButtonElement>(`.tl-tab[data-beat="${beat.name}"]`),
      cameraKey: "",
      line: -2,
      metaIndex: -2,
    }
  }

  get #isLive(): boolean {
    return !this.#reducedMotion.matches
  }

  #applyMotionPreference(): void {
    // Live, the moving paths are drawn on the canvas; settled, in the SVG: either way from scratch.
    for (const stage of Object.values(this.#stages)) stage.cameraKey = ""
    if (this.#isLive) {
      this.#root.setAttribute("data-live", "")
      this.#current = "tiles"
      this.#timeMs = 0
      this.#transition = null
      this.#replays = 0
      this.#showActive("tiles", true)
      this.#renderPlaying()
    } else {
      this.#root.removeAttribute("data-live")
      this.#stop()
      for (const stage of Object.values(this.#stages)) {
        const { camera, view } = stage.beat.shot(stage.beat.settledMs)
        this.#render(stage, stage.beat.settledMs, camera, view, 1)
      }
    }
    this.#schedule()
  }

  #jump(name: BeatName): void {
    if (!this.#isLive) return
    this.#replays = 1
    if (this.#transition) {
      if (this.#transition.to === name) return
      this.#current = this.#transition.to
      this.#transition = null
    }
    this.#transition = { from: this.#current, to: name, elapsedMs: 0 }
    this.#showActive(name)
    this.#schedule()
  }

  #schedule(): void {
    const shouldRun = this.#isLive && this.#isInView && document.visibilityState === "visible"
    if (shouldRun && this.#frameId === 0) {
      this.#lastFrameMs = null
      this.#frameId = requestAnimationFrame(now => this.#onFrame(now))
    } else if (!shouldRun) {
      this.#stop()
    }
  }

  #stop(): void {
    cancelAnimationFrame(this.#frameId)
    this.#frameId = 0
  }

  #onFrame(nowMs: number): void {
    this.#frameId = 0
    const dtMs = this.#lastFrameMs === null ? 0 : clamp(nowMs - this.#lastFrameMs, 0, MAX_FRAME_DELTA_MS)
    this.#lastFrameMs = nowMs
    this.#advance(dtMs)
    this.#renderPlaying()
    this.#frameId = requestAnimationFrame(now => this.#onFrame(now))
  }

  #advance(dtMs: number): void {
    const transition = this.#transition
    if (transition) {
      transition.elapsedMs += dtMs
      // The code and the pills change over halfway, when the incoming beat starts to show.
      if (transition.elapsedMs >= this.#transitionMs(transition) / 2) this.#showActive(transition.to)
      if (transition.elapsedMs >= this.#transitionMs(transition)) {
        this.#current = transition.to
        this.#timeMs = 0
        this.#transition = null
      }
      return
    }
    this.#timeMs += dtMs
    if (this.#timeMs < this.#stages[this.#current].beat.durationMs) return
    const next = this.#replays > 0 ? this.#current : ORDER[(ORDER.indexOf(this.#current) + 1) % ORDER.length]
    this.#replays = Math.max(0, this.#replays - 1)
    this.#transition = { from: this.#current, to: next, elapsedMs: 0 }
  }

  #transitionMs(transition: Transition): number {
    return transition.from === transition.to ? REPLAY_MS : TRANSITION_MS
  }

  #renderPlaying(): void {
    const transition = this.#transition
    if (!transition) {
      const stage = this.#stages[this.#current]
      const { camera, view } = stage.beat.shot(this.#timeMs)
      this.#render(stage, this.#timeMs, camera, view, 1)
      for (const other of Object.values(this.#stages)) if (other !== stage) this.#setOpacity(other, 0)
      this.#setProgress(stage, this.#timeMs / stage.beat.durationMs)
      return
    }
    for (const other of Object.values(this.#stages))
      if (other.beat.name !== transition.from && other.beat.name !== transition.to) this.#setOpacity(other, 0)
    const p = clamp(transition.elapsedMs / this.#transitionMs(transition), 0, 1)
    const from = this.#stages[transition.from]
    const to = this.#stages[transition.to]
    if (from === to) {
      // A replay: fade out on the end, fade in on the start, the camera where each leaves it.
      const end = from.beat.shot(from.beat.durationMs)
      const start = from.beat.shot(0)
      if (p < 0.5) this.#render(from, from.beat.durationMs, end.camera, end.view, 1 - easeInOut(p * 2))
      else this.#render(to, 0, start.camera, start.view, easeInOut(p * 2 - 1))
      return
    }
    // The outgoing beat moves its camera toward its exit as it fades; the incoming one arrives from
    // its entrance, so each move reads as one: into the accumulator and out at the warp, out over the
    // CTA's floors, and down through a warp into the tiles again.
    const eased = easeInOut(p)
    const outgoing = 1 - easeInOut(clamp(p / 0.6, 0, 1))
    const incoming = easeInOut(clamp((p - 0.4) / 0.6, 0, 1))
    const leaving = from.beat.shot(from.beat.durationMs)
    const arriving = to.beat.shot(0)
    this.#render(
      from,
      from.beat.durationMs,
      lerpCamera(leaving.camera, from.beat.exit.camera, eased),
      lerpView(leaving.view, from.beat.exit.view, eased),
      outgoing
    )
    this.#render(
      to,
      0,
      lerpCamera(to.beat.enter.camera, arriving.camera, eased),
      lerpView(to.beat.enter.view, arriving.view, eased),
      incoming
    )
    this.#setProgress(to, 0)
  }

  #showActive(name: BeatName, force = false): void {
    if (!force && this.#root.dataset.active === name) return
    this.#root.dataset.active = name
    for (const stage of Object.values(this.#stages)) {
      const isActive = stage.beat.name === name
      stage.element.classList.toggle("is-active", isActive)
      stage.tab?.setAttribute("aria-pressed", String(isActive))
      if (!isActive) stage.tab?.style.setProperty("--progress", "0")
    }
  }

  #setOpacity(stage: Stage, opacity: number): void {
    const value = String(Math.round(opacity * 1000) / 1000)
    if (stage.element.style.getPropertyValue("--stage-opacity") !== value)
      stage.element.style.setProperty("--stage-opacity", value)
  }

  // In hundredths, so the pill's hairline is restyled a few times a second rather than every frame.
  #setProgress(stage: Stage, fraction: number): void {
    const value = String(Math.round(clamp(fraction, 0, 1) * 100) / 100)
    if (stage.tab && stage.tab.style.getPropertyValue("--progress") !== value)
      stage.tab.style.setProperty("--progress", value)
  }

  #applyPaths(stage: Stage, paths: Record<string, PathState>): void {
    for (const [key, state] of Object.entries(paths)) {
      if (this.#isLive && stage.movingKeys.has(key)) {
        const before = stage.drawn.get(key)
        stage.drawn.set(key, { ...before, ...state, d: state.d ?? before?.d })
        continue
      }
      const path = stage.paths.get(key)
      if (!path) continue
      if (state.d !== undefined) setAttribute(path, "d", state.d)
      if (state.width !== undefined) setAttribute(path, "stroke-width", String(state.width))
      const opacity = String(state.opacity)
      if (path.style.opacity !== opacity) path.style.opacity = opacity
    }
  }

  #render(stage: Stage, tMs: number, camera: Camera, view: View, opacity: number): void {
    this.#setOpacity(stage, opacity)
    if (opacity <= 0) return
    const cameraKey = JSON.stringify([camera, view])
    if (cameraKey !== stage.cameraKey) {
      stage.cameraKey = cameraKey
      this.#applyPaths(stage, stage.beat.still(camera, view))
    }
    const frame = stage.beat.frame(tMs, camera, view)
    this.#applyPaths(stage, frame.paths)
    if (this.#isLive && stage.canvas) {
      const paths = stage.moving.map(({ key, element }) => ({ element, state: stage.drawn.get(key) }))
      stage.canvas.draw(paths, frame.particles, frame.glows)
    } else {
      applyCircles(stage.glows, frame.glows)
      applyCircles(stage.particles, frame.particles)
    }
    for (const [key, state] of Object.entries(frame.labels)) {
      const label = stage.labels.get(key)
      if (!label) continue
      const left = `${Math.round((state.x / VIEW_WIDTH_UNITS) * 10000) / 100}%`
      const top = `${Math.round((state.y / VIEW_HEIGHT_UNITS) * 10000) / 100}%`
      if (label.style.left !== left) label.style.left = left
      if (label.style.top !== top) label.style.top = top
    }
    const line = frame.mark === null ? -1 : (stage.marks[frame.mark] ?? -1)
    if (line !== stage.line) {
      stage.line = line
      stage.lines.forEach((element, index) => element.classList.toggle("is-current", index === line))
    }
    if (frame.metaIndex !== stage.metaIndex) {
      stage.metaIndex = frame.metaIndex
      for (const meta of stage.metas) {
        meta.classList.toggle("is-current", Number(meta.dataset.meta) === frame.metaIndex)
        meta.classList.toggle("is-done", Number(meta.dataset.meta) < frame.metaIndex)
      }
    }
  }
}

// Writes only what changed: an unchanged attribute written again still costs a style pass.
function setAttribute(element: Element, name: string, value: string): void {
  if (element.getAttribute(name) !== value) element.setAttribute(name, value)
}

function applyCircles(circles: SVGCircleElement[], states: ParticleState[]): void {
  states.forEach((state, index) => {
    const circle = circles[index]
    if (!circle) return
    if (state.opacity <= 0) {
      setAttribute(circle, "opacity", "0")
      return
    }
    setAttribute(circle, "cx", String(state.x))
    setAttribute(circle, "cy", String(state.y))
    setAttribute(circle, "r", String(state.r))
    setAttribute(circle, "opacity", String(state.opacity))
  })
}
