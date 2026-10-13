// The /inference/ hero's live scene: the layer stack, the orbit and the token stream on one 2D
// canvas, plus pointer parallax for the star layers. Loaded lazily after the page is idle. Runs only
// while the hero is on screen in a visible tab; under reduced motion it draws the opening frame once.

import { composeFrame, countsFor, inkOf, LEVELS, makeFrame, makeScene, VIEW_UNITS, type Frame, type Token } from "./model"
import { LABEL_OFFSET_UNITS, LABEL_UNITS, STATIC_TIME_S } from "./static-frame"

const MAX_DPR = 2
const MAX_STEP_MS = 50
const TILT_PITCH = 0.14
const TILT_YAW = 0.22
const TILT_EASE_MS = 420
// The smallest point the canvas draws, in CSS px.
const MIN_POINT_PX = 0.85
const TAU = Math.PI * 2

type Rgb = [number, number, number]
type Mode = "emit" | "stipple"

function parseRgb(value: string): Rgb {
  const numbers = value.match(/[\d.]+/g)?.map(Number) ?? [255, 255, 255]
  // color(srgb r g b) gives fractions; rgb() gives bytes.
  if (value.startsWith("color(")) return [numbers[0] * 255, numbers[1] * 255, numbers[2] * 255]
  return [numbers[0], numbers[1], numbers[2]]
}

const rgba = ([r, g, b]: Rgb, alpha: number) => `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`

export function start(root: HTMLElement): void {
  const canvas = root.querySelector<HTMLCanvasElement>(".inf-scene-canvas")
  const context = canvas?.getContext("2d", { alpha: true })
  if (!canvas || !context) return
  const ctx = context

  const scene = makeScene(countsFor(window.innerWidth))
  const frame: Frame = makeFrame(scene)
  const heads = new Int32Array(LEVELS)
  const next = new Int32Array(frame.x.length)
  const layers = [...root.querySelectorAll<HTMLElement>("[data-depth]")].map(element => ({
    element,
    depthPx: Number(element.dataset.depth) || 0,
  }))

  const reducedQuery = window.matchMedia("(prefers-reduced-motion: reduce)")
  let palette: string[] = []
  let glowRgb: Rgb = [0, 0, 0]
  let labelRgb: Rgb = [0, 0, 0]
  let labelFamily = ""
  let mode: Mode = "emit"
  let cssWidthPx = 0
  let unitsToPx = 1
  let timeS = STATIC_TIME_S
  let visible = false
  let frameId = 0
  let lastMs = 0
  const tilt = { pitch: 0, yaw: 0, targetPitch: 0, targetYaw: 0 }
  let rootRect: DOMRect | null = null

  const resolveTheme = () => {
    mode = document.documentElement.dataset.theme === "light" ? "stipple" : "emit"
    const probe = document.createElement("span")
    probe.style.display = "none"
    root.append(probe)
    const read = (token: Token | "dust"): Rgb => {
      probe.style.color = `var(--${token})`
      return parseRgb(getComputedStyle(probe).color)
    }
    const tokens: Record<Token, Rgb> = { brand: read("brand"), "brand-hi": read("brand-hi"), star: read("star"), nebula: read("nebula") }
    glowRgb = tokens["brand-hi"]
    labelRgb = mode === "emit" ? tokens["brand-hi"] : tokens.brand
    probe.style.fontFamily = "var(--font-mono)"
    labelFamily = getComputedStyle(probe).fontFamily
    probe.remove()
    palette = Array.from({ length: LEVELS }, (_, level) => {
      const ink = inkOf(level)
      const from = tokens[ink.from]
      const to = tokens[ink.to]
      const channel = (index: number) => from[index] + (to[index] - from[index]) * ink.mix
      // At dawn the faintest dots would vanish against the sky; lift them a little.
      const alpha = mode === "stipple" ? Math.min(1, ink.alpha * 0.85 + 0.08) : ink.alpha
      return rgba([channel(0), channel(1), channel(2)], alpha)
    })
  }

  const resize = () => {
    const rect = canvas.getBoundingClientRect()
    if (rect.width === 0) return
    cssWidthPx = rect.width
    const dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1)
    canvas.width = Math.round(rect.width * dpr)
    canvas.height = Math.round(rect.height * dpr)
    unitsToPx = (rect.width * dpr) / VIEW_UNITS
  }

  const draw = () => {
    composeFrame(scene, { timeS, pitch: tilt.pitch, yaw: tilt.yaw }, frame)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.setTransform(unitsToPx, 0, 0, unitsToPx, 0, 0)
    drawGlows()
    ctx.globalCompositeOperation = mode === "emit" ? "lighter" : "source-over"
    drawPoints((MIN_POINT_PX * VIEW_UNITS) / Math.max(1, cssWidthPx))
    ctx.globalCompositeOperation = "source-over"
    drawLabels()
  }

  // A soft light under each token, strongest as it leaves the stack.
  const drawGlows = () => {
    const strength = mode === "emit" ? 0.3 : 0.1
    for (const head of frame.heads) {
      const radius = 18 + 30 * head.heat
      const gradient = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, radius)
      gradient.addColorStop(0, rgba(glowRgb, strength * head.heat))
      gradient.addColorStop(1, rgba(glowRgb, 0))
      ctx.fillStyle = gradient
      ctx.fillRect(head.x - radius, head.y - radius, radius * 2, radius * 2)
    }
  }

  // One path per ink level: the fill style changes LEVELS times a frame, not once a point.
  const drawPoints = (minUnits: number) => {
    heads.fill(-1)
    for (let index = frame.count - 1; index >= 0; index -= 1) {
      const level = frame.level[index]
      next[index] = heads[level]
      heads[level] = index
    }
    for (let level = 0; level < LEVELS; level += 1) {
      let index = heads[level]
      if (index < 0) continue
      ctx.fillStyle = palette[level]
      ctx.beginPath()
      while (index >= 0) {
        const radius = Math.max(minUnits, frame.size[index]) / 2
        const x = frame.x[index]
        ctx.moveTo(x + radius, frame.y[index])
        ctx.arc(x, frame.y[index], radius, 0, TAU)
        index = next[index]
      }
      ctx.fill()
    }
  }

  // The words beside their tokens.
  const drawLabels = () => {
    if (frame.labels.length === 0) return
    ctx.font = `500 ${LABEL_UNITS}px ${labelFamily}`
    ctx.textBaseline = "middle"
    for (const label of frame.labels) {
      ctx.fillStyle = rgba(labelRgb, 0.85 * label.alpha)
      ctx.fillText(label.text, label.x + LABEL_OFFSET_UNITS.x, label.y + LABEL_OFFSET_UNITS.y)
    }
  }

  const applyParallax = () => {
    for (const { element, depthPx } of layers) {
      const x = (-tilt.yaw / TILT_YAW) * depthPx
      const y = (tilt.pitch / TILT_PITCH) * depthPx
      element.style.transform = `translate3d(${x.toFixed(2)}px, ${y.toFixed(2)}px, 0)`
    }
  }

  const tick = (nowMs: number) => {
    frameId = 0
    const stepMs = lastMs === 0 ? 16 : Math.min(MAX_STEP_MS, nowMs - lastMs)
    lastMs = nowMs
    timeS += stepMs / 1000
    const ease = 1 - Math.exp(-stepMs / TILT_EASE_MS)
    tilt.pitch += (tilt.targetPitch - tilt.pitch) * ease
    tilt.yaw += (tilt.targetYaw - tilt.yaw) * ease
    applyParallax()
    draw()
    schedule()
  }

  const running = () => visible && !document.hidden && !reducedQuery.matches

  const schedule = () => {
    if (frameId !== 0) return
    if (running()) frameId = requestAnimationFrame(tick)
    else lastMs = 0
  }

  const stillFrame = () => {
    timeS = STATIC_TIME_S
    tilt.pitch = tilt.yaw = tilt.targetPitch = tilt.targetYaw = 0
    applyParallax()
    draw()
  }

  resolveTheme()
  resize()
  draw()
  root.dataset.live = ""

  new ResizeObserver(() => {
    resize()
    rootRect = null
    if (!running()) draw()
  }).observe(canvas)

  new IntersectionObserver(
    entries => {
      visible = entries.some(entry => entry.isIntersecting)
      schedule()
    },
    { threshold: 0.05 }
  ).observe(canvas)

  document.addEventListener("visibilitychange", schedule)

  reducedQuery.addEventListener("change", () => {
    if (reducedQuery.matches) {
      if (frameId !== 0) cancelAnimationFrame(frameId)
      frameId = 0
      stillFrame()
    } else {
      schedule()
    }
  })

  new MutationObserver(() => {
    resolveTheme()
    if (!running()) draw()
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] })

  // The pointer is read over the whole hero, not just the scene's box.
  const area = root.closest<HTMLElement>(".content-panel") ?? root
  area.addEventListener("pointerenter", () => {
    rootRect = null
  })
  area.addEventListener("pointermove", event => {
    if (event.pointerType !== "mouse" || reducedQuery.matches) return
    rootRect ??= root.getBoundingClientRect()
    if (event.clientY > rootRect.bottom) {
      tilt.targetPitch = tilt.targetYaw = 0
      return
    }
    const x = (event.clientX - rootRect.left) / rootRect.width - 0.5
    const y = (event.clientY - rootRect.top) / rootRect.height - 0.5
    tilt.targetYaw = x * 2 * TILT_YAW
    tilt.targetPitch = y * 2 * TILT_PITCH
  })
  area.addEventListener("pointerleave", () => {
    tilt.targetPitch = 0
    tilt.targetYaw = 0
  })
  window.addEventListener("scroll", () => (rootRect = null), { passive: true })
}
