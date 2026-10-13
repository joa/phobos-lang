// The hero's live scene: the moon's point cloud, its orbit and the tile stream on one 2D canvas, plus
// pointer parallax for the star layers. Loaded lazily after the page is idle. Runs only while the
// hero is on screen in a visible tab; under reduced motion it draws the opening frame once.

import { composeFrame, inkOf, LEVELS, LIGHT, makeFrame, makeScene, type Frame, type InkMode, type SceneData, type Token } from "./frame"
import { VIEW_UNITS } from "./geometry"

const MAX_DPR = 2
const MAX_STEP_MS = 50
const TILT_PITCH = 0.16
const TILT_YAW = 0.26
const TILT_EASE_MS = 420
// The smallest point the canvas draws, in CSS px.
const MIN_POINT_PX = 0.85
const TAU = Math.PI * 2

type Rgb = [number, number, number]

function bodyCountFor(widthPx: number): number {
  return widthPx < 720 ? 2600 : widthPx < 1200 ? 4800 : 7000
}

function parseRgb(value: string): Rgb {
  const numbers = value.match(/[\d.]+/g)?.map(Number) ?? [255, 255, 255]
  // color(srgb r g b) gives fractions; rgb() gives bytes.
  if (value.startsWith("color(")) return [numbers[0] * 255, numbers[1] * 255, numbers[2] * 255]
  return [numbers[0], numbers[1], numbers[2]]
}

export function start(root: HTMLElement): void {
  const canvas = root.querySelector<HTMLCanvasElement>(".hero-canvas")
  const context = canvas?.getContext("2d", { alpha: true })
  if (!canvas || !context) return
  const ctx = context

  const scene: SceneData = makeScene(bodyCountFor(window.innerWidth))
  const frame: Frame = makeFrame(scene)
  const heads = new Int32Array(LEVELS)
  const next = new Int32Array(frame.x.length)
  const layers = [...root.querySelectorAll<HTMLElement>("[data-depth]")].map(element => ({
    element,
    depthPx: Number(element.dataset.depth) || 0,
  }))

  const reducedQuery = window.matchMedia("(prefers-reduced-motion: reduce)")
  let palette: string[] = []
  let sky: Rgb = [0, 0, 0]
  let tint: Rgb = [0, 0, 0]
  let mode: InkMode = "emit"
  let cssWidthPx = 0
  let unitsToPx = 1
  let timeS = 0
  let visible = false
  let frameId = 0
  let lastMs = 0
  const tilt = { pitch: 0, yaw: 0, targetPitch: 0, targetYaw: 0 }
  let heroRect: DOMRect | null = null

  const resolveTheme = () => {
    mode = document.documentElement.dataset.theme === "light" ? "stipple" : "emit"
    const probe = document.createElement("span")
    probe.style.display = "none"
    root.append(probe)
    const read = (token: Token | "sky-0"): Rgb => {
      probe.style.color = `var(--${token})`
      return parseRgb(getComputedStyle(probe).color)
    }
    const tokens = { brand: read("brand"), "brand-hi": read("brand-hi"), star: read("star") }
    sky = read("sky-0")
    tint = mode === "emit" ? tokens["brand-hi"] : tokens.star
    probe.remove()
    palette = Array.from({ length: LEVELS }, (_, level) => {
      const ink = inkOf(level)
      const from = tokens[ink.from]
      const to = tokens[ink.to]
      const channel = (index: number) => Math.round(from[index] + (to[index] - from[index]) * ink.mix)
      // At dawn the faintest dots would vanish against the sky; lift them a little.
      const alpha = mode === "stipple" ? Math.min(1, ink.alpha * 0.85 + 0.08) : ink.alpha
      return `rgba(${channel(0)}, ${channel(1)}, ${channel(2)}, ${alpha})`
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
    composeFrame(scene, { timeS, pitch: tilt.pitch, yaw: tilt.yaw }, mode, frame)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.setTransform(unitsToPx, 0, 0, unitsToPx, 0, 0)
    const minUnits = (MIN_POINT_PX * VIEW_UNITS) / Math.max(1, cssWidthPx)
    const blend: GlobalCompositeOperation = mode === "emit" ? "lighter" : "source-over"

    ctx.globalCompositeOperation = blend
    drawPoints(0, frame.farEnd, minUnits)
    drawOccluder()
    ctx.globalCompositeOperation = blend
    drawPoints(frame.farEnd, frame.count, minUnits)
    ctx.globalCompositeOperation = "source-over"
  }

  // One path per ink level: the fill style changes LEVELS times a frame, not once a point.
  const drawPoints = (from: number, to: number, minUnits: number) => {
    heads.fill(-1)
    for (let index = to - 1; index >= from; index -= 1) {
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

  // The body is solid: the sky behind it, and the orbit's far side, do not show through.
  const drawOccluder = () => {
    const { x, y, rx, ry, angle } = frame.silhouette
    ctx.save()
    ctx.globalCompositeOperation = "source-over"
    ctx.translate(x, y)
    ctx.rotate(angle)
    ctx.scale(rx, ry)
    const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 1.06)
    const [r, g, b] = sky
    gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.94)`)
    gradient.addColorStop(0.82, `rgba(${r}, ${g}, ${b}, 0.9)`)
    gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`)
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(0, 0, 1.06, 0, Math.PI * 2)
    ctx.fill()

    // Inside the silhouette, a soft wash: light on the sunward side by night, shade on the far side at dawn.
    ctx.beginPath()
    ctx.arc(0, 0, 0.98, 0, Math.PI * 2)
    ctx.clip()
    ctx.setTransform(unitsToPx, 0, 0, unitsToPx, 0, 0)
    const side = mode === "emit" ? 1 : -1
    const reach = Math.max(rx, ry)
    const cx = x + LIGHT[0] * side * reach * 0.55
    const cy = y - LIGHT[1] * side * reach * 0.55
    const wash = ctx.createRadialGradient(cx, cy, 0, cx, cy, reach * 1.25)
    const [tr, tg, tb] = tint
    const strength = mode === "emit" ? 0.2 : 0.09
    wash.addColorStop(0, `rgba(${tr}, ${tg}, ${tb}, ${strength})`)
    wash.addColorStop(1, `rgba(${tr}, ${tg}, ${tb}, 0)`)
    ctx.fillStyle = wash
    ctx.fillRect(x - reach * 1.2, y - reach * 1.2, reach * 2.4, reach * 2.4)
    ctx.restore()
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
    if (running()) {
      frameId = requestAnimationFrame(tick)
    } else {
      lastMs = 0
    }
  }

  const stillFrame = () => {
    timeS = 0
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
    heroRect = null
    if (!running()) draw()
  }).observe(canvas)

  new IntersectionObserver(
    entries => {
      visible = entries.some(entry => entry.isIntersecting)
      schedule()
    },
    { threshold: 0.05 }
  ).observe(root)

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

  root.addEventListener("pointerenter", () => {
    heroRect = null
  })
  root.addEventListener("pointermove", event => {
    if (event.pointerType !== "mouse" || reducedQuery.matches) return
    heroRect ??= root.getBoundingClientRect()
    const x = (event.clientX - heroRect.left) / heroRect.width - 0.5
    const y = (event.clientY - heroRect.top) / heroRect.height - 0.5
    tilt.targetYaw = x * 2 * TILT_YAW
    tilt.targetPitch = y * 2 * TILT_PITCH
  })
  root.addEventListener("pointerleave", () => {
    tilt.targetPitch = 0
    tilt.targetYaw = 0
  })
  window.addEventListener("scroll", () => (heroRect = null), { passive: true })
}
