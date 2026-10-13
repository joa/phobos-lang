// The compiler scene in the browser: runs the journey while the scene is on screen in a visible tab,
// turns the ring, switches the station drawings and captions, and drives the cloud. Under reduced
// motion nothing moves: a click on the ring switches the station at once. Without script the page
// keeps the graph IR it was built with.

import { easeInOut } from "../../../lib/figure/math"
import { Cloud, type CloudStation } from "./cloud"
import { PipelineClock } from "./clock"
import { passesStation } from "./passes"
import { ringSpot, ringTurn } from "./ring"
import { STATIONS, stationGeometry } from "./stations"
import type { Layout } from "./types"

const VISIBLE_RATIO = 0.3
const MAX_FRAME_MS = 50
const NARROW_BELOW_PX = 600

export function mountPipeline(root: HTMLElement): void {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches
  const canvas = root.querySelector<HTMLCanvasElement>(".pp-canvas")
  const probes = root.querySelector<HTMLElement>(".pp-probes")
  const spots = [...root.querySelectorAll<HTMLElement>(".pp-spot")]
  const buttons = [...root.querySelectorAll<HTMLButtonElement>("[data-goto]")]
  const captions = [...root.querySelectorAll<HTMLElement>(".pp-cap")]
  const groups = (layout: Layout) => [...root.querySelectorAll<SVGGElement>(`.pp-svg.is-${layout} .pp-st`)]
  if (!canvas || !probes) return
  root.dataset.live = reduced ? "still" : "motion"

  const count = STATIONS.length
  const clock = new PipelineClock({ count, dwellMs: index => STATIONS[index].dwellMs })
  let cloud: Cloud | null = null
  let layout: Layout = root.clientWidth < NARROW_BELOW_PX ? "narrow" : "wide"
  let shown = -1
  let on = -1
  let swing = 1
  let visible = false
  let started = false
  let frameId = 0
  let lastMs = 0

  const cloudStations = (forLayout: Layout): CloudStation[] =>
    STATIONS.map(station => ({
      id: station.id,
      geometry: stationGeometry(station.id, forLayout),
      timing: station.id === "passes" ? passesStation(forLayout).timing : undefined,
    }))

  const placeRing = (turn: number) => {
    spots.forEach((spot, index) => {
      const at = ringSpot(index, count, turn)
      spot.style.setProperty("--x", `${at.x}%`)
      spot.style.setProperty("--y", `${at.y}%`)
      spot.style.setProperty("--depth", String(at.depth))
      spot.style.zIndex = String(Math.round(at.depth * 10))
    })
  }

  const show = (index: number) => {
    shown = index
    root.dataset.station = STATIONS[index].id
    buttons.forEach((button, at) => button.setAttribute("aria-pressed", String(at === index)))
    captions.forEach((caption, at) => caption.classList.toggle("is-on", at === index))
    for (const forLayout of ["wide", "narrow"] as const) {
      groups(forLayout).forEach((group, at) => {
        if (at !== index) group.classList.remove("is-on")
      })
    }
  }

  const turnOn = (index: number) => {
    on = index
    for (const forLayout of ["wide", "narrow"] as const) groups(forLayout)[index]?.classList.add("is-on")
  }

  const pose = () => ({
    station: clock.station,
    moving: clock.moving,
    progress: clock.moveProgress,
    stationMs: clock.stationMs,
    swing,
  })

  const ensureCloud = () => {
    if (cloud) return cloud
    cloud = new Cloud(canvas, probes)
    cloud.setStations(cloudStations(layout))
    return cloud
  }

  const frame = (nowMs: number) => {
    frameId = 0
    const dt = Math.min(MAX_FRAME_MS, Math.max(0, nowMs - lastMs))
    lastMs = nowMs
    const before = clock.station
    clock.advance(dt)
    if (clock.station !== before) {
      swing = -swing
      cloud?.beginMove()
      show(clock.station)
    }
    if (clock.stationMs >= 0 && on !== clock.station) turnOn(clock.station)
    if (clock.moving) placeRing(ringTurn(clock.from, clock.station, count, easeInOut(clock.moveProgress)))
    root.style.setProperty("--p", String(Math.round(clock.restProgress * 1000) / 1000))
    cloud?.render(pose())
    schedule()
  }

  const schedule = () => {
    if (frameId || reduced || !visible || document.visibilityState !== "visible") return
    lastMs = performance.now()
    frameId = requestAnimationFrame(frame)
  }

  const stillRender = () => {
    ensureCloud().render({ station: clock.station, moving: false, progress: 1, stationMs: 20000, swing })
  }

  const go = (index: number) => {
    if (reduced) {
      clock.restart(index)
      show(index)
      turnOn(index)
      placeRing(index)
      stillRender()
      return
    }
    if (!started) start()
    if (index === clock.station && !clock.moving) return
    swing = -swing
    cloud?.beginMove()
    clock.jump(index)
    show(index)
  }

  const start = () => {
    started = true
    ensureCloud()
    clock.restart(0)
    show(0)
    turnOn(0)
    placeRing(0)
  }

  buttons.forEach((button, index) => button.addEventListener("click", () => go(index)))

  if (!reduced) {
    // The scene opens on the source, before it comes into view.
    show(0)
    on = -1
    placeRing(0)
  }

  new IntersectionObserver(
    entries => {
      visible = entries.some(entry => entry.intersectionRatio >= VISIBLE_RATIO)
      if (!visible) return
      if (reduced) {
        if (!cloud) {
          clock.restart(shown < 0 ? Number(root.dataset.overview ?? 0) : shown)
          stillRender()
        }
        return
      }
      if (!started) start()
      schedule()
    },
    { threshold: [0, VISIBLE_RATIO, 0.6] }
  ).observe(root.querySelector(".pp-stage") ?? root)

  document.addEventListener("visibilitychange", schedule)

  new ResizeObserver(() => {
    const next: Layout = root.clientWidth < NARROW_BELOW_PX ? "narrow" : "wide"
    if (!cloud) return
    if (next !== layout) {
      layout = next
      cloud.setStations(cloudStations(layout))
    } else {
      cloud.resize()
    }
    if (reduced || !frameId) cloud.render(reduced ? { ...pose(), moving: false, stationMs: 20000 } : pose())
  }).observe(root)

  new MutationObserver(() => {
    cloud?.readColors()
    if (reduced && cloud) stillRender()
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] })
}
