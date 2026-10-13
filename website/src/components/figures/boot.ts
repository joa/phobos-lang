// Starts every figure on the page. A figure is a module named <name>.figure.ts anywhere under src/,
// whose default export extends ChartFigure; the markup names it with data-chart-figure="<name>".
// Modules load lazily, one chunk each, and only once a figure comes within a screen of the viewport,
// so a page pays only for the figures the reader gets near; until then the build-time drawing shows
// the end state. One broken figure must not take the others down. Once started, each figure watches
// itself (chart-figure.ts): its frame loop, and with it its loop's hold, runs only while 30% of it is
// on screen in a visible tab, and resumes where it stopped.

import { ensureGlowDefs } from "../../lib/figure/draw"
import type { FigureConstructor } from "../../lib/figure/chart-figure"

const NEAR_MARGIN = "100% 0px"

const modules = import.meta.glob<{ default: FigureConstructor }>("/src/**/*.figure.ts")

const loaders = new Map<string, () => Promise<{ default: FigureConstructor }>>()
for (const [path, load] of Object.entries(modules)) {
  const name = path.slice(path.lastIndexOf("/") + 1, -".figure.ts".length)
  loaders.set(name, load)
}

// The two sides of a kernel pair step in lockstep, and on a narrow page one side is display: none,
// which never intersects; the pair is watched as one and starts both.
const TOGETHER_SELECTOR = "[data-viz-pair]"

// Each kind of figure's script calls startFigures(), so a page with two kinds calls it twice before
// any figure has started (data-live is set by start()); a root is queued once.
const queued = new WeakSet<Element>()
const waiting = new Map<Element, HTMLElement[]>()
let nearObserver: IntersectionObserver | null = null

async function startFigure(root: HTMLElement): Promise<void> {
  const name = root.dataset.chartFigure ?? ""
  try {
    const load = loaders.get(name)
    if (!load) throw new Error(`no module ${name}.figure.ts`)
    const { default: Figure } = await load()
    new Figure(root).start()
  } catch (error) {
    console.error(`Figure "${name}" failed to start`, error)
  }
}

function onNear(entries: IntersectionObserverEntry[], observer: IntersectionObserver): void {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue
    observer.unobserve(entry.target)
    for (const root of waiting.get(entry.target) ?? []) void startFigure(root)
    waiting.delete(entry.target)
  }
}

export function startFigures(): void {
  const roots = document.querySelectorAll<HTMLElement>("[data-chart-figure]:not([data-live])")
  if (roots.length === 0) return
  ensureGlowDefs()
  nearObserver ??= new IntersectionObserver(onNear, { rootMargin: NEAR_MARGIN })
  for (const root of roots) {
    if (queued.has(root)) continue
    queued.add(root)
    const target = root.closest(TOGETHER_SELECTOR) ?? root
    const group = waiting.get(target)
    if (group) {
      group.push(root)
    } else {
      waiting.set(target, [root])
      nearObserver.observe(target)
    }
  }
}
