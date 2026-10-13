// Starts every figure on the page. A figure is a module named <name>.figure.ts anywhere under src/,
// whose default export extends ChartFigure; the markup names it with data-chart-figure="<name>".
// Modules load lazily, one chunk each, so a page pays only for the figures it has. One broken figure
// must not take the others down.

import { ensureGlowDefs } from "../../lib/figure/draw"
import type { FigureConstructor } from "../../lib/figure/chart-figure"

const modules = import.meta.glob<{ default: FigureConstructor }>("/src/**/*.figure.ts")

const loaders = new Map<string, () => Promise<{ default: FigureConstructor }>>()
for (const [path, load] of Object.entries(modules)) {
  const name = path.slice(path.lastIndexOf("/") + 1, -".figure.ts".length)
  loaders.set(name, load)
}

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

export function startFigures(): void {
  const roots = document.querySelectorAll<HTMLElement>("[data-chart-figure]:not([data-live])")
  if (roots.length === 0) return
  ensureGlowDefs()
  for (const root of roots) void startFigure(root)
}
