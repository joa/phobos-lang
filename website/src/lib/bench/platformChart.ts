// The geometry of the platforms page's ratio chart: one line per card through every test its bench.py run
// measured on the platform model, prompt sizes then decode sizes, in viewBox units. Runs at build time and in
// vitest; the component only draws what this returns.

import type { Platform } from "./platforms"

export interface ChartPoint {
  test: string
  ratio: number
  // Each engine's mean rate, tokens per second.
  phobos: number
  other: number
}

export interface ChartSeries {
  card: string
  // A categorical color slot, 1-based, fixed by the card's place in the chip order.
  slot: number
  points: ChartPoint[]
  // Runs of points drawn as one polyline: a line never joins a prompt size to a decode size.
  segments: ChartPoint[][]
}

const kindOf = (test: string) => test.slice(0, 2)
const sizeOf = (test: string) => Number(test.slice(2))

function byTest(a: string, b: string): number {
  return kindOf(a).localeCompare(kindOf(b)) || sizeOf(a) - sizeOf(b)
}

export function chartSeries(platforms: Platform[]): ChartSeries[] {
  return platforms.flatMap((platform, index) => {
    const tests = platform.model?.tests ?? {}
    const points = Object.keys(tests)
      .sort(byTest)
      .flatMap(test => {
        const { phobos, other } = tests[test]
        if (!phobos || !other || other.mean === 0) return []
        return [{ test, ratio: phobos.mean / other.mean, phobos: phobos.mean, other: other.mean }]
      })
    if (points.length === 0) return []
    const segments = ["pp", "tg"]
      .map(kind => points.filter(point => kindOf(point.test) === kind))
      .filter(segment => segment.length > 0)
    return [{ card: platform.card, slot: index + 1, points, segments }]
  })
}

// Every test any card ran, prompt sizes first.
export function testOrder(series: ChartSeries[]): string[] {
  return [...new Set(series.flatMap(line => line.points.map(point => point.test)))].sort(byTest)
}

export interface ChartGeometry {
  width: number
  height: number
  plot: { left: number; right: number; top: number; bottom: number }
  x: (test: string) => number
  y: (ratio: number) => number
  yTicks: number[]
  groups: { label: string; center: number }[]
  // A card's name beside its last point, nudged apart so no two overlap.
  labels: { card: string; x: number; y: number }[]
}

const LABEL_GAP = 14

export function chartGeometry(series: ChartSeries[], tests: string[]): ChartGeometry {
  const width = 760
  const height = 380
  const plot = { left: 56, right: width - 112, top: 16, bottom: height - 52 }

  // A prompt group, a gap of one slot, then the decode group.
  const prompts = tests.filter(test => kindOf(test) === "pp")
  const decodes = tests.filter(test => kindOf(test) === "tg")
  const gapSlots = prompts.length > 0 && decodes.length > 0 ? 1 : 0
  const slots = tests.length + gapSlots
  const step = (plot.right - plot.left) / Math.max(slots, 1)
  const slotOf = (test: string) => {
    const index = tests.indexOf(test)
    if (index < 0) throw new Error(`no test ${test} on the chart`)
    return index + (kindOf(test) === "tg" ? gapSlots : 0)
  }
  const x = (test: string) => plot.left + step * (slotOf(test) + 0.5)

  const ratios = series.flatMap(line => line.points.map(point => point.ratio))
  const tick = niceStep(Math.max(...ratios, 1) - Math.min(...ratios, 1))
  const low = Math.floor(Math.min(...ratios, 1) / tick) * tick
  const high = Math.ceil(Math.max(...ratios, 1) / tick) * tick
  const y = (ratio: number) => plot.bottom - ((ratio - low) / (high - low || 1)) * (plot.bottom - plot.top)
  const yTicks: number[] = []
  for (let value = low; value <= high + tick / 2; value += tick) yTicks.push(Math.round(value * 100) / 100)

  const center = (group: string[]) => (x(group[0]) + x(group[group.length - 1])) / 2
  const groups = [
    ...(prompts.length > 0 ? [{ label: "prompt processing", center: center(prompts) }] : []),
    ...(decodes.length > 0 ? [{ label: "token generation", center: center(decodes) }] : []),
  ]

  const labels = spread(
    series.map(line => {
      const last = line.points[line.points.length - 1]
      return { card: line.card, x: plot.right + 8, y: y(last.ratio) }
    }),
    plot.top,
    plot.bottom
  )
  return { width, height, plot, x, y, yTicks, groups, labels }
}

// A step of 1, 2 or 5 times a power of ten that cuts the range into four to eight bands.
function niceStep(range: number): number {
  const rough = Math.max(range, 0.1) / 5
  const power = 10 ** Math.floor(Math.log10(rough))
  const scaled = rough / power
  return (scaled < 1.5 ? 1 : scaled < 3.5 ? 2 : 5) * power
}

// Pushes labels apart top to bottom, then back up from the bottom if they ran off the plot.
export function spread<T extends { y: number }>(labels: T[], top: number, bottom: number): T[] {
  const sorted = labels.map(label => ({ ...label })).sort((a, b) => a.y - b.y)
  for (let index = 1; index < sorted.length; index++) {
    sorted[index].y = Math.max(sorted[index].y, sorted[index - 1].y + LABEL_GAP)
  }
  if (sorted.length > 0) sorted[sorted.length - 1].y = Math.min(sorted[sorted.length - 1].y, bottom)
  for (let index = sorted.length - 2; index >= 0; index--) {
    sorted[index].y = Math.min(sorted[index].y, sorted[index + 1].y - LABEL_GAP)
  }
  if (sorted.length > 0 && sorted[0].y < top) {
    const shift = top - sorted[0].y
    for (const label of sorted) label.y += shift
  }
  return sorted
}
