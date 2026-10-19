// The small drawing above each compiler fact, as SVG markup built at build time. Each draws from the same
// data its fact states: the tiles' live ranges and offsets, the barrier sites in op order, the search
// space's size, the chip list. Motion is CSS only, and only under [data-live="motion"]; without it every
// cue is its settled drawing.

import type { Barriers, Placement } from "./facts"

export const CUE_WIDTH = 200
export const CUE_HEIGHT = 64
export const CUE_PERIOD_MS = 7000

const PAD = 8
const INNER = CUE_WIDTH - 2 * PAD

// The sweep runs from 5% to 85% of the period, then holds before the next pass.
const SWEEP_FROM = 5
const SWEEP_SPAN = 80

export interface Cue {
  markup: string
  // Keyframes only this cue needs, scoped by its id.
  css: string
}

const fixed = (value: number) => Number(value.toFixed(2))

function sweep(): string {
  return `<line class="cn-sweep" x1="${PAD}" x2="${PAD}" y1="2" y2="${CUE_HEIGHT - 2}" />`
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

// Live range across, offset down: a tile is a box from its first op to its last, at its bytes.
export function placementRects(place: Placement): Rect[] {
  const top = 4
  const height = CUE_HEIGHT - 2 * top
  return place.spans.map(span => {
    const x = PAD + (span.firstOp / place.opCount) * INNER
    const right = PAD + ((span.lastOp + 1) / place.opCount) * INNER
    return {
      x: fixed(x),
      y: fixed(top + (span.offsetBytes / place.peakBytes) * height),
      width: fixed(Math.max(right - x, 2)),
      height: fixed((span.bytes / place.peakBytes) * height - 1.5),
    }
  })
}

const sweepPercent = (x: number) => SWEEP_FROM + ((x - PAD) / INNER) * SWEEP_SPAN

export function placementCue(id: string, place: Placement): Cue {
  const rects = placementRects(place)
  const markup = rects
    .map(
      (rect, index) =>
        `<rect class="cn-tile" style="animation-name: ${id}-${index}" x="${rect.x}" y="${rect.y}" width="${rect.width}" height="${rect.height}" />`
    )
    .join("")
  // Each tile lights while the sweep is inside its live range.
  const css = rects
    .map((rect, index) => {
      const on = fixed(sweepPercent(rect.x))
      const off = fixed(sweepPercent(rect.x + rect.width))
      return `@keyframes ${id}-${index} { 0%, ${fixed(Math.max(on - 0.5, 0))}% { fill-opacity: 0.18; stroke-opacity: 0.5 } ${on}%, ${off}% { fill-opacity: 0.55; stroke-opacity: 1 } ${fixed(off + 2)}%, 100% { fill-opacity: 0.18; stroke-opacity: 0.5 } }`
    })
    .join("\n")
  return { markup: `<line class="cn-axis" x1="${PAD}" x2="${CUE_WIDTH - PAD}" y1="${CUE_HEIGHT - 1}" y2="${CUE_HEIGHT - 1}" />${markup}${sweep()}`, css }
}

export function barrierCue(sync: Barriers): Cue {
  const count = sync.sites.length
  const step = count > 1 ? INNER / (count - 1) : 0
  const markup = sync.sites
    .map((kept, index) => {
      const x = fixed(PAD + index * step)
      const delay = Math.round(((sweepPercent(x) - 1) / 100) * CUE_PERIOD_MS)
      return kept
        ? `<line class="cn-barrier is-kept" style="animation-delay: ${delay}ms" x1="${x}" x2="${x}" y1="12" y2="${CUE_HEIGHT - 12}" />`
        : `<line class="cn-barrier is-elided" x1="${x}" x2="${x}" y1="20" y2="${CUE_HEIGHT - 20}" />`
    })
    .join("")
  return { markup: `${markup}${sweep()}`, css: "" }
}

// Two staging slots: one fills from global memory while the other is read, then they swap.
export function pipelineCue(): Cue {
  const width = 56
  const gap = 16
  const left = (CUE_WIDTH - 2 * width - gap) / 2
  const slots = [0, 1]
    .map(slot => {
      const x = left + slot * (width + gap)
      return (
        `<rect class="cn-slot" x="${x}" y="10" width="${width}" height="44" />` +
        `<rect class="cn-fill is-slot-${slot}" x="${x + 3}" y="13" width="${width - 6}" height="38" />`
      )
    })
    .join("")
  return { markup: slots, css: "" }
}

// The 16 x 16 fragments of a 128 x 48 patch, lit in a wave as warps issue them.
export function tensorCoreCue(): Cue {
  const columns = 8
  const rows = 3
  const size = 14
  const gap = 4
  const left = (CUE_WIDTH - (columns * size + (columns - 1) * gap)) / 2
  const top = (CUE_HEIGHT - (rows * size + (rows - 1) * gap)) / 2
  const cells: string[] = []
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const delay = Math.round((row + column) * 160)
      cells.push(
        `<rect class="cn-frag" style="animation-delay: ${delay}ms" x="${left + column * (size + gap)}" y="${top + row * (size + gap)}" width="${size}" height="${size}" />`
      )
    }
  }
  return { markup: cells.join(""), css: "" }
}

// One dot per configuration, visited in order.
// The grid takes the box's proportions, so a large search still reads as dots: 64 is 16 by 4, 512 is 45
// by 12, and a dot shrinks once the spacing gets tight.
export function searchCue(configs: number): Cue {
  const height = CUE_HEIGHT - 16
  const columns = Math.max(1, Math.min(configs, Math.ceil(Math.sqrt((configs * INNER) / height))))
  const rows = Math.ceil(configs / columns)
  const spacing = Math.min(INNER / columns, height / Math.max(rows - 1, 1))
  const radius = fixed(Math.min(2.2, spacing * 0.4))
  const left = (CUE_WIDTH - (columns - 1) * spacing) / 2
  const top = (CUE_HEIGHT - (rows - 1) * spacing) / 2
  const dots: string[] = []
  for (let index = 0; index < configs; index++) {
    const delay = Math.round((index / configs) * CUE_PERIOD_MS * 0.8)
    const cx = fixed(left + (index % columns) * spacing)
    const cy = fixed(top + Math.floor(index / columns) * spacing)
    dots.push(`<circle class="cn-config" style="animation-delay: ${delay}ms" cx="${cx}" cy="${cy}" r="${radius}" />`)
  }
  return { markup: dots.join(""), css: "" }
}

// The chips on one line, first and last named, warmed one after another.
export function chipCue(chips: string[]): Cue {
  const left = 24
  const right = CUE_WIDTH - 24
  const y = 24
  const step = chips.length > 1 ? (right - left) / (chips.length - 1) : 0
  const dots = chips
    .map((chip, index) => {
      const x = fixed(left + index * step)
      const delay = Math.round(index * 700)
      return `<circle class="cn-chip" style="animation-delay: ${delay}ms" cx="${x}" cy="${y}" r="4"><title>${chip}</title></circle>`
    })
    .join("")
  const names =
    chips.length > 1
      ? `<text class="cn-chip-name" x="${left}" y="${y + 28}" text-anchor="middle">${chips[0]}</text>` +
        `<text class="cn-chip-name" x="${right}" y="${y + 28}" text-anchor="middle">${chips[chips.length - 1]}</text>`
      : ""
  return {
    markup: `<line class="cn-axis" x1="${left}" x2="${right}" y1="${y}" y2="${y}" />${dots}${names}`,
    css: "",
  }
}
