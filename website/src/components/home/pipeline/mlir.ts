// The MLIR station: the gpu.func's region tree as nested boxes, each sized by the ops it holds, and one
// point per op in the box of the region that owns it, colored by dialect. Counts are static, read off
// the printed module.

import type { MlirRegion } from "../pipeline-data"
import { structure } from "./data"
import { fill } from "./source"
import { FRAMES, round, target, text, textWidth, type Layout, type Rect, type Shape, type StationGeometry, type Target, type Tone } from "./types"

export const DIALECT_TONES: Record<string, Tone> = {
  arith: "dust",
  vector: "global",
  memref: "shared",
  gpu: "reg",
  scf: "hi",
  llvm: "cost",
}

export const DIALECT_ORDER = ["arith", "vector", "memref", "gpu", "scf", "llvm"]

const HEADER_UNITS = 20
const PAD_UNITS = 5
const MIN_CHILD_UNITS = 14

export function dialectOf(op: string): string {
  return op === "else" ? "scf" : op.split(".")[0]
}

/** Op counts per dialect over the gpu.func's region tree, which the points add up to. */
export function regionDialects(root: MlirRegion): Record<string, number> {
  const counts: Record<string, number> = {}
  const walk = (region: MlirRegion) => {
    for (const [op, count] of Object.entries(region.ops)) counts[dialectOf(op)] = (counts[dialectOf(op)] ?? 0) + count
    // The op that owns a nested region is counted in its parent's ops only by the printer; count it here.
    for (const child of region.children) {
      if (child.op !== "else") counts[dialectOf(child.op)] = (counts[dialectOf(child.op)] ?? 0) + 1
      walk(child)
    }
  }
  walk(root)
  return counts
}

export function ownCount(region: MlirRegion): number {
  return Object.values(region.ops).reduce((sum, count) => sum + count, 0)
}

export interface PlacedRegion {
  region: MlirRegion
  box: Rect
  /** The part of the box its own ops' points fill. */
  own: Rect
  depth: number
}

/** Splits extents in proportion to weights, each at least minUnits. */
export function split(totalUnits: number, weights: number[], minUnits: number): number[] {
  const sum = weights.reduce((acc, weight) => acc + weight, 0) || 1
  let sizes = weights.map(weight => (weight / sum) * totalUnits)
  const small = sizes.map(size => size < minUnits)
  const fixed = small.filter(Boolean).length * minUnits
  const rest = weights.reduce((acc, weight, index) => (small[index] ? acc : acc + weight), 0) || 1
  sizes = weights.map((weight, index) => (small[index] ? minUnits : (weight / rest) * (totalUnits - fixed)))
  return sizes
}

function layoutRegion(region: MlirRegion, box: Rect, depth: number, out: PlacedRegion[]): void {
  const inner = {
    x: box.x + PAD_UNITS,
    y: box.y + HEADER_UNITS,
    width: box.width - PAD_UNITS * 2,
    height: box.height - HEADER_UNITS - PAD_UNITS,
  }
  const own = ownCount(region)
  const items = [{ weight: own, region: null as MlirRegion | null }, ...region.children.map(child => ({ weight: child.total, region: child }))]
  const horizontal = inner.width >= inner.height
  const sizes = split(horizontal ? inner.width : inner.height, items.map(item => Math.max(item.weight, 1)), MIN_CHILD_UNITS)
  let cursor = horizontal ? inner.x : inner.y
  const placed: PlacedRegion = { region, box, own: inner, depth }
  out.push(placed)
  items.forEach((item, index) => {
    const size = sizes[index]
    const cell = horizontal
      ? { x: cursor, y: inner.y, width: size, height: inner.height }
      : { x: inner.x, y: cursor, width: inner.width, height: size }
    cursor += size
    const gap = 3
    const shrunk = horizontal ? { ...cell, x: cell.x + gap / 2, width: cell.width - gap } : { ...cell, y: cell.y + gap / 2, height: cell.height - gap }
    if (item.region) layoutRegion(item.region, shrunk, depth + 1, out)
    else placed.own = shrunk
  })
}

/** Regions laid out into a frame area. Pure, for tests. */
export function mlirLayout(area: Rect): PlacedRegion[] {
  const out: PlacedRegion[] = []
  layoutRegion(structure.mlir.root, area, 0, out)
  return out
}

function regionTitle(region: MlirRegion): string {
  return region.op === "else" ? "else" : `${region.op} ${region.role}`
}

export function mlirStation(layout: Layout): StationGeometry & { regions: PlacedRegion[] } {
  const frame = FRAMES[layout]
  const wide = layout === "wide"
  const area = wide ? { x: 40, y: 74, width: 1120, height: 440 } : { x: 10, y: 60, width: 380, height: 500 }
  const regions = mlirLayout(area)
  const shapes: Shape[] = []
  const targets: Target[] = []

  regions.forEach((placed, index) => {
    const { box, region, depth } = placed
    const delayMs = 120 + depth * 160 + index * 22
    shapes.push({
      tag: "rect",
      attrs: { x: round(box.x), y: round(box.y), width: round(box.width), height: round(box.height), class: `pp-region-box pp-in is-${region.role}` },
      delayMs,
    })
    const title = region.op === "gpu.func" ? "gpu.func @gemm" : regionTitle(region)
    const count = String(region.total)
    const room = box.width - 12
    const full = `${title}  ${count}`
    const label = textWidth(full, 12.5) <= room ? full : textWidth(region.op, 12.5) + textWidth(count, 12.5) + 8 <= room ? `${region.op} ${count}` : textWidth(count, 12.5) <= room ? count : ""
    if (label && box.height >= 18) shapes.push(text(box.x + 6, box.y + 14.5, label, `pp-label pp-in${depth === 0 ? " is-strong" : ""}`, { delayMs: delayMs + 60 }))
  })

  // One point per op, in reading order of dialect, inside its region's own cell.
  for (const placed of regions) {
    const ops: string[] = []
    for (const dialect of DIALECT_ORDER) {
      for (const [op, count] of Object.entries(placed.region.ops)) if (dialectOf(op) === dialect) for (let k = 0; k < count; k += 1) ops.push(dialect)
    }
    for (const child of placed.region.children) if (child.op !== "else") ops.push(dialectOf(child.op))
    ops.sort((a, b) => DIALECT_ORDER.indexOf(a) - DIALECT_ORDER.indexOf(b))
    // A region too small for its header has no cell of its own; its points
    // go in its box instead.
    const cell = placed.own.width > 0 && placed.own.height > 0 ? placed.own : placed.box
    if (ops.length === 0 || cell.width <= 0 || cell.height <= 0) continue
    const cols = Math.max(1, Math.round(Math.sqrt((ops.length * cell.width) / cell.height)))
    const rowsCount = Math.ceil(ops.length / cols)
    const dx = cell.width / cols
    const dy = cell.height / rowsCount
    ops.forEach((dialect, index) => {
      const col = index % cols
      const row = Math.floor(index / cols)
      targets.push(target(cell.x + (col + 0.5) * dx, cell.y + (row + 0.5) * dy, DIALECT_TONES[dialect] ?? "dust", index % 4 === 0 ? 2 : 1, 0, 0.95))
    })
  }

  // Legend: dialects and their counts.
  const counts = regionDialects(structure.mlir.root)
  let x = wide ? 40 : 14
  let y = wide ? frame.height - 22 : frame.height - 46
  for (const dialect of DIALECT_ORDER) {
    const entry = `${dialect} ${counts[dialect] ?? 0}`
    const width = 14 + textWidth(entry, 12.5) + (wide ? 28 : 16)
    if (!wide && x + width > frame.width - 8) {
      x = 14
      y += 20
    }
    shapes.push({ tag: "circle", attrs: { cx: x + 4, cy: y - 4.5, r: 3.5, class: `pp-glyph pp-in t-${DIALECT_TONES[dialect]}` }, delayMs: 500 })
    shapes.push(text(x + 13, y, entry, "pp-label pp-in", { delayMs: 500 }))
    x += width
  }
  if (wide) shapes.push(text(frame.width - 40, frame.height - 22, "one point per op", "pp-label is-faint pp-in", { anchor: "end", delayMs: 500 }))

  return { frame, shapes, targets: fill(targets, frame.width, frame.height), streams: [], regions }
}
