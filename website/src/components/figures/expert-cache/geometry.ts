// Layout of the expert-cache figure, pure so the tests can check it: one layer's experts in host memory,
// the PCIe link, the CPU, and the card with its expert cache and the rest of the model.

import type { Layout } from "../../../lib/figure/layout"
import { Polyline } from "../../../lib/figure/polyline"
import type { Rect, TextAt } from "../../inference/figure-geometry"
import { EXPERT_COUNT, SLOT_COUNT } from "./simulation"

export const GRID_COLUMNS = 16
export const CACHE_COLUMNS = 8

export interface ExpertCacheGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  experts: { cells: Rect[]; bounds: Rect; label: TextAt }
  link: { upper: Polyline; lower: Polyline; path: Polyline; label: TextAt }
  cpu: { box: Rect; label: TextAt; note: TextAt; result: Polyline }
  gpu: { frame: Rect; label: TextAt }
  cache: { slots: Rect[]; label: TextAt }
  rest: { box: Rect; label: TextAt }
  // Where a dashed wire from a missed expert meets the CPU box.
  cpuInlet: { x: number; y: number }
}

function lattice(count: number, columns: number, x: number, y: number, size: number, pitch: number): Rect[] {
  return Array.from({ length: count }, (_, index) => ({
    x: x + (index % columns) * pitch,
    y: y + Math.floor(index / columns) * pitch,
    width: size,
    height: size,
  }))
}

function bounds(cells: Rect[]): Rect {
  const first = cells[0]
  const last = cells[cells.length - 1]
  return { x: first.x, y: first.y, width: last.x + last.width - first.x, height: last.y + last.height - first.y }
}

export function expertCacheGeometry(layout: Layout): ExpertCacheGeometry {
  if (layout === "wide") {
    const cells = lattice(EXPERT_COUNT, GRID_COLUMNS, 30, 72, 11, 13)
    const grid = bounds(cells)
    const frame = { x: 470, y: 44, width: 226, height: 262 }
    const slots = lattice(SLOT_COUNT, CACHE_COLUMNS, 487, 92, 22, 25)
    const cache = bounds(slots)
    const linkY = 150
    const cpuBox = { x: 290, y: 226, width: 150, height: 54 }
    return {
      widthUnits: 720,
      heightUnits: 320,
      plateLabel: "FIG / MOE / EXPERT CACHE",
      experts: { cells, bounds: grid, label: { x: 30, y: 56, text: "256 experts, host memory" } },
      link: {
        upper: line(256, linkY - 5, 456, linkY - 5),
        lower: line(256, linkY + 5, 456, linkY + 5),
        path: line(258, linkY, frame.x + 8, linkY),
        label: { x: 356, y: linkY - 20, text: "PCIe 3.0 x8, 6.4 GB/s", anchor: "middle", sizeUnits: 11 },
      },
      cpu: {
        box: cpuBox,
        label: { x: cpuBox.x + cpuBox.width / 2, y: cpuBox.y + 20, text: "CPU", anchor: "middle" },
        note: {
          x: cpuBox.x + cpuBox.width / 2,
          y: cpuBox.y + 38,
          text: "from host memory",
          anchor: "middle",
          sizeUnits: 11,
        },
        result: line(cpuBox.x + cpuBox.width, cpuBox.y + cpuBox.height / 2, frame.x + 8, cpuBox.y + cpuBox.height / 2),
      },
      gpu: { frame, label: { x: frame.x + 10, y: frame.y + 16, text: "GPU" } },
      cache: { slots, label: { x: cache.x, y: cache.y - 12, text: "expert cache", sizeUnits: 11 } },
      rest: {
        box: { x: cache.x, y: 238, width: cache.width, height: 50 },
        label: { x: cache.x + cache.width / 2, y: 263, text: "the rest of the model", anchor: "middle", sizeUnits: 11 },
      },
      cpuInlet: { x: cpuBox.x, y: cpuBox.y + cpuBox.height / 2 },
    }
  }
  const cells = lattice(EXPERT_COUNT, GRID_COLUMNS, 113, 66, 9, 11)
  const grid = bounds(cells)
  const linkX = 200
  const frame = { x: 20, y: 340, width: 360, height: 210 }
  const slots = lattice(SLOT_COUNT, CACHE_COLUMNS, 106, 384, 20, 24)
  const cache = bounds(slots)
  const cpuBox = { x: 20, y: 262, width: 140, height: 50 }
  return {
    widthUnits: 400,
    heightUnits: 566,
    plateLabel: "FIG / EXPERT CACHE",
    experts: { cells, bounds: grid, label: { x: 113, y: 50, text: "256 experts, host memory", sizeUnits: 11 } },
    link: {
      upper: line(linkX - 5, grid.y + grid.height + 10, linkX - 5, frame.y),
      lower: line(linkX + 5, grid.y + grid.height + 10, linkX + 5, frame.y),
      path: line(linkX, grid.y + grid.height + 12, linkX, frame.y + 8),
      label: { x: linkX + 14, y: 290, text: "PCIe, 6.4 GB/s", sizeUnits: 11 },
    },
    cpu: {
      box: cpuBox,
      label: { x: cpuBox.x + cpuBox.width / 2, y: cpuBox.y + 18, text: "CPU", anchor: "middle" },
      note: {
        x: cpuBox.x + cpuBox.width / 2,
        y: cpuBox.y + 35,
        text: "from host memory",
        anchor: "middle",
        sizeUnits: 11,
      },
      result: line(cpuBox.x + cpuBox.width / 2, cpuBox.y + cpuBox.height, cpuBox.x + cpuBox.width / 2, frame.y + 8),
    },
    gpu: { frame, label: { x: frame.x + 10, y: frame.y + 16, text: "GPU" } },
    cache: { slots, label: { x: cache.x, y: cache.y - 12, text: "expert cache", sizeUnits: 11 } },
    rest: {
      box: { x: cache.x, y: 508, width: cache.width, height: 30 },
      label: { x: cache.x + cache.width / 2, y: 523, text: "the rest of the model", anchor: "middle", sizeUnits: 11 },
    },
    cpuInlet: { x: cpuBox.x + cpuBox.width / 2, y: cpuBox.y },
  }
}

function line(x1: number, y1: number, x2: number, y2: number): Polyline {
  return new Polyline([
    { x: x1, y: y1 },
    { x: x2, y: y2 },
  ])
}
