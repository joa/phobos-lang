// Model and geometry of the ctx-hierarchy figure, pure so the tests can check them. The page's worked
// example: a grid of 4 x 3 CTAs launched with @launch(256) on NVIDIA. Each level is drawn to fill
// the same stage, and the zoom maps a picked cell of one level onto the whole stage of the next.

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { lerp } from "../../../lib/figure/math"
import { DEFAULT_THREADS, LANES, threadOf, warpsOf, type ThreadPath } from "./rules"

export const GRID_X = 4
export const GRID_Y = 3
export const THREADS = DEFAULT_THREADS
export const WARPS = warpsOf(THREADS)
export const LEVELS = ["grid", "cta", "warp", "lane"] as const
export type Level = (typeof LEVELS)[number]

export const DEFAULT_PATH = { block: { x: 2, y: 1 }, warp: 3, lane: 17 }

export function pathOf(block: { x: number; y: number }, warp: number, lane: number): ThreadPath {
  return { block, warp, lane, thread: threadOf(warp, lane) }
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface TextAt {
  x: number
  y: number
  text: string
  anchor?: "start" | "middle" | "end"
  small?: boolean
}

export interface PanelSection {
  level: Level
  heading: TextAt
  // y of each field line under the heading; the figure fills in the text.
  lineYs: number[]
  x: number
}

export interface HierarchyGeometry {
  widthUnits: number
  heightUnits: number
  stage: Rect
  pathY: number
  grid: { boxes: Rect[][]; warpLines: Rect[][][]; columnLabels: TextAt[]; rowLabels: TextAt[]; axes: TextAt[] }
  cta: { rows: Rect[]; cells: Rect[][]; rowLabels: TextAt[]; axis: TextAt[] }
  warp: { cells: Rect[]; rowLabels: TextAt[]; heading: TextAt }
  lane: { box: Rect; lines: TextAt[]; registers: Rect[]; registersLabel: TextAt }
  panel: PanelSection[]
}

// Fields under each level's heading in the panel.
export const PANEL_LINES: Record<Level, number> = { grid: 2, cta: 3, warp: 2, lane: 2 }

interface Spec {
  heightUnits: number
  stage: Rect
}

const SPECS: Record<Layout, Spec> = {
  wide: { heightUnits: 330, stage: { x: 24, y: 60, width: 440, height: 250 } },
  narrow: { heightUnits: 486, stage: { x: 16, y: 60, width: 368, height: 236 } },
}

const AXIS_UNITS = 52
const GAP_UNITS = 6
const LINE_UNITS = 16

export function hierarchyGeometry(layout: Layout): HierarchyGeometry {
  const { heightUnits, stage: s } = SPECS[layout]
  const widthUnits = VIEW_WIDTH_UNITS[layout]
  const left = s.x + AXIS_UNITS
  const usable = s.width - AXIS_UNITS

  // Grid: 4 x 3 CTA boxes, each with 8 faint warp bars as a preview of the next level.
  const gridTop = s.y + 40
  const boxWidth = (usable - (GRID_X - 1) * GAP_UNITS) / GRID_X
  const boxHeight = (s.height - 40 - (GRID_Y - 1) * GAP_UNITS) / GRID_Y
  const boxes = Array.from({ length: GRID_Y }, (_, y) =>
    Array.from({ length: GRID_X }, (_, x) => ({
      x: left + x * (boxWidth + GAP_UNITS),
      y: gridTop + y * (boxHeight + GAP_UNITS),
      width: boxWidth,
      height: boxHeight,
    }))
  )
  const warpLines = boxes.map(row =>
    row.map(box => {
      const top = box.y + 20
      const pitch = (box.height - 26) / WARPS
      return Array.from({ length: WARPS }, (_, warp) => ({
        x: box.x + 6,
        y: top + warp * pitch,
        width: box.width - 12,
        height: Math.max(1.5, pitch - 3),
      }))
    })
  )

  // CTA: 8 warps as rows of 32 lanes.
  const ctaTop = s.y + 30
  const rowPitch = (s.height - 30) / WARPS
  const lanePitch = usable / LANES
  const cells = Array.from({ length: WARPS }, (_, warp) =>
    Array.from({ length: LANES }, (_, lane) => ({
      x: left + lane * lanePitch,
      y: ctaTop + warp * rowPitch,
      width: lanePitch - 2,
      height: rowPitch - 9,
    }))
  )
  const rows = cells.map(row => boundsOf(row))

  // Warp: 32 lanes as 2 rows of 16, numbered.
  const warpCellPitch = usable / 16
  const warpCellHeight = 44
  const warpRowGap = 34
  const warpTop = s.y + (s.height - (2 * warpCellHeight + warpRowGap)) / 2 + 8
  const warpCells = Array.from({ length: LANES }, (_, lane) => ({
    x: left + (lane % 16) * warpCellPitch,
    y: warpTop + Math.floor(lane / 16) * (warpCellHeight + warpRowGap),
    width: warpCellPitch - 3,
    height: warpCellHeight,
  }))

  // Lane: one thread with its own registers.
  const laneBox = { x: s.x + s.width / 2 - 120, y: s.y + s.height / 2 - 78, width: 240, height: 156 }
  const registers = Array.from({ length: 6 }, (_, index) => ({
    x: laneBox.x + 18 + index * 16,
    y: laneBox.y + laneBox.height - 34,
    width: 12,
    height: 12,
  }))

  return {
    widthUnits,
    heightUnits,
    stage: s,
    pathY: 42,
    grid: {
      boxes,
      warpLines,
      columnLabels: boxes[0].map((box, x) => ({
        x: box.x + box.width / 2,
        y: s.y + 12,
        text: String(x),
        anchor: "middle",
      })),
      rowLabels: boxes.map((row, y) => ({
        x: left - 12,
        y: row[0].y + row[0].height / 2,
        text: String(y),
        anchor: "end",
      })),
      axes: [
        { x: s.x, y: s.y + 12, text: "ctx.block.x", small: true },
        { x: s.x, y: s.y + 28, text: "ctx.block.y", small: true },
      ],
    },
    cta: {
      rows,
      cells,
      rowLabels: rows.map((row, warp) => ({
        x: left - 8,
        y: row.y + row.height / 2,
        text: `warp ${warp}`,
        anchor: "end",
        small: true,
      })),
      axis: [
        { x: left, y: s.y + 12, text: "lane 0", small: true },
        { x: left + usable - 2, y: s.y + 12, text: "31", anchor: "end", small: true },
      ],
    },
    warp: {
      cells: warpCells,
      rowLabels: [0, 16].map(lane => ({
        x: left - 6,
        y: warpCells[lane].y + warpCellHeight / 2,
        text: `lane ${lane}`,
        anchor: "end",
        small: true,
      })),
      heading: { x: left, y: s.y + 12, text: "", small: true },
    },
    lane: {
      box: laneBox,
      lines: [0, 1, 2].map(index => ({
        x: laneBox.x + 18,
        y: laneBox.y + 24 + index * 22,
        text: "",
        small: index > 0,
      })),
      registers,
      registersLabel: {
        x: registers[registers.length - 1].x + 22,
        y: registers[0].y + 6,
        text: "its own registers",
        small: true,
      },
    },
    panel: panelSections(layout, s),
  }
}

function panelSections(layout: Layout, s: Rect): PanelSection[] {
  const sections: PanelSection[] = []
  if (layout === "wide") {
    const x = s.x + s.width + 26
    let y = s.y + 6
    for (const level of LEVELS) {
      const lineYs = Array.from({ length: PANEL_LINES[level] }, (_, index) => y + (index + 1) * LINE_UNITS)
      sections.push({ level, x, heading: { x, y, text: level }, lineYs })
      y = lineYs[lineYs.length - 1] + LINE_UNITS + 10
    }
    return sections
  }
  const columns: [Level[], number][] = [
    [["grid", "cta"], s.x + 2],
    [["warp", "lane"], s.x + 196],
  ]
  for (const [levels, x] of columns) {
    let y = s.y + s.height + 30
    for (const level of levels) {
      const lineYs = Array.from({ length: PANEL_LINES[level] }, (_, index) => y + (index + 1) * LINE_UNITS)
      sections.push({ level, x, heading: { x, y, text: level }, lineYs })
      y = lineYs[lineYs.length - 1] + LINE_UNITS + 10
    }
  }
  return LEVELS.map(level => sections.find(section => section.level === level) as PanelSection)
}

// The field lines of each level's panel section for a picked thread.
export function panelLines(path: ThreadPath): Record<Level, string[]> {
  return {
    grid: [`ctx.blocks.x, .y = ${GRID_X}, ${GRID_Y}`, `ctx.block.x, .y = ${path.block.x}, ${path.block.y}`],
    cta: [`ctx.threads = ${THREADS}`, `ctx.warps = ${WARPS}`, `ctx.warp = ${path.warp}`],
    warp: [`ctx.lanes = ${LANES}`, `ctx.lane = ${path.lane}`],
    lane: [`ctx.thread = ${path.thread}`, `= ${path.warp} * ${LANES} + ${path.lane}`],
  }
}

// The zoom. A level shows the window W of its own coordinates on the stage S; zooming into a cell C
// moves W from S to C grown to the stage's aspect, and the next level shows the same window mapped
// into its own coordinates, so both move as one.

export function fitAspect(cell: Rect, stage: Rect): Rect {
  const scale = Math.max(cell.width / stage.width, cell.height / stage.height)
  const width = stage.width * scale
  const height = stage.height * scale
  return { x: cell.x + cell.width / 2 - width / 2, y: cell.y + cell.height / 2 - height / 2, width, height }
}

// The window part of the way from the stage to a target of the stage's aspect: the size moves
// geometrically, so the zoom feels even, the center linearly.
export function windowAt(stage: Rect, target: Rect, amount: number): Rect {
  const width = stage.width * Math.pow(target.width / stage.width, amount)
  const height = width * (stage.height / stage.width)
  const cx = lerp(stage.x + stage.width / 2, target.x + target.width / 2, amount)
  const cy = lerp(stage.y + stage.height / 2, target.y + target.height / 2, amount)
  return { x: cx - width / 2, y: cy - height / 2, width, height }
}

// Maps a rect given in the coordinates where `from` is drawn into those where `from` fills `to`.
export function mapRect(rect: Rect, from: Rect, to: Rect): Rect {
  const scale = to.width / from.width
  return {
    x: to.x + (rect.x - from.x) * scale,
    y: to.y + (rect.y - from.y) * scale,
    width: rect.width * scale,
    height: rect.height * scale,
  }
}

export interface Affine {
  scale: number
  dx: number
  dy: number
}

// The transform that shows a window of a level's coordinates on the whole stage.
export function affineFor(window: Rect, stage: Rect): Affine {
  const scale = stage.width / window.width
  return { scale, dx: stage.x - window.x * scale, dy: stage.y - window.y * scale }
}

export function transformAttribute({ scale, dx, dy }: Affine): string {
  const r = (value: number) => Math.round(value * 1000) / 1000
  return `translate(${r(dx)} ${r(dy)}) scale(${r(scale)})`
}

export function boundsOf(rects: Rect[]): Rect {
  const minX = Math.min(...rects.map(r => r.x))
  const minY = Math.min(...rects.map(r => r.y))
  const maxX = Math.max(...rects.map(r => r.x + r.width))
  const maxY = Math.max(...rects.map(r => r.y + r.height))
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
}
