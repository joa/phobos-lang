// Where everything of the visualizer sits, in viewBox units, for both layouts and both zoom levels.
// Pure, so tests can check that nothing overlaps or leaves the plate.

import type { Layout } from "../figure/layout"
import type { KernelModel } from "./model"

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
}

export interface TensorPlace {
  name: string
  /** The drawn tensor; element (row, col) sits at x + col * sx, y + row * sy. */
  rect: Rect
  sx: number
  sy: number
  /** Elements per drawn cell along rows and columns. */
  cellRows: number
  cellCols: number
  label: TextAt
  note: TextAt
}

export interface TileLevel {
  bandLabels: TextAt[]
  grid: { area: Rect; nodes: { x: number; y: number }[]; links: [number, number][] }
  tensors: TensorPlace[]
  shared: {
    x0: number
    x1: number
    top: number
    rowUnits: number
    rows: number
    buffers: { id: number; row: number; rect: Rect }[]
    axisY: number
    zero: TextAt
    peak: TextAt
    empty: TextAt | null
  }
  program: { x0: number; x1: number; y: number; barriers: { index: number; x: number; kept: boolean }[]; note: TextAt }
  threads: {
    dots: { thread: number; x: number; y: number }[]
    rect: Rect
    text: TextAt
    slots: Rect
    slotsText: TextAt
  }
}

export interface LaneLevel {
  bandLabels: TextAt[]
  folded: { warp: number; y: number; xs: number[] }[]
  foldedNote: TextAt | null
  lanes: { x: number; y: number }[]
  laneLabels: TextAt[]
  arcMaxUnits: number
  slots: Rect
  slotsText: TextAt
  stageText: TextAt
  kinds: { name: TextAt; kind: TextAt; dotsX: number; dotsY: number; pitchUnits: number }[]
  idleText: TextAt
}

export interface VizGeometry {
  layout: Layout
  widthUnits: number
  heightUnits: number
  note: TextAt
  tile: TileLevel
  lane: LaneLevel
}

const MARGIN = 14
// Where both levels start, past the widest band label (THREADS, SHUFFLE in letter-spaced caps).
const BAND_X = 92
const MIN_THICK_UNITS = 6
const TARGET_CELL_UNITS = 5
const ROW_UNITS = 15
const KIND_ROWS_MAX = 6

export function vizGeometry(model: KernelModel, layout: Layout): VizGeometry {
  const widthUnits = layout === "wide" ? 720 : 400
  const tile = tileLevel(model, layout, widthUnits)
  const lane = laneLevel(model, layout, widthUnits)
  const bottom = (level: { bottom: number }) => level.bottom
  const heightUnits = Math.ceil(Math.max(bottom(tile), bottom(lane)) + 16)
  return {
    layout,
    widthUnits,
    heightUnits,
    note: { x: widthUnits - MARGIN, y: 20, text: "order, not time", anchor: "end" },
    tile: tile.level,
    lane: lane.level,
  }
}

function tileLevel(model: KernelModel, layout: Layout, widthUnits: number): { level: TileLevel; bottom: number } {
  const wide = layout === "wide"
  const right = widthUnits - MARGIN
  const gridArea: Rect = wide
    ? { x: MARGIN, y: 62, width: 136, height: 92 }
    : { x: MARGIN, y: 62, width: 112, height: 66 }
  const globalX = wide ? 172 : 140
  const tensors = placeTensors(model, { x: globalX, y: 62, width: right - globalX, height: 0 }, layout)
  const tensorsBottom = Math.max(...tensors.map(place => place.note.y), gridArea.y + gridArea.height)
  const bandLabels: TextAt[] = [
    { x: MARGIN, y: 46, text: "grid" },
    { x: globalX, y: 46, text: "global" },
  ]

  const x0 = BAND_X
  const sharedTop = tensorsBottom + 34
  const placed = placeBuffers(model)
  const rows = Math.max(1, ...placed.map(entry => entry.row + 1))
  const peak = Math.max(1, model.sharedPeakBytes)
  const scale = (right - x0) / peak
  const buffers = placed.map(({ id, row, offset, bytes }) => ({
    id,
    row,
    rect: {
      x: x0 + offset * scale,
      y: sharedTop + row * ROW_UNITS,
      width: Math.max(1.5, bytes * scale),
      height: ROW_UNITS - 3,
    },
  }))
  const axisY = sharedTop + rows * ROW_UNITS + 2
  bandLabels.push({ x: MARGIN, y: sharedTop + 5, text: "shared" })

  const programY = axisY + 40
  const ops = Math.max(1, model.opCount - 1)
  const barriers = model.barriers.map(barrier => {
    const index = model.barrierAt(barrier)
    return { index, x: x0 + (index / ops) * (right - x0), kept: barrier.kept }
  })
  const kept = barriers.filter(barrier => barrier.kept).length
  bandLabels.push({ x: MARGIN, y: programY, text: "barriers" })

  const threadsTop = programY + 36
  const warps = model.trace.launch.warps
  const lanes = model.trace.launch.lanes
  const dotPitch = wide ? 5.5 : 4.6
  const rowPitch = 6
  const dots: { thread: number; x: number; y: number }[] = []
  for (let warp = 0; warp < warps; warp += 1) {
    for (let lane = 0; lane < lanes; lane += 1) {
      dots.push({ thread: warp * lanes + lane, x: x0 + lane * dotPitch, y: threadsTop + warp * rowPitch })
    }
  }
  const dotsRect: Rect = {
    x: x0 - 3,
    y: threadsTop - 3,
    width: (lanes - 1) * dotPitch + 6,
    height: (warps - 1) * rowPitch + 6,
  }
  bandLabels.push({ x: MARGIN, y: threadsTop + 2, text: "threads" })
  const dotsBottom = threadsTop + (warps - 1) * rowPitch
  let slots: Rect
  let slotsText: TextAt
  let threadsText: TextAt
  let bottom: number
  if (wide) {
    const slotsX = x0 + lanes * dotPitch + 40
    slots = { x: slotsX, y: threadsTop - 3, width: right - slotsX, height: 14 }
    // The lines sentence gets its own line: beside the threads sentence the two can meet.
    threadsText = { x: x0, y: dotsBottom + 18, text: "" }
    slotsText = { x: slotsX, y: dotsBottom + 36, text: "" }
    bottom = dotsBottom + 36
  } else {
    threadsText = { x: x0, y: dotsBottom + 16, text: "" }
    slots = { x: x0, y: dotsBottom + 30, width: right - x0, height: 14 }
    slotsText = { x: MARGIN, y: dotsBottom + 58, text: "" }
    bandLabels.push({ x: MARGIN, y: dotsBottom + 37, text: "lines" })
    bottom = dotsBottom + 58
  }

  const level: TileLevel = {
    bandLabels,
    grid: placeGrid(model, gridArea),
    tensors,
    shared: {
      x0,
      x1: right,
      top: sharedTop,
      rowUnits: ROW_UNITS,
      rows,
      buffers,
      axisY,
      zero: { x: x0, y: axisY + 12, text: "0" },
      peak: { x: right, y: axisY + 12, text: `${model.sharedPeakBytes} bytes`, anchor: "end" },
      empty: model.buffers.length === 0 ? { x: x0, y: sharedTop + 5, text: "no shared memory" } : null,
    },
    program: {
      x0,
      x1: right,
      y: programY,
      barriers,
      note: {
        x: right,
        y: programY + 18,
        text: `${kept} kept, ${barriers.length - kept} elided, in program order`,
        anchor: "end",
      },
    },
    threads: { dots, rect: dotsRect, text: threadsText, slots, slotsText },
  }
  return { level, bottom }
}

function laneLevel(model: KernelModel, layout: Layout, widthUnits: number): { level: LaneLevel; bottom: number } {
  const wide = layout === "wide"
  const right = widthUnits - MARGIN
  const x0 = BAND_X
  const warps = model.trace.launch.warps
  const lanes = model.trace.launch.lanes
  const foldPitch = wide ? 5.5 : 4.6
  const folded: { warp: number; y: number; xs: number[] }[] = []
  for (let warp = 1; warp < warps; warp += 1) {
    folded.push({
      warp,
      y: 58 + (warp - 1) * 5,
      xs: Array.from({ length: lanes }, (_, lane) => x0 + lane * foldPitch),
    })
  }
  const foldedBottom = folded.length > 0 ? folded[folded.length - 1].y : 50
  const arcMaxUnits = wide ? 70 : 52
  const laneY = foldedBottom + 26 + arcMaxUnits
  const pitch = (right - x0) / (lanes - 1)
  const laneNodes = Array.from({ length: lanes }, (_, lane) => ({ x: x0 + lane * pitch, y: laneY }))
  const laneLabels: TextAt[] = laneNodes
    .map((node, lane) => ({ node, lane }))
    .filter(({ lane }) => lane % 8 === 0 || lane === lanes - 1)
    .map(({ node, lane }) => ({ x: node.x, y: laneY + 15, text: String(lane), anchor: "middle" as const }))
  const slotsY = laneY + 58
  const slots: Rect = { x: x0, y: slotsY, width: right - x0, height: 14 }
  const kindsTop = slotsY + 56
  const values = model.values.slice(0, KIND_ROWS_MAX)
  const columns = wide && values.length > 3 ? 2 : 1
  const perColumn = Math.ceil(values.length / columns)
  const columnWidth = (right - x0) / columns
  const kinds = values.map((value, index) => {
    const column = Math.floor(index / perColumn)
    const row = index % perColumn
    const x = x0 + column * columnWidth
    const y = kindsTop + row * 15
    return {
      name: { x, y, text: value.name },
      kind: { x: x + (wide ? 64 : 70), y, text: kindName(value.kind) },
      dotsX: x + (wide ? 160 : 168),
      dotsY: y,
      pitchUnits: wide ? 3.6 : 3.4,
    }
  })
  const bandLabels: TextAt[] = [
    { x: MARGIN, y: 60, text: "warps" },
    { x: MARGIN, y: laneY - arcMaxUnits / 2, text: "shuffle" },
    { x: MARGIN, y: laneY + 2, text: "lanes" },
    { x: MARGIN, y: slotsY + 7, text: "lines" },
    { x: MARGIN, y: kindsTop, text: "kind" },
  ]
  const bottom = kinds.length > 0 ? kindsTop + (perColumn - 1) * 15 + 6 : kindsTop + 6
  return {
    level: {
      bandLabels,
      folded,
      foldedNote:
        folded.length > 0 ? { x: right, y: 60, text: `warps 1 to ${warps - 1}, folded`, anchor: "end" } : null,
      lanes: laneNodes,
      laneLabels,
      arcMaxUnits,
      slots,
      // Narrow, the sentence starts at the margin so it fits the plate.
      slotsText: { x: wide ? x0 : MARGIN, y: slotsY + 30, text: "" },
      stageText: { x: right, y: laneY - arcMaxUnits - 8, text: "", anchor: "end" },
      kinds,
      idleText: { x: x0, y: laneY - arcMaxUnits / 2, text: "" },
    },
    bottom,
  }
}

export function kindName(kind: string): string {
  return kind === "warp" ? "warp-uniform" : kind
}

function placeGrid(model: KernelModel, area: Rect): TileLevel["grid"] {
  const [gx, gy] = model.trace.launch.grid
  const pitchX = Math.min(40, gx > 1 ? (area.width - 12) / (gx - 1) : 0)
  const pitchY = Math.min(36, gy > 1 ? (area.height - 12) / (gy - 1) : 0)
  const spanX = pitchX * (gx - 1)
  const spanY = pitchY * (gy - 1)
  const left = area.x + (area.width - spanX) / 2
  const top = area.y + (area.height - spanY) / 2
  const nodes = model.ctas.map(cta => ({ x: left + cta[0] * pitchX, y: top + cta[1] * pitchY }))
  const links: [number, number][] = []
  model.ctas.forEach((cta, index) => {
    model.ctas.forEach((other, otherIndex) => {
      if (otherIndex <= index) return
      const dx = Math.abs(cta[0] - other[0])
      const dy = Math.abs(cta[1] - other[1])
      if (dx + dy === 1) links.push([index, otherIndex])
    })
  })
  return { area, nodes, links }
}

function placeTensors(model: KernelModel, area: Rect, layout: Layout): TensorPlace[] {
  const tensors = model.tensors
  const count = Math.max(1, tensors.length)
  const wide = layout === "wide"
  const perRow = wide ? count : Math.min(2, count)
  const gap = 16
  const columnWidth = (area.width - (perRow - 1) * gap) / perRow
  const boxHeight = wide ? 92 : 58
  const rowPitch = boxHeight + 46
  return tensors.map((tensor, index) => {
    const column = index % perRow
    const row = Math.floor(index / perRow)
    const shape = tensor.shape.length === 1 ? [1, tensor.shape[0]] : tensor.shape.slice(-2)
    const [rows, cols] = shape
    const x = area.x + column * (columnWidth + gap)
    const y = area.y + row * rowPitch
    const fit = Math.min(columnWidth / cols, boxHeight / rows)
    let sx = fit
    let sy = fit
    // A thin tensor gets at least two units per row (or column), so a one-row box stays visible.
    const thickRows = Math.min(boxHeight, Math.max(MIN_THICK_UNITS, Math.min(rows * 2, 24)))
    const thickCols = Math.min(columnWidth, Math.max(MIN_THICK_UNITS, Math.min(cols * 2, 24)))
    if (rows * sy < thickRows) {
      sx = columnWidth / cols
      sy = thickRows / rows
    }
    if (cols * sx < thickCols) {
      sy = boxHeight / rows
      sx = thickCols / cols
    }
    const width = cols * sx
    const height = rows * sy
    const cellCols = cellGroup(sx)
    const cellRows = cellGroup(sy)
    const cellNote = cellRows > 1 || cellCols > 1 ? `cell ${cellRows} x ${cellCols}` : "cell 1 x 1"
    return {
      name: tensor.name,
      rect: { x, y, width, height },
      sx,
      sy,
      cellRows,
      cellCols,
      label: {
        x,
        y: y + height + 15,
        text: `${tensor.name}${wide ? "  " : " "}${tensor.elem}[${tensor.shape.join(", ")}]`,
      },
      note: { x, y: y + height + 31, text: cellNote },
    }
  })
}

// Elements per drawn cell: the smallest power of two whose cells are at least a few units across.
function cellGroup(unitsPerElement: number): number {
  let group = 1
  while (group * unitsPerElement < TARGET_CELL_UNITS && group < 4096) group *= 2
  return group
}

// Buffers in rows along the byte ruler: two that share bytes never share a row.
function placeBuffers(model: KernelModel): { id: number; row: number; offset: number; bytes: number }[] {
  const rows: [number, number][][] = []
  return model.buffers.map(buffer => {
    const offset = buffer.offsetBytes ?? 0
    const end = offset + buffer.bytes
    let row = rows.findIndex(taken => taken.every(([from, to]) => end <= from || offset >= to))
    if (row < 0) {
      rows.push([])
      row = rows.length - 1
    }
    rows[row].push([offset, end])
    return { id: buffer.id, row, offset, bytes: buffer.bytes }
  })
}

/** `count` line slots spread over a region, at most `maxUnits` wide each. */
export function slotRects(region: Rect, count: number, maxUnits = 26): Rect[] {
  if (count <= 0) return []
  const gap = count > 24 ? 1.5 : 3
  const width = Math.min(maxUnits, (region.width - (count - 1) * gap) / count)
  return Array.from({ length: count }, (_, index) => ({
    x: region.x + index * (width + gap),
    y: region.y,
    width,
    height: region.height,
  }))
}

/** A box of a tensor in drawing units, clipped to the tensor. */
export function boxRect(place: TensorPlace, box: [number, number][]): Rect {
  const [rows, cols] = box.length === 1 ? [[0, 1], box[0]] : [box[box.length - 2], box[box.length - 1]]
  const right = place.rect.x + place.rect.width
  const bottom = place.rect.y + place.rect.height
  const width = Math.max(1, (cols[1] - cols[0]) * place.sx)
  const height = Math.max(1, (rows[1] - rows[0]) * place.sy)
  return {
    x: Math.min(place.rect.x + cols[0] * place.sx, right - width),
    y: Math.min(place.rect.y + rows[0] * place.sy, bottom - height),
    width,
    height,
  }
}

/** Points along an arc above the lane line from one lane to another, for packets and the path. */
export function arcPoints(from: { x: number; y: number }, to: { x: number; y: number }, maxUnits: number) {
  const span = Math.abs(to.x - from.x)
  const height = Math.min(maxUnits, 10 + span * 0.32)
  const points: { x: number; y: number }[] = []
  const samples = 16
  for (let index = 0; index <= samples; index += 1) {
    const t = index / samples
    points.push({ x: from.x + (to.x - from.x) * t, y: from.y - Math.sin(Math.PI * t) * height })
  }
  return points
}
