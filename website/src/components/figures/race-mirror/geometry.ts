// Geometry of the race-mirror figure, pure so the tests can check it. The kernel is the races page's:
// 64 threads in two warps, each writing its own element of a shared tile t[0, 64] and then reading
// its mirror's, t[0, 63 - ctx.thread]. Wide lays the threads, the tile and the reads out as rows,
// narrow as columns.

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { Polyline, type Point } from "../../../lib/figure/polyline"

export const THREADS = 64
export const WARP = 32

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

export interface RaceMirrorGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  // Indexed by thread, and for the tile by element.
  threads: Rect[]
  tile: Rect[]
  reads: Rect[]
  // Thread t's store into t[t].
  writes: Polyline[]
  // Thread t's load of t[63 - t] into its own register.
  loads: Polyline[]
  // The barrier, one segment per warp, between the tile and the reads.
  barrier: { from: Point; to: Point }[]
  barrierLabel: TextAt
  rowLabels: TextAt[]
  warpLabels: TextAt[]
  // The pair the race check names: thread 63's store and thread 0's load.
  writerLabel: TextAt
  readerLabel: TextAt
}

export function warpOf(thread: number): number {
  return Math.floor(thread / WARP)
}

// The element thread t reads.
export function mirrorOf(thread: number): number {
  return THREADS - 1 - thread
}

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  // Where element 0 sits along the axis, the pitch between elements, the extra gap between warps.
  startUnits: number
  pitchUnits: number
  warpGapUnits: number
  // Each row's (wide) or column's (narrow) offset across the axis and its depth.
  threads: [number, number]
  tile: [number, number]
  reads: [number, number]
  barrierAt: number
}

const CELL_FRACTION = 0.82
const BARRIER_OVERHANG_UNITS = 4

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: VIEW_WIDTH_UNITS.wide,
    heightUnits: 290,
    plateLabel: "FIG / RACE / A MIRRORED READ",
    startUnits: 150,
    pitchUnits: 8,
    warpGapUnits: 16,
    threads: [92, 10],
    tile: [150, 12],
    reads: [240, 10],
    barrierAt: 182,
  },
  narrow: {
    widthUnits: VIEW_WIDTH_UNITS.narrow,
    heightUnits: 568,
    plateLabel: "FIG / RACE",
    startUnits: 98,
    pitchUnits: 6.5,
    warpGapUnits: 12,
    threads: [104, 16],
    tile: [194, 18],
    reads: [324, 16],
    barrierAt: 234,
  },
}

export function raceMirrorGeometry(layout: Layout): RaceMirrorGeometry {
  const spec = SPECS[layout]
  const wide = layout === "wide"
  const along = (index: number) => spec.startUnits + index * spec.pitchUnits + (warpOf(index) > 0 ? spec.warpGapUnits : 0)
  const cellUnits = spec.pitchUnits * CELL_FRACTION
  // A rect at an element's place along the axis, in a row or column.
  const at = (index: number, [offset, depth]: [number, number]): Rect =>
    wide
      ? { x: along(index), y: offset, width: cellUnits, height: depth }
      : { x: offset, y: along(index), width: depth, height: cellUnits }
  const row = (band: [number, number]) => Array.from({ length: THREADS }, (_, index) => at(index, band))
  const threads = row(spec.threads)
  const tile = row(spec.tile)
  const reads = row(spec.reads)

  // Wires leave the far side of one band and reach the near side of the next.
  const far = (rect: Rect): Point =>
    wide ? { x: rect.x + rect.width / 2, y: rect.y + rect.height } : { x: rect.x + rect.width, y: rect.y + rect.height / 2 }
  const near = (rect: Rect): Point =>
    wide ? { x: rect.x + rect.width / 2, y: rect.y } : { x: rect.x, y: rect.y + rect.height / 2 }
  const writes = threads.map((thread, index) => new Polyline([far(thread), near(tile[index])]))
  const loads = reads.map((read, thread) => new Polyline([far(tile[mirrorOf(thread)]), near(read)]))

  const barrier = [0, 1].map(warp => {
    const first = along(warp * WARP) - BARRIER_OVERHANG_UNITS
    const last = along(warp * WARP + WARP - 1) + cellUnits + BARRIER_OVERHANG_UNITS
    return wide
      ? { from: { x: first, y: spec.barrierAt }, to: { x: last, y: spec.barrierAt } }
      : { from: { x: spec.barrierAt, y: first }, to: { x: spec.barrierAt, y: last } }
  })
  const end = along(THREADS - 1) + cellUnits
  const warpMid = (warp: number) => (along(warp * WARP) + along(warp * WARP + WARP - 1) + cellUnits) / 2
  const center = ([offset, depth]: [number, number]) => offset + depth / 2

  if (wide) {
    const labelX = 24
    return {
      widthUnits: spec.widthUnits,
      heightUnits: spec.heightUnits,
      plateLabel: spec.plateLabel,
      threads,
      tile,
      reads,
      writes,
      loads,
      barrier,
      barrierLabel: { x: labelX, y: spec.barrierAt, text: "barrier()" },
      rowLabels: [
        { x: labelX, y: center(spec.threads), text: "threads" },
        { x: labelX, y: center(spec.tile), text: "shared t" },
        { x: labelX, y: center(spec.reads), text: "reads" },
      ],
      warpLabels: [0, 1].map(warp => ({ x: warpMid(warp), y: 68, text: `warp ${warp}`, anchor: "middle" as const })),
      writerLabel: { x: end, y: 68, text: "t63 writes", anchor: "end" },
      readerLabel: { x: spec.startUnits, y: spec.reads[0] + spec.reads[1] + 22, text: "t0 reads" },
    }
  }
  const headerY = 66
  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    threads,
    tile,
    reads,
    writes,
    loads,
    barrier,
    barrierLabel: { x: spec.barrierAt, y: end + 22, text: "barrier()", anchor: "middle" },
    rowLabels: [
      { x: center(spec.threads), y: headerY, text: "threads", anchor: "middle" },
      { x: center(spec.tile), y: headerY, text: "shared t", anchor: "middle" },
      { x: center(spec.reads), y: headerY, text: "reads", anchor: "middle" },
    ],
    warpLabels: [0, 1].map(warp => ({ x: 16, y: warpMid(warp), text: `warp ${warp}` })),
    writerLabel: { x: 16, y: along(THREADS - 1) + cellUnits / 2, text: "t63 writes" },
    readerLabel: { x: center(spec.reads), y: headerY + 18, text: "t0 reads", anchor: "middle" },
  }
}
