// The passes station: shared-memory placement and barrier elision, played over program order. A cursor
// walks the ops; each buffer drops into the byte strip at the offset the planner gave it when its
// first op is reached and fades to an outline once its last op is passed, so c_old landing on the
// bytes a and b held is the reuse. Barriers light where they are kept and fade where they are elided.
// It is the build without @aligned, the one that stages its tiles; the plate says so.

import { stagedPasses, writtenPasses } from "./data"
import { fill } from "./source"
import { FRAMES, round, target, text, type Layout, type Shape, type StationGeometry, type Target, type Tone } from "./types"

export const SWEEP_START_MS = 700
export const SWEEP_MS = 4600
const BYTE_POINTS = 440

export interface PassesTiming {
  firstOp: number
  lastOp: number
  /** Per buffer, when it drops into the strip and when it retires (null if it lives to the end). */
  buffers: { dropMs: number; retireMs: number | null }[]
  /** Frame units of the timeline's ends. */
  left: number
  right: number
}

export function opX(op: number, timing: PassesTiming): number {
  return timing.left + ((op - timing.firstOp) / (timing.lastOp - timing.firstOp)) * (timing.right - timing.left)
}

export function opTimeMs(op: number, timing: PassesTiming): number {
  return SWEEP_START_MS + ((op - timing.firstOp) / (timing.lastOp - timing.firstOp)) * SWEEP_MS
}

export function cursorX(stationMs: number, timing: PassesTiming): number {
  const t = Math.min(1, Math.max(0, (stationMs - SWEEP_START_MS) / SWEEP_MS))
  return timing.left + t * (timing.right - timing.left)
}

const OP_TONES: Record<string, Tone> = {
  materialize: "shared",
  alloc: "shared",
  fill: "shared",
  dot_into: "reg",
  scaled_add: "reg",
  slice: "global",
  assume_align: "global",
}

export function passesStation(layout: Layout): StationGeometry & { timing: PassesTiming } {
  const frame = FRAMES[layout]
  const wide = layout === "wide"
  const view = stagedPasses()
  const written = writtenPasses()
  const firstOp = view.ops[0].id
  const lastOp = view.ops[view.ops.length - 1].id
  const left = wide ? 150 : 24
  const right = wide ? 1080 : 376
  const timing: PassesTiming = { firstOp, lastOp, left, right, buffers: [] }
  timing.buffers = view.buffers.map(buffer => ({
    dropMs: opTimeMs(buffer.firstOp, timing),
    retireMs: buffer.lastOp < lastOp ? opTimeMs(buffer.lastOp + 1, timing) : null,
  }))
  const trackY = wide ? 214 : 236
  const stripTop = wide ? 340 : 360
  const stripHeight = wide ? 150 : 150
  const byteX = (bytes: number) => left + (bytes / view.peakBytes) * (right - left)
  const shapes: Shape[] = []
  const targets: Target[] = []

  // What this build is, and the kernel as written beside it.
  const variant = view.variant?.drop ? `without ${view.variant.drop}` : "as written"
  const head = `${variant}: ${view.buffers.length} buffers, ${view.peakBytes} B peak, ${view.barrierCounts.kept + view.barrierCounts.elided} barriers`
  const asWritten = `as written: ${written.buffers.length} buffers, ${written.peakBytes} B, ${written.barrierCounts.kept} barrier${written.barrierCounts.kept === 1 ? "" : "s"}`
  if (wide) {
    shapes.push(text(left, 92, head, "pp-label t-shared is-strong pp-in", { delayMs: 60 }))
    shapes.push(text(right, 92, asWritten, "pp-label is-faint pp-in", { anchor: "end", delayMs: 120 }))
  } else {
    shapes.push(text(left, 74, `${variant}`, "pp-label t-shared is-strong pp-in", { delayMs: 60 }))
    shapes.push(text(left, 92, `${view.buffers.length} buffers, ${view.peakBytes} B, ${view.barrierCounts.kept + view.barrierCounts.elided} barriers`, "pp-label pp-in", { delayMs: 80 }))
    shapes.push(text(left, 110, asWritten, "pp-label is-faint pp-in", { delayMs: 120 }))
  }

  // Program order: the track, the loop's span, the cursor.
  shapes.push(text(left, trackY - (wide ? 52 : 48), "program order", "pp-label pp-in", { delayMs: 140 }))
  shapes.push({
    tag: "path",
    attrs: { d: `M${left} ${trackY}H${right}`, pathLength: 1, class: "pp-wire pp-draw" },
    delayMs: 160,
  })
  for (const op of view.ops) {
    const x = round(opX(op.id, timing))
    const tone = OP_TONES[op.kind] ?? (op.parent !== undefined ? "hi" : "dust")
    const tall = tone === "shared" || tone === "reg"
    shapes.push({
      tag: "path",
      attrs: { d: `M${x} ${trackY - (tall ? 6 : 3)}V${trackY + (tall ? 6 : 3)}`, class: `pp-op pp-in t-${tone}` },
      delayMs: 180 + (op.id - firstOp) * 8,
    })
  }
  const loop = view.loops[0]
  if (loop) {
    const x0 = round(opX(loop.firstOp, timing))
    const x1 = round(opX(loop.id, timing))
    const y = trackY - 24
    shapes.push({
      tag: "path",
      attrs: { d: `M${x0} ${y + 6}V${y}H${x1}V${y + 6}`, pathLength: 1, class: "pp-wire pp-draw t-reg" },
      delayMs: 260,
    })
    const label = wide ? `for ${loop.var}, ${loop.trips} trips: body and remainder` : `for ${loop.var}, ${loop.trips} trips`
    shapes.push(text((x0 + x1) / 2, y - 8, label, "pp-label t-reg pp-in", { anchor: "middle", delayMs: 300 }))
  }

  for (const barrier of view.barriers) {
    const x = round(opX(barrier.afterOp + 0.5, timing))
    const atMs = opTimeMs(barrier.afterOp + 0.5, timing)
    shapes.push({
      tag: "path",
      attrs: { d: `M${x} ${trackY - 13}V${trackY + 13}`, class: `pp-barrier ${barrier.kept ? "is-kept" : "is-elided"}` },
      delayMs: atMs,
    })
  }
  const legendY = trackY + (wide ? 40 : 36)
  const kept = view.barrierCounts.kept
  const elided = view.barrierCounts.elided
  shapes.push({ tag: "path", attrs: { d: `M${left} ${legendY - 13}V${legendY + 1}`, class: "pp-barrier is-kept is-static" }, delayMs: 200 })
  shapes.push(text(left + 9, legendY, `${kept} barriers kept`, "pp-label pp-in", { delayMs: 200 }))
  const elidedX = left + (wide ? 190 : 170)
  shapes.push({ tag: "path", attrs: { d: `M${elidedX} ${legendY - 13}V${legendY + 1}`, class: "pp-barrier is-elided is-static" }, delayMs: 200 })
  shapes.push(text(elidedX + 9, legendY, `${elided} elided`, "pp-label pp-in", { delayMs: 200 }))

  shapes.push({
    tag: "path",
    attrs: { d: `M${left} ${trackY - 34}V${stripTop + stripHeight + 6}`, class: "pp-cursor", style: `--sweep: ${round(right - left)}px` },
    delayMs: SWEEP_START_MS,
  })

  // The byte strip.
  shapes.push(text(left, stripTop - 14, wide ? "shared memory, by byte offset" : "shared memory", "pp-label t-shared pp-in", { delayMs: 200 }))
  shapes.push({
    tag: "rect",
    attrs: { x: left, y: stripTop, width: round(right - left), height: stripHeight, class: "pp-strip pp-in" },
    delayMs: 220,
  })
  const ticks = wide ? [...new Set(view.buffers.map(buffer => buffer.offsetBytes).concat(5120, view.peakBytes))] : [0, view.peakBytes / 2, view.peakBytes]
  for (const bytes of ticks.sort((a, b) => a - b)) {
    const x = round(byteX(bytes))
    shapes.push({ tag: "path", attrs: { d: `M${x} ${stripTop + stripHeight}v6`, class: "pp-tick pp-in" }, delayMs: 240 })
    const label = bytes === view.peakBytes ? `${bytes} B` : String(bytes)
    shapes.push(text(x, stripTop + stripHeight + 22, label, "pp-label pp-in", { anchor: bytes === view.peakBytes ? "end" : bytes === 0 ? "start" : "middle", delayMs: 240 }))
  }

  const seen = new Set<string>()
  for (const buffer of view.buffers) {
    const x = byteX(buffer.offsetBytes)
    const width = byteX(buffer.offsetBytes + buffer.bytes) - x
    const retires = buffer.lastOp < lastOp
    const dropMs = opTimeMs(buffer.firstOp, timing)
    const retireMs = opTimeMs(buffer.lastOp + 1, timing)
    shapes.push({
      tag: "rect",
      attrs: {
        x: round(x + 1),
        y: stripTop + 1,
        width: round(width - 2),
        height: stripHeight - 2,
        class: `pp-buf${retires ? " is-retiring" : ""}`,
      },
      delayMs: dropMs,
      laterMs: retires ? retireMs : undefined,
    })
    const first = !seen.has(buffer.name)
    seen.add(buffer.name)
    if (first) {
      const shape = buffer.shape?.join("x") ?? ""
      const label = wide && width > 300 ? `${buffer.name}  ${shape} ${buffer.elem ?? ""}, ${buffer.bytes} B` : buffer.name
      shapes.push({
        ...text(x + width / 2, stripTop + 18, label, `pp-label pp-buf-label${retires ? " is-retiring" : ""}`, { anchor: "middle" }),
        delayMs: dropMs + 200,
        laterMs: retires ? retireMs : undefined,
      })
    }
  }
  const reuse = view.buffers.find(buffer => buffer.name === "c_old")
  if (reuse) {
    const at = opTimeMs(reuse.firstOp, timing) + 250
    const x = byteX(reuse.offsetBytes + reuse.bytes / 2)
    shapes.push(text(x, stripTop - 14, wide ? "c_old takes the bytes a and b held" : "c_old reuses a, b", "pp-label t-shared pp-in", { anchor: "middle", delayMs: at }))
  }

  // The cloud: the bytes, a point per few of them, inside each buffer; they fall in with it.
  const totalBytes = view.buffers.reduce((sum, buffer) => sum + buffer.bytes, 0)
  if (wide) {
    const note = `a buffer drops in at its first op, fades after its last; a point is about ${Math.round(totalBytes / BYTE_POINTS)} bytes`
    shapes.push(text(left, stripTop + stripHeight + 56, note, "pp-label is-faint pp-in", { delayMs: 400 }))
  }
  view.buffers.forEach((buffer, index) => {
    const x = byteX(buffer.offsetBytes) + 3
    const width = byteX(buffer.offsetBytes + buffer.bytes) - x - 3
    const count = Math.round((buffer.bytes / totalBytes) * BYTE_POINTS)
    const rowsCount = Math.max(1, Math.round(Math.sqrt((count * (stripHeight - 34)) / Math.max(1, width))))
    const cols = Math.ceil(count / rowsCount)
    for (let k = 0; k < count; k += 1) {
      const col = k % cols
      const row = Math.floor(k / cols)
      const tone: Tone = buffer.name === "acc" ? "reg" : "shared"
      targets.push({
        ...target(x + ((col + 0.5) / cols) * width, stripTop + 28 + ((row + 0.5) / rowsCount) * (stripHeight - 34), tone, k % 5 === 0 ? 2 : 1, 0, 0.8),
        group: index,
      })
    }
  })

  return { frame, shapes, targets: fill(targets, frame.width, frame.height), streams: [], timing }
}
