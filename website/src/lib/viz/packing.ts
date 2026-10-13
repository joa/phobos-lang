// The shared-memory packing plate: a trace's buffers as lifetime intervals (program order across,
// byte offset down) and its barriers, kept and elided, at their place in program order. Pure.

import type { KernelModel } from "./model"

export interface PackingFrame {
  left: number
  right: number
  top: number
  bottom: number
}

export interface PackedBuffer {
  id: number
  name: string
  x: number
  y: number
  width: number
  height: number
  offsetBytes: number
  bytes: number
  scratch: boolean
}

export interface PackedBarrier {
  x: number
  kept: boolean
}

export interface Packing {
  buffers: PackedBuffer[]
  barriers: PackedBarrier[]
  peakBytes: number
  /** Byte offsets that start a buffer, for the ruler's ticks. */
  offsets: number[]
}

export function packing(model: KernelModel, frame: PackingFrame): Packing {
  const ranges = model.buffers.map(buffer => model.liveRange(buffer))
  const barrierAt = model.barriers.map(barrier => model.barrierAt(barrier))
  const first = Math.min(...ranges.map(range => range[0]), ...barrierAt) - 1
  const last = Math.max(...ranges.map(range => range[1]), ...barrierAt) + 1
  const peakBytes = Math.max(1, model.sharedPeakBytes)
  const x = (order: number) => frame.left + ((order - first) / (last - first)) * (frame.right - frame.left)
  const y = (bytes: number) => frame.top + (bytes / peakBytes) * (frame.bottom - frame.top)
  const seen = new Map<string, number>()
  const buffers = model.buffers.map((buffer, index) => {
    const [from, to] = ranges[index]
    const offsetBytes = buffer.offsetBytes ?? 0
    const body = model.bodyOf(buffer.firstOp)
    const count = seen.get(buffer.name) ?? 0
    seen.set(buffer.name, count + 1)
    const name = body > 0 && count > 0 ? `${buffer.name}, remainder` : buffer.name
    return {
      id: buffer.id,
      name,
      x: x(from),
      y: y(offsetBytes),
      width: Math.max(2, x(to) - x(from)),
      height: Math.max(2, y(offsetBytes + buffer.bytes) - y(offsetBytes)),
      offsetBytes,
      bytes: buffer.bytes,
      scratch: buffer.scratch,
    }
  })
  const offsets = [...new Set([0, ...buffers.map(buffer => buffer.offsetBytes)])].sort((a, b) => a - b)
  return {
    buffers,
    barriers: model.barriers.map((barrier, index) => ({ x: x(barrierAt[index]), kept: barrier.kept })),
    peakBytes,
    offsets,
  }
}
