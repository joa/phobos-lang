import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { boxRect, slotRects, vizGeometry, type Rect } from "../src/lib/viz/geometry"
import { extraLines, formatBox, KernelModel, type LineFacts } from "../src/lib/viz/model"
import { shownSource } from "../src/lib/viz/shown"
import type { Trace } from "../src/lib/viz/trace"

const TRACES = join(import.meta.dirname, "../src/data/traces")
const files = readdirSync(TRACES).filter(file => file.endsWith(".json"))

function load(stem: string): KernelModel {
  return new KernelModel(JSON.parse(readFileSync(join(TRACES, `${stem}.json`), "utf8")) as Trace)
}

const gemm = load("gemm_fp32.sm_75")
const q8 = load("q8_matvec.sm_75")
const lanes = load("lanes_matvec.sm_75")
const saxpy = load("saxpy_fp32.sm_75")
const noAligned = load("gemm_fp32_no_aligned.sm_75")

describe("statement steps", () => {
  it("steps gemm through its k loop and its epilogue", () => {
    const statements = gemm.statements(0)
    expect(statements).toHaveLength(17)
    expect(statements[0].span.line).toBe(19)
    expect(statements.slice(0, 16).map(statement => statement.iter.kt)).toEqual(
      Array.from({ length: 16 }, (_, index) => index * 4)
    )
    expect(statements[16].span.line).toBe(26)
  })

  it("groups the ops of one statement into one step", () => {
    const part = q8.statements(0).find(statement => statement.span.line === 25)
    expect(part?.steps.map(step => step.op)).toEqual([34, 35])
    const line9 = lanes.statements(0)[0]
    expect(line9.steps.map(step => step.op)).toEqual([29, 31])
  })

  it("reads the right panel for a CTA and an iteration", () => {
    const cta = gemm.ctaIndex([1, 0, 0])
    const step = gemm.statements(cta).findIndex(statement => statement.iter.kt === 8)
    const state = gemm.stateAt(cta, step)
    const tensors = state.reads.filter(access => access.tensor)
    expect(tensors.map(access => formatBox(access.tensor ?? "", access.box))).toEqual([
      "A[32:64, 8:12]",
      "B[8:12, 0:32]",
    ])
  })

  it("alternates the GEMM's staging buffers per iteration under @pipeline", () => {
    const staged = (step: number) => gemm.stateAt(0, step).writes.map(access => access.buffer)
    expect(staged(0)).toEqual([0, 1])
    expect(staged(1)).toEqual([2, 3])
    expect(staged(2)).toEqual([0, 1])
    expect(gemm.stateAt(0, 0).barriers).toEqual([{ opId: 39, kept: true }])
    expect(gemm.ancestry(39)).toEqual([39, 41])
  })

  it("steps a butterfly one shuffle at a time", () => {
    const kinds = lanes
      .statements(0)
      .map(statement => statement.steps.map(step => step.lanes?.kind ?? "store").join("+"))
    expect(kinds).toEqual([
      "lane_load+lane_load",
      "lane_load+lane_load",
      "shuffle",
      "shuffle",
      "shuffle",
      "shuffle",
      "shuffle",
      "collective",
      "store",
    ])
    expect(lanes.shuffleStage(lanes.statements(0)[3])).toEqual({ index: 2, count: 5 })
    expect(lanes.stateAt(0, 2).lanes[0].event.partners?.slice(0, 3)).toEqual([16, 17, 18])
  })

  it("moves by a loop iteration and leaves the loop at its end", () => {
    expect(gemm.iterationStep(0, 0, 1)).toBe(1)
    expect(gemm.iterationStep(0, 15, 1)).toBe(16)
    expect(gemm.iterationStep(0, 16, -1)).toBe(15)
    expect(q8.statements(0)[q8.iterationStep(0, 1, 1)].iter.k).toBe(32)
  })

  it("marks only the lines past the fewest that hold the bytes as extra", () => {
    const facts = (used: number[], requestedBytes: number): LineFacts => ({
      lines: used.length,
      idealLines: Math.ceil(requestedBytes / 128),
      requestedBytes,
      touched: used.map((usedBytes, line) => ({ line, usedBytes })),
      laneLines: [],
      laneBytes: [],
    })
    // 8 half-used lines for 512 bytes: 4 needed, the last 4 extra.
    expect([...extraLines(facts([64, 64, 64, 64, 64, 64, 64, 64], 512))].sort()).toEqual([4, 5, 6, 7])
    // A misaligned run: the least used line is the extra one.
    expect([...extraLines(facts([32, 128, 128, 128, 96], 512))]).toEqual([0])
    expect(extraLines(facts([128, 128], 256)).size).toBe(0)
  })

  it("matches statements across a variant by text and iteration", () => {
    const key = gemm.statements(0)[3].key
    const matched = noAligned.findByKey(0, key)
    expect(matched).not.toBeNull()
    expect(noAligned.statements(0)[matched!].iter).toEqual(gemm.statements(0)[3].iter)
  })
})

describe("state", () => {
  it("lights the GEMM's staging buffers across its loop", () => {
    expect(gemm.stateAt(0, 0).liveBuffers).toEqual([0, 1, 2, 3])
  })

  it("lights only what a ragged loop's first body allocated", () => {
    const step = q8.statements(0).findIndex(statement => statement.steps.some(entry => entry.op === 30))
    expect(q8.stateAt(0, step).liveBuffers).toEqual([0, 1, 2])
    for (let index = 0; index < q8.stepCount(0); index += 1) {
      const live = q8.stateAt(0, index).liveBuffers
      for (const remainder of [7, 8, 9, 10]) expect(live).not.toContain(remainder)
    }
    for (let index = 0; index < noAligned.stepCount(0); index += 1) {
      expect(noAligned.stateAt(0, index).liveBuffers).not.toContain(3)
    }
  })

  it("seeks to the same state forward and backward", () => {
    for (const model of [gemm, q8, noAligned]) {
      const count = model.stepCount(0)
      const forward = Array.from({ length: count }, (_, index) => model.stateAt(0, index).written)
      for (let index = count - 1; index >= 0; index -= 1)
        expect(model.stateAt(0, index).written).toEqual(forward[index])
    }
    expect(gemm.stateAt(0, 16).written.C).toEqual([
      [
        [0, 32],
        [0, 32],
      ],
    ])
    expect(gemm.stateAt(0, 15).written.C).toBeUndefined()
  })

  it("badges each named value with its kind", () => {
    const kinds = Object.fromEntries(lanes.values.map(value => [value.name, value.kind]))
    expect(kinds).toEqual({ row: "warp", acc: "varying", at: "varying", total: "warp" })
  })
})

describe("lines touched by warp 0", () => {
  it("counts a column panel's 32 rows as 32 lines", () => {
    const lines = gemm.stateAt(0, 0).mapped?.lines
    expect(lines?.lines).toBe(32)
    expect(lines?.idealLines).toBe(4)
    expect(lines?.requestedBytes).toBe(512)
  })

  it("counts a row panel as the lines it fills", () => {
    const facts = gemm.mappedAccess({
      op: 39,
      cta: [0, 0, 0],
      iter: { kt: 0 },
      reads: [
        {
          tensor: "B",
          box: [
            [0, 4],
            [0, 32],
          ],
        },
      ],
      map: { shape: [4, 32], width: 4, threads: 256, vectorBytes: 16 },
    })
    expect(facts?.lines?.lines).toBe(4)
    expect(facts?.lines?.idealLines).toBe(4)
  })

  it("reads lane runs directly", () => {
    const event = lanes.stateAt(0, 0).lanes[0].event
    const lines = lanes.laneRunLines(event)
    expect(lines?.lines).toBe(4)
    expect(lines?.requestedBytes).toBe(512)
    expect(lines?.laneLines[9]).toEqual([1])
    expect(lines?.laneBytes[9]).toEqual([16, 32])
  })
})

describe("readout", () => {
  it("describes a copy into shared memory with its bytes and threads", () => {
    expect(saxpy.readout(saxpy.ctaIndex([3, 0, 0]), 0)).toBe(
      "CTA 3: copies x[48:60] (4 masked off) into shared tile0 (bytes 0 to 63); then copies y[48:60] (4 masked off) into shared tile1 (bytes 64 to 127); then computes elementwise from shared tile0 (bytes 0 to 63) and shared tile1 (bytes 64 to 127) into out[48:60] (4 masked off); 16 of 256 threads, 1 f32 each. 2 barriers follow its ops: 0 kept, 2 elided."
    )
  })

  it("describes the GEMM's k step and its coalescing", () => {
    expect(gemm.readout(gemm.ctaIndex([1, 1, 0]), 2)).toBe(
      "CTA (1, 1), kt = 8: stages A[32:64, 8:12] and B[8:12, 32:64] in shared tile0 (bytes 0 to 511) and shared tile1 (bytes 512 to 1023) for the GEMM's k step on fma; A: 32 of 256 threads, 4 f32 each, 16-byte vectors. Warp 0 touches 32 memory lines of 128 bytes for 512 bytes, where 4 would hold them; the 28 extra lines are drawn dashed in orange. A barrier follows."
    )
  })

  it("describes lane loads and the butterfly", () => {
    expect(lanes.readout(0, 2)).toBe("CTA 0: shuffle 1 of 5: each lane of warp 0 takes the value of lane xor 16.")
    expect(lanes.readout(0, 6)).toBe("CTA 0: shuffle 5 of 5: each lane of warp 0 takes the value of lane xor 1.")
    expect(lanes.readout(0, 7)).toBe(
      "CTA 0: the lane collective is done: after 5 xor shuffle stages (16, 8, 4, 2, 1), every one of warp 0's 32 lanes holds the result."
    )
    expect(lanes.readout(1, 0)).toContain("warp 0's 32 lanes each load 4 f32 of W[4, 0:128] as one 16-byte vector")
  })

  it("names a barrier kept or elided", () => {
    expect(noAligned.readout(0, 1)).toMatch(/The barrier after it is elided\.$/)
    expect(noAligned.readout(0, 2)).toMatch(/A barrier follows\.$/)
  })
})

function inside(inner: Rect, outer: Rect): boolean {
  return (
    inner.x >= outer.x - 0.01 &&
    inner.y >= outer.y - 0.01 &&
    inner.x + inner.width <= outer.x + outer.width + 0.01 &&
    inner.y + inner.height <= outer.y + outer.height + 0.01
  )
}

function overlaps(a: Rect, b: Rect): boolean {
  const e = 0.01
  return a.x < b.x + b.width - e && b.x < a.x + a.width - e && a.y < b.y + b.height - e && b.y < a.y + a.height - e
}

describe("geometry", () => {
  it.each(files)("keeps %s on its plate in both layouts", file => {
    const model = load(file.replace(/\.json$/, ""))
    for (const layout of ["wide", "narrow"] as const) {
      const geometry = vizGeometry(model, layout)
      const plate = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }
      const tensors = geometry.tile.tensors.map(place => place.rect)
      for (const rect of tensors) expect(inside(rect, plate)).toBe(true)
      tensors.forEach((rect, index) =>
        tensors.slice(index + 1).forEach(other => expect(overlaps(rect, other)).toBe(false))
      )
      for (const buffer of geometry.tile.shared.buffers) expect(inside(buffer.rect, plate)).toBe(true)
      for (const dot of geometry.tile.threads.dots) expect(dot.x).toBeLessThan(geometry.widthUnits)
      for (const lane of geometry.lane.lanes) expect(lane.x).toBeLessThanOrEqual(geometry.widthUnits - 14)
      expect(geometry.heightUnits).toBeLessThan(layout === "wide" ? 480 : 640)
      for (const statement of model.statements(0)) {
        for (const step of statement.steps) {
          for (const access of [...(step.reads ?? []), ...(step.writes ?? [])]) {
            const place = geometry.tile.tensors.find(candidate => candidate.name === access.tensor)
            if (place) expect(inside(boxRect(place, access.box), place.rect)).toBe(true)
          }
        }
      }
    }
  })

  it("never lets two buffers that share bytes share a ruler row", () => {
    const geometry = vizGeometry(q8, "wide")
    const rects = geometry.tile.shared.buffers.map(buffer => buffer.rect)
    rects.forEach((rect, index) => rects.slice(index + 1).forEach(other => expect(overlaps(rect, other)).toBe(false)))
  })

  it("spreads line slots without overlap", () => {
    const slots = slotRects({ x: 0, y: 0, width: 400, height: 14 }, 32)
    slots.forEach((slot, index) => slots.slice(index + 1).forEach(other => expect(overlaps(slot, other)).toBe(false)))
    expect(slots[31].x + slots[31].width).toBeLessThanOrEqual(400.01)
  })
})

describe("shown source", () => {
  it("hides @cluster lines and keeps the original numbers", () => {
    const shown = shownSource("a\n@cluster(X in [1, 2])\nb\r\n")
    expect(shown.code).toBe("a\nb")
    expect(shown.lineMap).toEqual([1, 3])
    expect(shown.hiddenLines).toEqual([2])
  })
})
