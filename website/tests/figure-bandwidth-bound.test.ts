import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import BandwidthBoundFigure, { smsForDelivery } from "../src/components/figures/bandwidth-bound/bandwidth-bound.figure"
import { bandwidthGeometry } from "../src/components/figures/bandwidth-bound/geometry"
import {
  BALANCE_OPS_PER_BYTE,
  CARD,
  DECODE_BUSY_SMS,
  DECODE_OPS_PER_BYTE,
  PROMPT_OPS_PER_BYTE,
  Q8_0_BLOCK_BYTES,
  Q8_0_BLOCK_WEIGHTS,
  READOUTS,
} from "../src/components/figures/bandwidth-bound/model"
import { inside, overlaps, textRect } from "../src/components/inference/figure-geometry"
import { loadRuns } from "../src/lib/bench/load"

describe("bandwidth-bound arithmetic", () => {
  it("uses the Q8_0 block phobos-gguf defines", () => {
    const source = readFileSync(resolve(process.cwd(), "../phobos-gguf/src/quant/q8_0.rs"), "utf8")
    expect(source).toMatch(new RegExp(`pub const BLOCK: usize = ${Q8_0_BLOCK_WEIGHTS};`))
    expect(source).toMatch(new RegExp(`const BLOCK_BYTES: usize = ${Q8_0_BLOCK_BYTES};`))
  })

  it("describes the card the results were measured on", () => {
    for (const run of loadRuns()) expect(run.card).toBe(CARD.name)
  })

  it("puts decode below the balance point and a prompt above it", () => {
    expect(DECODE_OPS_PER_BYTE).toBeCloseTo(1.88, 2)
    expect(PROMPT_OPS_PER_BYTE).toBeCloseTo(964, 0)
    expect(BALANCE_OPS_PER_BYTE).toBeGreaterThan(20)
    expect(BALANCE_OPS_PER_BYTE).toBeLessThan(25)
    expect(DECODE_BUSY_SMS).toBe(4)
  })

  it("lights the same distinct SMs for a delivery on every run", () => {
    for (let delivery = 0; delivery < 50; delivery += 1) {
      const sms = smsForDelivery(delivery)
      expect(new Set(sms).size).toBe(DECODE_BUSY_SMS)
      expect(smsForDelivery(delivery)).toEqual(sms)
      for (const sm of sms) expect(sm).toBeLessThan(CARD.smCount)
    }
  })

  it("writes ASCII readouts", () => {
    for (const text of Object.values(READOUTS)) expect(text).toMatch(/^[\x20-\x7e]+$/)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`bandwidth-bound geometry, ${layout}`, () => {
    const geometry = bandwidthGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps the blocks and labels on the plate, apart", () => {
      const blocks = [geometry.weights.box, geometry.sms.box]
      for (const block of blocks) expect(inside(block, plateRect)).toBe(true)
      expect(overlaps(blocks[0], blocks[1])).toBe(false)
      for (const cell of geometry.sms.cells) expect(inside(cell, geometry.sms.box)).toBe(true)
      const labels = [
        geometry.weights.label,
        geometry.bus.label,
        geometry.sms.label,
        geometry.axis.balanceLabel,
        geometry.axis.memoryLabel,
        geometry.axis.computeLabel,
        geometry.axis.title,
        geometry.pointLabel("decode"),
        geometry.pointLabel("prompt"),
        ...geometry.axis.ticks.map(tick => tick.label),
      ].map(textRect)
      for (const rect of labels) expect(inside(rect, plateRect), JSON.stringify(rect)).toBe(true)
      for (let a = 0; a < labels.length; a += 1) {
        for (let b = a + 1; b < labels.length; b += 1) {
          // The two point labels never show at once.
          if (a === labels.length - 6 && b === labels.length - 5) continue
          expect(overlaps(labels[a], labels[b]), `${a} ${b}`).toBe(false)
        }
      }
    })

    it("renders statically in both phases", () => {
      for (const phase of ["decode", "prompt"]) {
        const render = BandwidthBoundFigure.renderStatic(layout, { phase })
        expect(render.viewBox).toBe(`0 0 ${geometry.widthUnits} ${geometry.heightUnits}`)
        expect(render.readout).toBe(READOUTS[phase as "decode" | "prompt"])
        expect(render.drawing).not.toMatch(/NaN/)
      }
    })
  })
}
