import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import ExpertCacheFigure, { readoutFor } from "../src/components/figures/expert-cache/expert-cache.figure"
import { expertCacheGeometry } from "../src/components/figures/expert-cache/geometry"
import {
  EXPERT_COUNT,
  PICKS_PER_TOKEN,
  RouterSimulation,
  SLOT_COUNT,
  type TokenResult,
} from "../src/components/figures/expert-cache/simulation"
import { boundsOf, inside, overlaps, textRect } from "../src/components/inference/figure-geometry"

describe("expert-cache simulation", () => {
  it("routes as many experts as phobos-gguf does", () => {
    const source = readFileSync(resolve(process.cwd(), "../phobos-gguf/src/backend/device/kernels/moe.rs"), "utf8")
    expect(source).toContain(`pub(crate) const MOE_USED: usize = ${PICKS_PER_TOKEN};`)
  })

  it("picks distinct experts and keeps the cache consistent", () => {
    const simulation = new RouterSimulation({ path: "prompt" })
    for (let token = 0; token < 500; token += 1) {
      const result = simulation.step()
      const experts = result.picks.map(pick => pick.expert)
      expect(new Set(experts).size).toBe(PICKS_PER_TOKEN)
      for (const expert of experts) expect(expert).toBeLessThan(EXPERT_COUNT)
      // A prompt copies every miss, so after a token every pick is cached.
      const cached = new Set(simulation.slots)
      for (const expert of experts) expect(cached.has(expert)).toBe(true)
    }
    const filled = simulation.slots.filter(slot => slot !== null)
    expect(filled).toHaveLength(SLOT_COUNT)
    expect(new Set(filled).size).toBe(SLOT_COUNT)
  })

  it("copies one miss a decode step", () => {
    const simulation = new RouterSimulation({ path: "decode" })
    for (let token = 0; token < 200; token += 1) {
      const result = simulation.step()
      expect(result.copies.length).toBe(result.hitCount === PICKS_PER_TOKEN ? 0 : 1)
      for (const copy of result.copies) expect(copy.isRefill).toBe(true)
    }
  })

  it("evicts the least recently used expert", () => {
    const simulation = new RouterSimulation({ path: "prompt", warmupTokens: 0, slotCount: 8 })
    const first = simulation.step()
    const second = simulation.step()
    const evicted = second.copies.length
    const kept = first.picks.map(pick => pick.expert).filter(expert => simulation.slots.includes(expert))
    // Only experts of the first token that the second did not touch can leave, oldest first.
    expect(kept.length).toBeGreaterThanOrEqual(PICKS_PER_TOKEN - evicted)
  })

  it("is deterministic and hits most of the time", () => {
    const run = () => {
      const simulation = new RouterSimulation()
      let last: TokenResult | null = null
      for (let token = 0; token < 3000; token += 1) last = simulation.step()
      return last!
    }
    const [a, b] = [run(), run()]
    expect(a).toEqual(b)
    expect(a.hitRate).toBeGreaterThan(0.6)
    expect(a.hitRate).toBeLessThan(0.95)
  })

  it("writes ASCII readouts", () => {
    const simulation = new RouterSimulation()
    for (let token = 0; token < 20; token += 1) expect(readoutFor(simulation.step())).toMatch(/^[\x20-\x7e]+$/)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`expert-cache geometry, ${layout}`, () => {
    const geometry = expertCacheGeometry(layout)
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("keeps the parts on the plate and apart", () => {
      const cache = boundsOf(geometry.cache.slots)
      for (const part of [geometry.experts.bounds, geometry.cpu.box, geometry.gpu.frame]) {
        expect(inside(part, plateRect)).toBe(true)
      }
      expect(inside(cache, geometry.gpu.frame)).toBe(true)
      expect(inside(geometry.rest.box, geometry.gpu.frame)).toBe(true)
      expect(overlaps(cache, geometry.rest.box)).toBe(false)
      expect(overlaps(geometry.experts.bounds, geometry.cpu.box)).toBe(false)
      expect(overlaps(geometry.cpu.box, geometry.gpu.frame)).toBe(false)
      const labels = [
        geometry.experts.label,
        geometry.link.label,
        geometry.cpu.label,
        geometry.cpu.note,
        geometry.gpu.label,
        geometry.cache.label,
        geometry.rest.label,
      ].map(textRect)
      for (const rect of labels) expect(inside(rect, plateRect), JSON.stringify(rect)).toBe(true)
      for (let a = 0; a < labels.length; a += 1) {
        for (let b = a + 1; b < labels.length; b += 1) expect(overlaps(labels[a], labels[b]), `${a} ${b}`).toBe(false)
      }
      expect(overlaps(labels[0], geometry.experts.bounds)).toBe(false)
      expect(overlaps(labels[5], cache)).toBe(false)
    })

    it("renders statically on a settled token", () => {
      for (const path of ["decode", "prompt"]) {
        const render = ExpertCacheFigure.renderStatic(layout, { path })
        expect(render.readout).toMatch(/^Token 1: \d of 8 picks/)
        expect(render.drawing).not.toMatch(/NaN/)
      }
    })
  })
}
