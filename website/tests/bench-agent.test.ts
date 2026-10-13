// The replay under memory pressure, read from results/agent-squeeze-release.json, against BENCHMARKS.md's
// table of it, and the committed copy the elastic-cache figure draws from. After a new run:
//
//   npx vitest run tests/bench-agent.test.ts -u

import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { loadSqueezeRelease, SQUEEZE_RELEASE, squeezeResizes } from "../src/lib/bench/agent"
import { loadRuns, repoRoot } from "../src/lib/bench/load"

const data = loadSqueezeRelease()
const benchmarks = readFileSync(resolve(repoRoot(), "BENCHMARKS.md"), "utf8").replace(/\r\n?/g, "\n")

// "| llama.cpp CUDA | 131.7 s |      31.1 t/s |  9.0 t/s | 31.2 t/s |", in the section on the squeeze
// that ends.
function benchmarksRow(engine: string): number[] {
  const section = benchmarks.slice(benchmarks.indexOf("The other program can also close again"))
  const line = section.split("\n").find(row => row.startsWith(`| ${engine} `))
  if (!line) throw new Error(`BENCHMARKS.md has no release row for ${engine}`)
  return [...line.matchAll(/([\d.]+) (?:s|t\/s)/g)].map(found => Number(found[1]))
}

describe("the squeeze and release replay", () => {
  it("matches BENCHMARKS.md to its rounding", () => {
    for (const [engine, row] of [
      ["phobos", benchmarksRow("Phobos GPU")],
      ["llama.cpp", benchmarksRow("llama.cpp CUDA")],
    ] as const) {
      const derived = data.engines.find(entry => entry.engine === engine)!
      const [wall, ...decode] = row
      expect(Math.abs(derived.wallSecs - wall), engine).toBeLessThan(0.05)
      // BENCHMARKS.md averages the per-repetition rates after rounding them, which can move the last digit.
      decode.forEach((rate, phase) => expect(Math.abs(derived.decode[phase] - rate), `${engine} ${phase}`).toBeLessThan(0.1))
    }
  })

  it("finds the shrinks the squeeze caused and the grow after it", () => {
    expect(data.cache.before).toBe(38)
    expect(data.cache.shrinks.at(-1)).toBe(14)
    expect(data.cache.after).toBe(39)
    expect(benchmarks).toContain(`from ${data.cache.shrinks.at(-1)} back to ${data.cache.after} experts a block`)
    expect(data.squeezeMiB).toBe(2048)
    expect(data.cardGiB).toBe(8)
  })

  it("tells a context shrink from the squeeze by the free memory it read", () => {
    const resizes = [
      { freeMiB: 259, fromPerBlock: 49, toPerBlock: 38 },
      { freeMiB: 0, fromPerBlock: 38, toPerBlock: 24 },
      { freeMiB: 323, fromPerBlock: 24, toPerBlock: 14 },
      { freeMiB: 3413, fromPerBlock: 14, toPerBlock: 39 },
    ]
    expect(squeezeResizes(resizes)).toEqual({ before: 38, shrinks: [24, 14], after: 39 })
  })

  it("names the card and model BENCHMARKS.md and the 35B's bench run name", () => {
    const moe = loadRuns().find(run => run.models.some(model => model.model.startsWith(SQUEEZE_RELEASE.model)))
    expect(moe?.card).toBe(SQUEEZE_RELEASE.card)
    expect(benchmarks).toContain(SQUEEZE_RELEASE.card.replace(/^NVIDIA GeForce /, ""))
    expect(benchmarks).toContain(`The model is ${SQUEEZE_RELEASE.model}.`)
    expect(Number.isNaN(Date.parse(data.date))).toBe(false)
  })

  it("is what src/data/elastic-cache.json holds", async () => {
    await expect(JSON.stringify(data, null, 2) + "\n").toMatchFileSnapshot("../src/data/elastic-cache.json")
  })
})
