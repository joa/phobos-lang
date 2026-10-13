import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { formatFamilies, quantFormats, testedModels, displayName } from "../src/lib/bench/catalog"
import { benchFiles, loadRuns, modelRows, parseRun, ratioOf, repoRoot } from "../src/lib/bench/load"

const runs = loadRuns()
const rows = modelRows(runs)

describe("bench results", () => {
  it("loads every results/bench*.json", () => {
    expect(runs.map(run => run.file)).toEqual(benchFiles())
    expect(runs.length).toBeGreaterThan(0)
  })

  it("gives every model a card, driver, commit and date", () => {
    for (const row of rows) {
      expect(row.run.card, row.model).toMatch(/\S/)
      expect(row.run.driver, row.model).toMatch(/\S/)
      expect(row.run.commit, row.model).toMatch(/^[0-9a-f?]+$/)
      expect(Number.isNaN(Date.parse(row.run.date)), row.model).toBe(false)
    }
  })

  it("has no NaN anywhere", () => {
    for (const row of rows) {
      expect(Number.isFinite(row.paramsBillions), row.model).toBe(true)
      for (const [test, comparison] of Object.entries(row.tests)) {
        for (const stat of [comparison.phobos, comparison.other]) {
          expect(stat, `${row.model} ${test}`).toBeDefined()
          expect(Number.isFinite(stat!.mean)).toBe(true)
          expect(Number.isFinite(stat!.stderr)).toBe(true)
        }
        expect(Number.isFinite(ratioOf(comparison))).toBe(true)
      }
    }
  })

  it("has decode at tg128 and prefill at pp128 for every model", () => {
    for (const row of rows) {
      expect(row.tests.tg128, row.model).toBeDefined()
      expect(row.tests.pp128, row.model).toBeDefined()
    }
  })

  it("matches the tables in BENCHMARKS.md", () => {
    // Every "| test | other | phobos |" row of a table headed by a model name, as BENCHMARKS.md prints them.
    const text = readFileSync(resolve(repoRoot(), "BENCHMARKS.md"), "utf8").replace(/\r\n?/g, "\n")
    let checked = 0
    for (const row of rows) {
      const heading = text.toLowerCase().indexOf(`\n${row.model.toLowerCase()}`)
      if (heading < 0) continue
      const table = text.slice(heading + 1).split("\n\n")[1] ?? ""
      for (const line of table.split("\n")) {
        const cells = line.split("|").map(cell => cell.trim())
        const test = cells[1]
        const comparison = row.tests[test]
        if (!comparison) continue
        const [other, phobos] = [cells[2], cells[3]].map(cell => Number(cell.split("+/-")[0]))
        expect(comparison.other!.mean, `${row.model} ${test}`).toBeCloseTo(other, 1)
        expect(comparison.phobos!.mean, `${row.model} ${test}`).toBeCloseTo(phobos, 1)
        checked += 1
      }
    }
    expect(checked).toBeGreaterThanOrEqual(15)
  })

  it("falls back to the committed date and prefers meta.date", () => {
    const text = JSON.stringify({
      meta: { card: "c", driver: "d", commit: "abc" },
      samples: [
        { round: 1, engine: "phobos", model: "m", test: "tg128", rate: 2 },
        { round: 1, engine: "llama.cpp CUDA", model: "m", test: "tg128", rate: 1 },
      ],
    })
    expect(parseRun("f", text, "2026-01-02T00:00:00Z").dateSource).toBe("git")
    const dated = text.replace('"commit":"abc"', '"commit":"abc","date":"2026-03-04T00:00:00+00:00"')
    expect(parseRun("f", dated, "2026-01-02T00:00:00Z").date).toBe("2026-03-04T00:00:00+00:00")
    expect(() => parseRun("f", text, undefined)).toThrow(/meta.date/)
  })

  it("drops contended rounds the way plot.py does", () => {
    const sample = (round: number, engine: string, rate: number) => ({ round, engine, model: "m", test: "tg8", rate })
    const run = parseRun(
      "f",
      JSON.stringify({
        meta: { card: "c", driver: "d", commit: "abc", date: "2026-01-01" },
        summary: { uncontended_rounds: [1, 2] },
        samples: [
          sample(1, "phobos", 10),
          sample(1, "phobos", 12),
          sample(2, "phobos", 14),
          sample(3, "phobos", 1),
          sample(1, "other", 5),
          sample(2, "other", 5),
          sample(3, "other", 5),
        ],
      }),
      undefined
    )
    expect(run.rounds).toBe(2)
    expect(run.droppedRounds).toBe(1)
    expect(run.models[0].tests.tg8.phobos!.mean).toBeCloseTo(12.5)
    expect(run.models[0].tests.tg8.phobos!.stderr).toBeCloseTo(1.5)
  })
})

describe("catalog", () => {
  it("lists the README's GGUF models and names every benchmarked one", () => {
    const tested = testedModels()
    expect(tested.length).toBeGreaterThan(0)
    expect(tested.some(model => /gpt2|onnx/i.test(model.name))).toBe(false)
    for (const row of rows) expect(tested.map(model => model.name)).toContain(displayName(row.model, tested))
  })

  it("finds a format for every file in phobos-gguf/src/quant and puts each in a family", () => {
    const formats = quantFormats()
    expect(formats).toContain("Q8_0")
    expect(formats).toContain("PTQ1_0")
    const grouped = formatFamilies(formats).flatMap(group => group.formats)
    expect(grouped.sort()).toEqual([...formats].sort())
  })
})
