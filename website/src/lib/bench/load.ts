// Reads what scripts/bench.py wrote to results/bench*.json and summarizes it the way bench.py's table and
// scripts/plot.py do: only the rounds bench.py found uncontended, a mean per round, then the mean and the
// standard error over rounds. Runs at build time and in vitest; nothing here touches the DOM.

import { readdirSync, readFileSync } from "node:fs"
import { resolve } from "node:path"
import committedDates from "./dates.json"

export const REPO_URL = "https://github.com/joa/phobos-lang"

// Builds and vitest run from website/.
export function repoRoot(): string {
  return resolve(process.cwd(), "..")
}

export interface Stat {
  mean: number
  stderr: number
  rounds: number
}

export interface Comparison {
  phobos?: Stat
  other?: Stat
}

export interface ModelBench {
  // The model's file name without its extension, as the samples carry it.
  model: string
  // Parameter count in billions, from the label phobos reported for the file.
  paramsBillions: number
  tests: Record<string, Comparison>
}

export interface BenchRun {
  // Repo-relative path, such as results/bench.json.
  file: string
  card: string
  driver: string
  commit: string
  // ISO 8601. From meta.date, or for older files the last commit that touched the file.
  date: string
  dateSource: "meta" | "git"
  otherEngine: string
  // Extra flags the other engine ran with, such as "-ncmoe 31 -t 8"; empty when none.
  otherArgs: string
  rounds: number
  droppedRounds: number
  reps: number
  models: ModelBench[]
}

interface Sample {
  round: number
  engine: string
  model: string
  test: string
  rate: number
}

interface RawRun {
  meta?: Record<string, string | undefined>
  args?: Record<string, string | undefined>
  labels?: Record<string, string>
  summary?: { uncontended_rounds?: number[] }
  samples?: Sample[]
}

export function isPhobos(engine: string): boolean {
  return engine.startsWith("phobos")
}

function summarize(rates: Map<number, number[]>): Stat {
  const means = [...rates.values()].map(values => values.reduce((sum, value) => sum + value, 0) / values.length)
  const mean = means.reduce((sum, value) => sum + value, 0) / means.length
  let stderr = 0
  if (means.length > 1) {
    const variance = means.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (means.length - 1)
    stderr = Math.sqrt(variance / means.length)
  }
  return { mean, stderr, rounds: means.length }
}

function paramsOf(label: string | undefined): number {
  const found = label?.match(/(\d+(?:\.\d+)?)B\b/)
  return found ? Number(found[1]) : Number.NaN
}

export function parseRun(file: string, text: string, gitDate: string | undefined): BenchRun {
  const raw = JSON.parse(text) as RawRun
  const meta = raw.meta ?? {}
  for (const field of ["card", "driver", "commit"] as const) {
    if (!meta[field]) throw new Error(`${file}: meta.${field} is missing`)
  }
  const date = meta.date ?? gitDate
  if (!date) throw new Error(`${file}: no meta.date and no entry in src/lib/bench/dates.json`)
  const samples = raw.samples ?? []
  if (samples.length === 0) throw new Error(`${file}: no samples`)
  for (const sample of samples) {
    if (!Number.isFinite(sample.rate)) throw new Error(`${file}: a ${sample.test} sample has rate ${sample.rate}`)
  }

  const allRounds = new Set(samples.map(sample => sample.round))
  const clean = new Set(raw.summary?.uncontended_rounds ?? [])
  const keptRounds = [...allRounds].filter(round => clean.has(round))
  // As plot.py: with no record, or every round contended, everything is kept.
  const kept = keptRounds.length > 0 ? new Set(keptRounds) : allRounds
  const used = samples.filter(sample => kept.has(sample.round))

  const engines = [...new Set(used.map(sample => sample.engine))]
  const others = engines.filter(engine => !isPhobos(engine))
  if (others.length !== 1 || !engines.some(isPhobos)) {
    throw new Error(`${file}: expected phobos and one other engine, got ${engines.join(", ")}`)
  }
  const otherEngine = others[0]

  const cells = new Map<string, Map<number, number[]>>()
  for (const sample of used) {
    const key = `${sample.engine}\n${sample.model}\n${sample.test}`
    const rounds = cells.get(key) ?? new Map<number, number[]>()
    rounds.set(sample.round, [...(rounds.get(sample.round) ?? []), sample.rate])
    cells.set(key, rounds)
  }

  const modelNames = [...new Set(used.map(sample => sample.model))]
  const models = modelNames.map(model => {
    const tests: Record<string, Comparison> = {}
    for (const [key, rates] of cells) {
      const [engine, cellModel, test] = key.split("\n")
      if (cellModel !== model) continue
      tests[test] ??= {}
      tests[test][isPhobos(engine) ? "phobos" : "other"] = summarize(rates)
    }
    return { model, paramsBillions: paramsOf(raw.labels?.[`phobos/${model}`]), tests }
  })

  const roundRepCells = new Set(
    used.map(sample => `${sample.round}\n${sample.engine}\n${sample.model}\n${sample.test}`)
  )
  return {
    file,
    card: meta.card!,
    driver: meta.driver!,
    commit: meta.commit!,
    date,
    dateSource: meta.date ? "meta" : "git",
    otherEngine,
    otherArgs: (raw.args?.llama_args ?? "").trim(),
    rounds: kept.size,
    droppedRounds: allRounds.size - kept.size,
    reps: Math.round(used.length / Math.max(roundRepCells.size, 1)),
    models,
  }
}

export function benchFiles(root = repoRoot()): string[] {
  return readdirSync(resolve(root, "results"))
    .filter(name => /^bench.*\.json$/.test(name))
    .sort()
    .map(name => `results/${name}`)
}

const DATES = committedDates as Record<string, string>

export function loadRuns(root = repoRoot()): BenchRun[] {
  return benchFiles(root).map(file => parseRun(file, readFileSync(resolve(root, file), "utf8"), DATES[file]))
}

export interface ModelRow extends ModelBench {
  run: BenchRun
}

// Every model of every run, smallest first.
export function modelRows(runs: BenchRun[]): ModelRow[] {
  return runs
    .flatMap(run => run.models.map(model => ({ ...model, run })))
    .sort((a, b) => a.paramsBillions - b.paramsBillions || a.run.file.localeCompare(b.run.file))
}

// The test names a set of rows carries, prompt before decode, then by token count.
export function testsOf(rows: ModelRow[], kind: "pp" | "tg"): string[] {
  const names = new Set(rows.flatMap(row => Object.keys(row.tests)).filter(test => test.startsWith(kind)))
  return [...names].sort((a, b) => Number(a.slice(2)) - Number(b.slice(2)))
}

export function engineLabel(engine: string): string {
  if (isPhobos(engine)) return "Phobos"
  if (/prism/i.test(engine)) return "PrismML's llama.cpp fork"
  return "llama.cpp"
}

export function shortDate(iso: string): string {
  return iso.slice(0, 10)
}

export function formatRate(rate: number): string {
  if (rate >= 1000) return Math.round(rate).toLocaleString("en-US")
  if (rate >= 100) return rate.toFixed(0)
  return rate.toFixed(1)
}

export function ratioOf(comparison: Comparison | undefined): number | undefined {
  if (!comparison?.phobos || !comparison.other || comparison.other.mean === 0) return undefined
  return comparison.phobos.mean / comparison.other.mean
}

export function commitUrl(commit: string): string {
  return `${REPO_URL}/commit/${commit}`
}
