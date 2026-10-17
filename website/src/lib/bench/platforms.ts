// Reads results/platforms/*/card.json: one card each, the bench.py run that measured Phobos against
// llama.cpp on it and the phobos-kbench runs beside it. A card's files may live elsewhere under
// results/, which is how the development card points at the runs the Benchmarks page already shows.
// Runs at build time and in vitest.

import { readdirSync, readFileSync } from "node:fs"
import { resolve } from "node:path"
import committedDates from "./dates.json"
import { parseKernelCsv, type KernelResults } from "./kernels"
import { parseRun, repoRoot, type Comparison, type ModelBench, type Stat } from "./load"

export const PLATFORMS_DIR = "results/platforms"

// The one model every card runs: small enough for an 8 GB card.
export const PLATFORM_MODEL = "Qwen3.5-4B-Q4_K_M"

interface CardFile {
  card: string
  chip: string
  memoryGiB: number
  powerLimitWatts: number
  // Where the card ran, such as "vast.ai, Spain".
  host: string
  // ISO 8601 date of the kbench runs, which carry none of their own.
  date?: string
  // The commit the kbench runs were built from, when it is not the bench.py run's.
  kbenchCommit?: string
  // Where the kbench runs ran and the card's power limit there, when they ran on another rental
  // than the bench.py run.
  kbenchHost?: string
  kbenchPowerLimitWatts?: number
  bench: string
  kbench: string[]
}

export interface KernelStat {
  benchmark: string
  label: string
  example?: string
  phobos: Stat
  cublas?: Stat
  // cuBLAS ran, though maybe at another precision than Phobos.
  cublasRan: boolean
  // Phobos over cuBLAS, the mean and standard error over runs of each run's ratio.
  ratio?: Stat
}

export interface Platform extends Omit<CardFile, "bench" | "kbench" | "kbenchCommit"> {
  dir: string
  driver: string
  commit: string
  kbenchCommit: string
  benchDate: string
  benchFile: string
  rounds: number
  reps: number
  model: ModelBench
  kernelFiles: string[]
  kernels: KernelStat[]
}

function statOf(values: number[]): Stat {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  let stderr = 0
  if (values.length > 1) {
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)
    stderr = Math.sqrt(variance / values.length)
  }
  return { mean, stderr, rounds: values.length }
}

// Each kbench run autotunes on its own, so runs are the samples, as rounds are for bench.py.
export function combineKernelRuns(runs: KernelResults[]): KernelStat[] {
  return runs[0].benches.map(first => {
    const across = runs.map(run => {
      const bench = run.benches.find(entry => entry.benchmark === first.benchmark)
      if (!bench) throw new Error(`${run.file} has no ${first.benchmark}`)
      return bench
    })
    const comparable = across.every(bench => bench.comparable && bench.cublas)
    return {
      benchmark: first.benchmark,
      label: first.label,
      example: first.example,
      phobos: statOf(across.map(bench => bench.phobos.gflops)),
      cublasRan: across.every(bench => bench.cublas !== undefined),
      cublas: comparable ? statOf(across.map(bench => bench.cublas!.gflops)) : undefined,
      ratio: comparable ? statOf(across.map(bench => bench.phobos.gflops / bench.cublas!.gflops)) : undefined,
    }
  })
}

const DATES = committedDates as Record<string, string>

export function loadPlatform(dir: string, root = repoRoot()): Platform {
  const read = (file: string) => readFileSync(resolve(root, file), "utf8")
  const { bench, kbench, kbenchCommit, ...card } = JSON.parse(read(`${dir}/card.json`)) as CardFile
  const run = parseRun(bench, read(bench), DATES[bench])
  const model = run.models.find(entry => entry.model === PLATFORM_MODEL)
  if (!model) throw new Error(`${bench} has no ${PLATFORM_MODEL}`)
  if (kbench.length === 0) throw new Error(`${dir}/card.json names no kbench run`)
  const kernelRuns = kbench.map(file => parseKernelCsv(read(file), DATES[file] ?? card.date, file, card.card))
  return {
    ...card,
    dir,
    driver: run.driver,
    commit: run.commit,
    kbenchCommit: kbenchCommit ?? run.commit,
    benchDate: run.date,
    benchFile: bench,
    rounds: run.rounds,
    reps: run.reps,
    model,
    kernelFiles: kbench,
    kernels: combineKernelRuns(kernelRuns),
  }
}

// Every card, oldest chip first.
export function loadPlatforms(root = repoRoot()): Platform[] {
  return readdirSync(resolve(root, PLATFORMS_DIR), { withFileTypes: true })
    .filter(entry => entry.isDirectory())
    .map(entry => loadPlatform(`${PLATFORMS_DIR}/${entry.name}`, root))
    .sort((a, b) => chipNumber(a.chip) - chipNumber(b.chip) || a.card.localeCompare(b.card))
}

function chipNumber(chip: string): number {
  return Number(chip.replace(/^sm_/, ""))
}

export function testOf(platform: Platform, test: string): Comparison | undefined {
  return platform.model.tests[test]
}
