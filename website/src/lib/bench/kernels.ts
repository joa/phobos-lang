// Reads what phobos-kbench wrote to results/results.csv (`--csv`): one row per benchmark and
// implementation, with the achieved GFLOP/s, the peak phobos-kbench derives for that row's precision from
// the card's SM count and clock, and the share of it. The CSV names no card, driver or commit; the card
// is the one README.md and BENCHMARKS.md give for these numbers, and the date is the last commit that
// touched the file (src/lib/bench/dates.json). Runs at build time and in vitest.

import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import committedDates from "./dates.json"
import { repoRoot } from "./load"

export const KERNEL_RESULTS_FILE = "results/results.csv"

// README.md: "SGEMM performance is at 77% throughput of cuBLAS `cublasSgemm_v2` on a 2080 SUPER".
export const KERNEL_CARD = "RTX 2080 SUPER"

export interface KernelMeasure {
  gflops: number
  peakGflops: number
  pctOfPeak: number
  // As the CSV spells it: f32, f16-tc or f16-tc-f32-acc.
  precision: string
}

export interface KernelSpec {
  // The page under /docs/examples/ that explains the kernel, if there is one.
  example?: string
  // The .ph file phobos-kbench compiles.
  source: string
  label: string
  // The problem phobos-kbench runs, from its source (phobos-kbench/src/{saxpy,gemm,flash}.rs).
  size: string
  // What the cuBLAS row times, when there is one.
  cublasCall?: string
  // Bytes moved per FLOP, for a kernel whose measure is bandwidth rather than arithmetic.
  bytesPerFlop?: number
}

// saxpy reads x and y and writes out, 12 bytes per element for 2 FLOP; cuBLAS's in-place
// y = alpha * x + y moves the same 12. GEMM counts 2MNK + MN FLOP, flash attention 4 * Nq * Nk * D.
export const KERNEL_SPECS: Record<string, KernelSpec> = {
  saxpy_fp32: {
    example: "/docs/examples/saxpy-fp32/",
    source: "examples/saxpy_fp32.ph",
    label: "saxpy",
    size: "n = 2^25",
    cublasCall: "cublasSaxpy_v2",
    bytesPerFlop: 12 / 2,
  },
  gemm_fp32: {
    example: "/docs/examples/gemm-fp32/",
    source: "examples/gemm_fp32.ph",
    label: "gemm, f32",
    size: "M = N = K = 4096",
    cublasCall: "cublasSgemm_v2",
  },
  gemm_fp16tc_fp32acc: {
    example: "/docs/examples/gemm-fp16tc-fp32acc/",
    source: "examples/gemm_fp16tc_fp32acc.ph",
    label: "gemm, f32 on tensor cores",
    size: "M = N = K = 4096",
    cublasCall: "cublasSgemm_v2",
  },
  gemm_fp16: {
    source: "examples/gemm_fp16.ph",
    label: "gemm, f16",
    size: "M = N = K = 4096",
    cublasCall: "cublasHgemm",
  },
  flash_fp32: {
    example: "/docs/examples/flash-attention-fp32/",
    source: "examples/flash_attention_fp32.ph",
    label: "flash attention, f32",
    size: "Nq = Nk = 4096, D = 64",
  },
  flash_fp16: {
    source: "examples/flash_attention_fp16.ph",
    label: "flash attention, f16",
    size: "Nq = Nk = 4096, D = 64",
  },
}

export interface KernelBench extends KernelSpec {
  benchmark: string
  phobos: KernelMeasure
  // The cuBLAS row phobos-kbench timed beside it, at whatever precision cuBLAS ran.
  cublas?: KernelMeasure
  // Both rows ran at the same precision, so their ratio means something.
  comparable: boolean
}

export interface KernelResults {
  file: string
  // ISO 8601, the last commit that touched the CSV.
  date: string
  card: string
  benches: KernelBench[]
}

const COLUMNS = ["benchmark", "impl", "precision", "gflops", "peak_gflops", "pct_of_peak"] as const

export function parseKernelCsv(text: string, date: string | undefined): KernelResults {
  const file = KERNEL_RESULTS_FILE
  if (!date) throw new Error(`${file}: no entry in src/lib/bench/dates.json; run scripts/bench-dates.mjs`)
  const lines = text
    .replace(/^﻿/, "")
    .split(/\r?\n/)
    .filter(line => line.trim() !== "")
  const header = lines[0]?.split(",").map(cell => cell.trim()) ?? []
  const missing = COLUMNS.filter(column => !header.includes(column))
  if (missing.length > 0) throw new Error(`${file}: missing ${missing.join(", ")}`)
  const at = (cells: string[], column: (typeof COLUMNS)[number]) => cells[header.indexOf(column)].trim()

  const rows = new Map<string, Map<string, KernelMeasure>>()
  for (const line of lines.slice(1)) {
    const cells = line.split(",")
    const measure: KernelMeasure = {
      gflops: Number(at(cells, "gflops")),
      peakGflops: Number(at(cells, "peak_gflops")),
      pctOfPeak: Number(at(cells, "pct_of_peak")),
      precision: at(cells, "precision"),
    }
    for (const value of [measure.gflops, measure.peakGflops, measure.pctOfPeak]) {
      if (!Number.isFinite(value)) throw new Error(`${file}: a non-number in "${line}"`)
    }
    const benchmark = at(cells, "benchmark")
    const impls = rows.get(benchmark) ?? new Map<string, KernelMeasure>()
    impls.set(at(cells, "impl"), measure)
    rows.set(benchmark, impls)
  }

  const benches = [...rows].map(([benchmark, impls]) => {
    const spec = KERNEL_SPECS[benchmark]
    if (!spec) throw new Error(`${file}: ${benchmark} has no entry in KERNEL_SPECS`)
    const phobos = impls.get("phobos")
    if (!phobos) throw new Error(`${file}: ${benchmark} has no phobos row`)
    const cublas = impls.get("cuBLAS")
    return {
      ...spec,
      benchmark,
      phobos,
      cublas,
      comparable: cublas !== undefined && cublas.precision === phobos.precision,
    }
  })
  return { file, date, card: KERNEL_CARD, benches }
}

const DATES = committedDates as Record<string, string>

export function loadKernelResults(root = repoRoot()): KernelResults {
  const text = readFileSync(resolve(root, KERNEL_RESULTS_FILE), "utf8")
  return parseKernelCsv(text, DATES[KERNEL_RESULTS_FILE])
}

export function kernelBench(results: KernelResults, benchmark: string): KernelBench {
  const bench = results.benches.find(entry => entry.benchmark === benchmark)
  if (!bench) throw new Error(`${results.file} has no ${benchmark}`)
  return bench
}

// Phobos's rate over cuBLAS's, only when the two ran at the same precision.
export function cublasRatio(bench: KernelBench): number | undefined {
  if (!bench.comparable || !bench.cublas) return undefined
  return bench.phobos.gflops / bench.cublas.gflops
}

// GB/s, for a kernel measured by the bytes it moves.
export function bandwidthGBs(bench: KernelBench, measure: KernelMeasure): number | undefined {
  return bench.bytesPerFlop === undefined ? undefined : measure.gflops * bench.bytesPerFlop
}

export function formatGflops(gflops: number): string {
  return gflops >= 1000 ? `${(gflops / 1000).toFixed(1)} TFLOP/s` : `${gflops.toFixed(1)} GFLOP/s`
}

export function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`
}

export function precisionLabel(precision: string): string {
  switch (precision) {
    case "f32":
      return "f32"
    case "f16-tc":
      return "f16 tensor cores"
    case "f16-tc-f32-acc":
      return "f16 tensor cores, f32 accumulation"
    default:
      return precision
  }
}
