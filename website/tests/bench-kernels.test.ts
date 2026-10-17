import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { repoRoot } from "../src/lib/bench/load"
import {
  bandwidthGBs,
  cublasRatio,
  formatPercent,
  kernelBench,
  loadKernelResults,
  parseKernelCsv,
  KERNEL_RESULTS_FILE,
} from "../src/lib/bench/kernels"
import {
  barriers,
  compilerFacts,
  configCount,
  pipelining,
  placement,
  searchSpace,
  supportedChips,
  tensorCoreGain,
} from "../src/components/home/compiler-numbers/facts"

const results = loadKernelResults()
const csv = readFileSync(resolve(repoRoot(), KERNEL_RESULTS_FILE), "utf8")

// The CSV's own cells, independent of the loader.
function cell(benchmark: string, impl: string, column: string): number {
  const [header, ...rows] = csv.trim().split(/\r?\n/)
  const columns = header.replace(/^﻿/, "").split(",")
  const row = rows.map(line => line.split(",")).find(cells => cells[0] === benchmark && cells[1] === impl)
  if (!row) throw new Error(`no ${benchmark}/${impl}`)
  return Number(row[columns.indexOf(column)])
}

describe("kernel results", () => {
  it("dates the CSV from the commit that last touched it", () => {
    expect(Number.isNaN(Date.parse(results.date))).toBe(false)
    expect(() => parseKernelCsv(csv, undefined)).toThrow(/dates\.json/)
  })

  it("loads every benchmark with a phobos row and its numbers as written", () => {
    expect(results.benches.length).toBeGreaterThan(0)
    for (const bench of results.benches) {
      expect(bench.phobos.gflops).toBe(cell(bench.benchmark, "phobos", "gflops"))
      expect(bench.phobos.pctOfPeak).toBe(cell(bench.benchmark, "phobos", "pct_of_peak"))
      if (bench.cublas) expect(bench.cublas.gflops).toBe(cell(bench.benchmark, "cuBLAS", "gflops"))
    }
  })

  it("puts SGEMM at 88% of cuBLAS, as the README says", () => {
    const ratio = cublasRatio(kernelBench(results, "gemm_fp32"))!
    expect(ratio).toBeCloseTo(cell("gemm_fp32", "phobos", "gflops") / cell("gemm_fp32", "cuBLAS", "gflops"), 10)
    expect(ratio).toBeCloseTo(9621.3 / 10957.4, 10)
    expect(formatPercent(ratio)).toBe("88%")
  })

  it("compares f16 gemm against cublasGemmEx, both on tensor cores", () => {
    const bench = kernelBench(results, "gemm_fp16")
    expect(bench.comparable).toBe(true)
    expect(formatPercent(cublasRatio(bench)!)).toBe("60%")
  })

  it("holds the tensor-core f32 gemm against cuBLAS on the same contract", () => {
    const bench = kernelBench(results, "gemm_fp16tc_fp32acc")
    expect(bench.cublas?.precision).toBe("f16-tc-f32-acc")
    expect(bench.phobos.precision).toBe("f16-tc-f32-acc")
    expect(bench.comparable).toBe(true)
    expect(formatPercent(cublasRatio(bench)!)).toBe("57%")
  })

  it("gives no ratio where cuBLAS ran at another precision", () => {
    const csv = [
      "benchmark,impl,precision,gflops,peak_gflops,pct_of_peak",
      "gemm_fp16tc_fp32acc,phobos,f16-tc-f32-acc,2,4,50",
      "gemm_fp16tc_fp32acc,cuBLAS,f32,1,2,50",
    ].join("\n")
    const bench = kernelBench(parseKernelCsv(csv, "2026-01-01"), "gemm_fp16tc_fp32acc")
    expect(bench.comparable).toBe(false)
    expect(cublasRatio(bench)).toBeUndefined()
  })

  it("has no cuBLAS row for flash attention", () => {
    for (const name of ["flash_fp32", "flash_fp16"]) {
      expect(kernelBench(results, name).cublas).toBeUndefined()
      expect(cublasRatio(kernelBench(results, name))).toBeUndefined()
    }
  })

  it("reads saxpy as bandwidth: 12 bytes per element for 2 FLOP", () => {
    const bench = kernelBench(results, "saxpy_fp32")
    expect(bandwidthGBs(bench, bench.phobos)).toBeCloseTo(cell("saxpy_fp32", "phobos", "gflops") * 6, 10)
    expect(bandwidthGBs(bench, bench.cublas!)).toBeCloseTo(cell("saxpy_fp32", "cuBLAS", "gflops") * 6, 10)
    expect(bandwidthGBs(kernelBench(results, "gemm_fp32"), bench.phobos)).toBeUndefined()
  })
})

describe("compiler facts", () => {
  it("places gemm without @aligned: 6 tiles, 10 KiB summed, in 8 KiB", () => {
    const place = placement()
    expect(place.buffers).toBe(6)
    expect(place.summedBytes).toBe(10240)
    expect(place.peakBytes).toBe(8192)
    for (const span of place.spans) expect(span.offsetBytes + span.bytes).toBeLessThanOrEqual(place.peakBytes)
  })

  it("counts flash attention's barriers", () => {
    const sync = barriers()
    expect(sync.kept + sync.elided).toBe(sync.sites.length)
    expect(sync.kept).toBe(20)
    expect(sync.elided).toBe(9)
  })

  it("doubles gemm's staging under @pipeline and issues cp.async only on sm_80", () => {
    const pipe = pipelining()
    expect(pipe.stagingAfter.buffers).toBe(2 * pipe.stagingBefore.buffers)
    expect(pipe.stagingAfter.bytes).toBe(2 * pipe.stagingBefore.bytes)
    expect(pipe.asyncCopies.find(entry => entry.chip === "sm_75")?.count).toBe(0)
    expect(pipe.asyncCopies.find(entry => entry.chip === "sm_80")?.count).toBeGreaterThan(0)
  })

  it("expands @autotune bounds in doubling steps", () => {
    const space = searchSpace("@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])")
    expect(space.map(dim => dim.values)).toEqual([
      [32, 64, 128, 256],
      [32, 64, 128, 256],
      [4, 8, 16, 32],
    ])
    expect(configCount(space)).toBe(64)
    expect(searchSpace("@autotune(U in [1, 2, 4], D in [64])").map(dim => dim.values)).toEqual([[1, 2, 4], [64]])
  })

  it("reads the supported chips from the source", () => {
    expect(supportedChips()).toContain("sm_75")
    expect(supportedChips().length).toBeGreaterThan(1)
  })

  it("compares the gemm with and without @tensorcore from the CSV", () => {
    const gain = tensorCoreGain(results)
    expect(gain.ratio).toBeCloseTo(22002.9 / 9621.3, 10)
  })

  it("gives every fact a figure and an internal link", () => {
    for (const fact of compilerFacts(results)) {
      expect(fact.figure, fact.id).toMatch(/\d/)
      expect(fact.href, fact.id).toMatch(/^\/.*\/$/)
    }
  })
})
