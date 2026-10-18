import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it } from "vitest"
import { parseKernelCsv } from "../src/lib/bench/kernels"
import { loadRuns, modelRows, repoRoot } from "../src/lib/bench/load"
import { combineKernelRuns, loadPlatforms, PLATFORM_MODEL, testOf } from "../src/lib/bench/platforms"

const platforms = loadPlatforms()

describe("platforms", () => {
  it("orders the cards by chip", () => {
    const chips = platforms.map(platform => Number(platform.chip.replace("sm_", "")))
    expect(chips).toEqual([...chips].sort((a, b) => a - b))
    expect(platforms[0].card).toBe("RTX 2080 SUPER")
  })

  it("has both engines on the headline tests of every card with a bench.py run", () => {
    for (const platform of platforms.filter(entry => entry.model)) {
      for (const test of ["tg128", "pp128", "pp512"]) {
        const comparison = testOf(platform, test)
        expect(comparison?.phobos?.rounds, `${platform.card} ${test}`).toBeGreaterThan(1)
        expect(comparison?.other?.rounds, `${platform.card} ${test}`).toBeGreaterThan(1)
      }
    }
  })

  it("shows the development card's numbers as the Benchmarks page does", () => {
    const row = modelRows(loadRuns()).find(
      entry => entry.model === PLATFORM_MODEL && entry.run.file === "results/bench.json"
    )!
    expect(platforms[0].model?.tests).toEqual(row.tests)
  })

  it("averages kbench runs, and the ratio is the mean of each run's ratio", () => {
    const header = "benchmark,impl,precision,gflops,peak_gflops,pct_of_peak\n"
    const run = (phobos: number, cublas: number) =>
      parseKernelCsv(
        `${header}gemm_fp32,phobos,f32,${phobos},100,1\ngemm_fp32,cuBLAS,f32,${cublas},100,1\nflash_fp32,phobos,f32,5,100,1\n`,
        "2026-10-16",
        "x.csv",
        "card"
      )
    const [gemm, flash] = combineKernelRuns([run(60, 100), run(80, 100)])
    expect(gemm.phobos.mean).toBe(70)
    expect(gemm.phobos.stderr).toBeCloseTo(10, 10)
    expect(gemm.ratio!.mean).toBeCloseTo(0.7, 10)
    expect(flash.cublas).toBeUndefined()
    expect(flash.ratio).toBeUndefined()
    expect(gemm.label).toBe("gemm, f32")
    expect(gemm.precision).toBe("f32")
    expect(gemm.peakGflops).toBe(100)
  })

  it("reads every kbench file a card names", () => {
    for (const platform of platforms) {
      const gemm = platform.kernels.find(kernel => kernel.benchmark === "gemm_fp32")!
      const values = platform.kernelFiles.map(file => {
        const line = readFileSync(resolve(repoRoot(), file), "utf8")
          .split(/\r?\n/)
          .find(entry => entry.startsWith("gemm_fp32,phobos,"))!
        return Number(line.split(",")[3])
      })
      expect(gemm.phobos.rounds).toBe(values.length)
      expect(gemm.phobos.mean).toBeCloseTo(values.reduce((sum, value) => sum + value, 0) / values.length, 6)
    }
  })
})
