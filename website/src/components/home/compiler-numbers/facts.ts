// What the compiler does to a kernel, as numbers read from the committed traces (src/data/traces, written
// by `site -- traces`), the kernel benchmark CSV and the source tree. Pure: imports in vitest and at
// build time, never in the browser.

import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import type { Trace } from "../../../lib/viz/trace"
import { repoRoot } from "../../../lib/bench/load"
import { formatGflops, kernelBench, loadKernelResults, type KernelResults } from "../../../lib/bench/kernels"
import gemmNoAligned from "../../../data/traces/gemm_fp32_no_aligned.sm_75.json"
import gemmPipelined from "../../../data/traces/gemm_fp32.sm_75.json"
import gemmUnpipelined from "../../../data/traces/gemm_fp32_no_pipeline.sm_75.json"
import flash from "../../../data/traces/flash_attention_fp32.sm_75.json"
import fp16PipelineSm75 from "../../../data/traces/gemm_fp16_pipeline.sm_75.json"
import fp16PipelineSm80 from "../../../data/traces/gemm_fp16_pipeline.sm_80.json"

const asTrace = (json: unknown) => json as Trace

export interface Placement {
  trace: Trace
  buffers: number
  summedBytes: number
  peakBytes: number
  // One rectangle per buffer: the ops it is live over, the bytes it occupies.
  spans: { firstOp: number; lastOp: number; offsetBytes: number; bytes: number }[]
  opCount: number
}

export function placement(trace: Trace = asTrace(gemmNoAligned)): Placement {
  return {
    trace,
    buffers: trace.buffers.length,
    summedBytes: trace.buffers.reduce((sum, buffer) => sum + buffer.bytes, 0),
    peakBytes: trace.sharedPeakBytes ?? 0,
    spans: trace.buffers.map(({ firstOp, lastOp, offsetBytes, bytes }) => ({
      firstOp,
      lastOp,
      offsetBytes: offsetBytes ?? 0,
      bytes,
    })),
    opCount: trace.ops.length,
  }
}

export interface Barriers {
  trace: Trace
  kept: number
  elided: number
  // Every site in op order, kept or not.
  sites: boolean[]
}

export function barriers(trace: Trace = asTrace(flash)): Barriers {
  const sites = [...(trace.barriers ?? [])].sort((a, b) => a.afterOp - b.afterOp).map(barrier => barrier.kept)
  return {
    trace,
    kept: sites.filter(Boolean).length,
    elided: sites.filter(kept => !kept).length,
    sites,
  }
}

export interface Pipelining {
  stagingBefore: { buffers: number; bytes: number }
  stagingAfter: { buffers: number; bytes: number }
  // Staging copies the trace marks asynchronous (cp.async), per chip.
  asyncCopies: { chip: string; count: number }[]
}

const staging = (trace: Trace) => ({
  buffers: trace.buffers.length,
  bytes: trace.buffers.reduce((sum, buffer) => sum + buffer.bytes, 0),
})

export function pipelining(): Pipelining {
  const asyncCount = (trace: Trace) => trace.steps.filter(step => step.map?.async === true).length
  return {
    stagingBefore: staging(asTrace(gemmUnpipelined)),
    stagingAfter: staging(asTrace(gemmPipelined)),
    asyncCopies: [asTrace(fp16PipelineSm75), asTrace(fp16PipelineSm80)].map(trace => ({
      chip: trace.compiler.chip,
      count: asyncCount(trace),
    })),
  }
}

export interface SearchDim {
  name: string
  values: number[]
}

// The configurations an @autotune line names: two values are inclusive bounds searched in doubling
// steps, three or more an explicit list (SPEC.md, Attributes).
export function searchSpace(source: string): SearchDim[] {
  const line = source.match(/@autotune\(([^)]*)\)/)
  if (!line) return []
  return [...line[1].matchAll(/(\w+)\s+in\s+\[([^\]]*)\]/g)].map(([, name, list]) => {
    const given = list.split(",").map(value => Number(value.trim()))
    if (given.length !== 2) return { name, values: given }
    const values: number[] = []
    for (let value = given[0]; value <= given[1]; value *= 2) values.push(value)
    return { name, values }
  })
}

export function configCount(space: SearchDim[]): number {
  return space.reduce((count, dim) => count * dim.values.length, 1)
}

// phobos-base/src/context.rs: the chips a release compiles and caches kernels for.
export function supportedChips(root = repoRoot()): string[] {
  const text = readFileSync(resolve(root, "phobos-base/src/context.rs"), "utf8")
  const found = text.match(/SUPPORTED_CHIPS:\s*\[&str;\s*\d+\]\s*=\s*\[([^\]]*)\]/)
  if (!found) throw new Error("phobos-base/src/context.rs: no SUPPORTED_CHIPS")
  return [...found[1].matchAll(/"([^"]+)"/g)].map(match => match[1])
}

export interface TensorCoreGain {
  plain: number
  tensorCore: number
  ratio: number
}

// The f32 gemm and the same body with @tensorcore, both Phobos, from the kernel benchmark.
export function tensorCoreGain(results: KernelResults = loadKernelResults()): TensorCoreGain {
  const plain = kernelBench(results, "gemm_fp32").phobos.gflops
  const tensorCore = kernelBench(results, "gemm_fp16tc_fp32acc").phobos.gflops
  return { plain, tensorCore, ratio: tensorCore / plain }
}

export function kib(bytes: number): string {
  const value = bytes / 1024
  return `${Number.isInteger(value) ? value : value.toFixed(1)} KiB`
}

export interface Fact {
  id: string
  label: string
  figure: string
  text: string
  href: string
  linkText: string
}

export function compilerFacts(results: KernelResults = loadKernelResults()): Fact[] {
  const place = placement()
  const sync = barriers()
  const pipe = pipelining()
  const gemmSource = asTrace(gemmPipelined).source.text
  const configs = configCount(searchSpace(gemmSource))
  const chips = supportedChips()
  const gain = tensorCoreGain(results)
  const sm80 = pipe.asyncCopies.find(entry => entry.chip === "sm_80")
  const sm75 = pipe.asyncCopies.find(entry => entry.chip === "sm_75")

  return [
    {
      id: "placement",
      label: "Shared memory by liveness",
      figure: `${place.buffers} tiles, ${kib(place.summedBytes)} in ${kib(place.peakBytes)}`,
      text: "A tile lives from its declaration to its last use, and tiles whose lifetimes do not overlap share bytes. Shown for gemm_fp32 without @aligned.",
      href: "/viz/gemm_fp32_no_aligned/",
      linkText: "Step through it",
    },
    {
      id: "barriers",
      label: "Barriers where threads meet",
      figure: `${sync.kept} kept, ${sync.elided} elided`,
      text: `A barrier follows a tile store only when other threads next touch those bytes. Of flash attention's ${sync.kept + sync.elided} candidate sites, ${sync.elided} need none.`,
      href: "/docs/language/tiles-and-shared-memory/",
      linkText: "Tiles and shared memory",
    },
    {
      id: "pipeline",
      label: "Double-buffered loops",
      figure: `${pipe.stagingBefore.buffers} to ${pipe.stagingAfter.buffers} staging tiles`,
      text: `The next k-slice loads while this one computes, on by default. On sm_80 the copies become cp.async: ${sm80?.count ?? 0} in the gemm_fp16 trace, ${sm75?.count ?? 0} on sm_75.`,
      href: "/docs/attributes/pipeline/",
      linkText: "@pipeline",
    },
    {
      id: "tensorcore",
      label: "Tensor cores from one attribute",
      figure: `${gain.ratio.toFixed(2)}x the f32 gemm`,
      text: `The same gemm body with @tensorcore: ${formatGflops(gain.tensorCore)} against ${formatGflops(gain.plain)}, operands rounded to f16. mma.sync and ldmatrix with swizzled staging, or WMMA.`,
      href: "/docs/attributes/tensorcore/",
      linkText: "@tensorcore",
    },
    {
      id: "autotune",
      label: "A search, not a guess",
      figure: `${configs} gemm configurations`,
      text: "@autotune names the tile sizes, the unroll and the CTA size. phobos-kbench compiles and checks every configuration, ranks them on warm clocks, then races the fastest over 30 interleaved rounds.",
      href: "/docs/language/autotuning/",
      linkText: "Autotuning",
    },
    {
      id: "chips",
      label: "One instruction seam",
      figure: `${chips.length} chips, ${chips[0]} to ${chips[chips.length - 1]}`,
      text: "Only the target layer names an instruction. Kernels are compiled and cached for every chip ahead of time, with no GPU in the machine.",
      href: "/docs/inference/kernel-cache/",
      linkText: "The kernel cache",
    },
  ]
}
