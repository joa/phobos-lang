// The committed traces, for build time: which trace belongs to a kernel file, a fence or a variant,
// and on which chips. Imports every trace eagerly, so only Astro frontmatter and tests use it.

import kernels from "../../../kernels.json"
import type { Trace } from "./trace"

const modules = import.meta.glob<Trace>("/src/data/traces/*.json", { import: "default", eager: true })

export const DEFAULT_CHIP = "sm_75"

export interface TraceEntry {
  /** The trace's base name, such as `gemm_fp32` or `gemm_fp32_no_aligned`. */
  name: string
  chips: string[]
  traces: Record<string, Trace>
}

const entries = new Map<string, TraceEntry>()
for (const [path, trace] of Object.entries(modules)) {
  const stem = path.slice(path.lastIndexOf("/") + 1, -".json".length)
  const chip = stem.slice(stem.lastIndexOf(".") + 1)
  const name = stem.slice(0, stem.lastIndexOf("."))
  const entry = entries.get(name) ?? { name, chips: [], traces: {} }
  entry.chips.push(chip)
  entry.traces[chip] = trace
  entries.set(name, entry)
}
for (const entry of entries.values()) entry.chips.sort()

export function traceEntries(): TraceEntry[] {
  return [...entries.values()].sort((a, b) => a.name.localeCompare(b.name))
}

export function traceEntry(name: string): TraceEntry | undefined {
  return entries.get(name)
}

/** The trace of a kernel file as it is, without a variant, from kernels.json. */
export function traceForFile(file: string): TraceEntry | undefined {
  const kernel = kernels.kernels.find(candidate => candidate.path === file && !("variant" in candidate))
  return kernel ? entries.get(kernel.name) : undefined
}

export function defaultChip(entry: TraceEntry): string {
  return entry.chips.includes(DEFAULT_CHIP) ? DEFAULT_CHIP : entry.chips[0]
}
