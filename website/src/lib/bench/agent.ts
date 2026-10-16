// Reads what scripts/agent_bench.py wrote for a replay under memory pressure (results/agent-squeeze*.json)
// and summarizes it the way agent_bench.py's report does: each task's requests split at the squeeze into
// before, during and after, decode rates token-weighted within a task, then the mean over repetitions.
// Runs at build time and in vitest; nothing here touches the DOM.

import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import committedDates from "./dates.json"
import { repoRoot } from "./load"

// agent_bench.py records neither the card nor the model. Both are BENCHMARKS.md's, "Under Memory
// Pressure": the 35B, replayed on the card every other table there was measured on.
export const SQUEEZE_RELEASE = {
  file: "results/agent-squeeze-release.json",
  card: "NVIDIA GeForce RTX 2080 SUPER",
  model: "Qwen3.6-35B-A3B",
} as const

export type Phase = "before" | "during" | "after"
export const PHASES: readonly Phase[] = ["before", "during", "after"]

export interface Resize {
  // Free memory on the whole card when the cache decided to resize.
  freeMiB: number
  fromPerBlock: number
  toPerBlock: number
}

export interface EngineSqueeze {
  engine: string
  label: string
  reps: number
  // Median over repetitions, as the BENCHMARKS.md tables quote it.
  wallSecs: number
  // Decode tokens per second before, during and after the squeeze, mean over repetitions.
  decode: [number, number, number]
}

// What the elastic-cache figure draws, written to src/data/elastic-cache.json for the browser.
export interface ElasticCacheData {
  file: string
  date: string
  card: string
  model: string
  cardGiB: number
  // The whole card's memory in use before the squeeze, with the cache at its size before it.
  usedBeforeGiB: number
  squeezeMiB: number
  requests: number
  // 1-based: the squeeze starts before request squeezedAt and ends before request squeezeEndedAt.
  squeezedAt: number
  squeezeEndedAt: number
  cache: {
    blocks: number
    slotMiB: number
    mostPerBlock: number
    // Experts a block before the squeeze, each shrink it caused, and after it ended.
    before: number
    shrinks: number[]
    after: number
  }
  engines: EngineSqueeze[]
}

interface RawRequest {
  generated: number
  decode_secs: number
}

interface RawTask {
  engine: string
  rep: number
  wall_secs: number
  requests: RawRequest[]
  squeezed_at: number
  squeeze_ended?: number
  squeeze?: string
  resizes?: [number, number, number][]
}

interface RawSqueezeRun {
  meta?: { phobos?: { expert_cache?: string } }
  tasks?: RawTask[]
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
}

function round(value: number, digits: number): number {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

function decodeRate(requests: RawRequest[]): number {
  const seconds = requests.reduce((sum, request) => sum + request.decode_secs, 0)
  return requests.reduce((sum, request) => sum + request.generated, 0) / seconds
}

export function phaseRates(task: RawTask): [number, number, number] {
  const start = task.squeezed_at - 1
  const end = task.squeeze_ended ? task.squeeze_ended - 1 : task.requests.length
  return [
    decodeRate(task.requests.slice(0, start)),
    decodeRate(task.requests.slice(start, end)),
    decodeRate(task.requests.slice(end)),
  ]
}

// The resizes the squeeze caused. The JSON has no times, and the cache also shrinks as the context
// grows and grows when that frees up again, so the squeeze is told by the free memory around it: its
// end is the grow that read the card emptiest, its shrinks are the unbroken run before that grow, and
// its first shrink is the one in that run that read the card fullest.
export function squeezeResizes(resizes: Resize[]): { before: number; shrinks: number[]; after: number } {
  const grows = resizes.flatMap((resize, index) => (resize.toPerBlock > resize.fromPerBlock ? [index] : []))
  if (grows.length === 0) throw new Error("the cache never grew back")
  const grow = grows.reduce((best, index) => (resizes[index].freeMiB > resizes[best].freeMiB ? index : best))
  let start = grow
  while (start > 0 && resizes[start - 1].toPerBlock < resizes[start - 1].fromPerBlock) start--
  const shrinks = resizes.slice(start, grow)
  if (shrinks.length === 0) throw new Error("expected shrinks before the grow")
  const fullest = Math.min(...shrinks.map(resize => resize.freeMiB))
  const onset = shrinks.findIndex(resize => resize.freeMiB === fullest)
  return {
    before: shrinks[onset].fromPerBlock,
    shrinks: shrinks.slice(onset).map(resize => resize.toPerBlock),
    after: resizes[grow].toPerBlock,
  }
}

function matchNumber(text: string | undefined, pattern: RegExp, what: string): number {
  const found = text?.match(pattern)
  if (!found) throw new Error(`no ${what} in ${JSON.stringify(text)}`)
  return Number(found[1])
}

export function parseSqueezeRun(
  file: string,
  text: string,
  date: string | undefined,
  { card, model }: { card: string; model: string }
): ElasticCacheData {
  if (!date) throw new Error(`${file}: no entry in src/lib/bench/dates.json; run scripts/bench-dates.mjs`)
  const raw = JSON.parse(text) as RawSqueezeRun
  const tasks = raw.tasks ?? []
  const phobos = tasks.filter(task => task.engine.startsWith("phobos"))
  const others = tasks.filter(task => !task.engine.startsWith("phobos"))
  if (phobos.length === 0 || others.length === 0) throw new Error(`${file}: expected phobos and llama.cpp tasks`)
  const first = phobos[0]
  if (!first.squeeze_ended) throw new Error(`${file}: the squeeze never ended`)
  for (const task of tasks) {
    if (task.squeezed_at !== first.squeezed_at || task.squeeze_ended !== first.squeeze_ended)
      throw new Error(`${file}: tasks were squeezed at different requests`)
  }

  // "expert cache: 1960 slots of up to 1.94 MiB, 49 a block, 3644 MiB in 4 slabs"
  const layout = raw.meta?.phobos?.expert_cache
  const slots = matchNumber(layout, /(\d+) slots/, "slot count")
  const mostPerBlock = matchNumber(layout, /(\d+) a block/, "slots a block")
  const slabMiB = matchNumber(layout, /(\d+) MiB in/, "slab size")

  // "holding 2048 MiB, ...; card 6.02 of 8.0 GiB used, ..., then 7.55 of 8.0 GiB used, ..."
  const squeezeMiB = matchNumber(first.squeeze, /holding (\d+) MiB/, "squeeze size")
  const cardGiB = matchNumber(first.squeeze, /of ([\d.]+) GiB/, "card size")
  const usedBeforeGiB = mean(phobos.map(task => matchNumber(task.squeeze, /card ([\d.]+) of/, "memory in use")))

  const sequences = phobos.map(task =>
    squeezeResizes(
      (task.resizes ?? []).map(([freeMiB, fromPerBlock, toPerBlock]) => ({ freeMiB, fromPerBlock, toPerBlock }))
    )
  )
  // A repetition can resize a step differently; the figure draws what most of them did.
  const counts = new Map<string, number>()
  for (const sequence of sequences) {
    const key = JSON.stringify(sequence)
    counts.set(key, (counts.get(key) ?? 0) + 1)
  }
  const [common, seen] = [...counts].reduce((best, entry) => (entry[1] > best[1] ? entry : best))
  if (seen * 2 <= sequences.length) throw new Error(`${file}: no resize sequence is shared by most repetitions`)
  const sequence = JSON.parse(common) as ReturnType<typeof squeezeResizes>

  const summarize = (engineTasks: RawTask[]): EngineSqueeze => {
    const rates = engineTasks.map(phaseRates)
    const engine = engineTasks[0].engine
    return {
      engine,
      label: engine.startsWith("phobos") ? "Phobos" : "llama.cpp",
      reps: engineTasks.length,
      wallSecs: round(median(engineTasks.map(task => task.wall_secs)), 3),
      decode: [0, 1, 2].map(phase => round(mean(rates.map(rate => rate[phase])), 3)) as [number, number, number],
    }
  }

  return {
    file,
    date,
    card,
    model,
    cardGiB,
    usedBeforeGiB: round(usedBeforeGiB, 2),
    squeezeMiB,
    requests: first.requests.length,
    squeezedAt: first.squeezed_at,
    squeezeEndedAt: first.squeeze_ended,
    cache: {
      blocks: slots / mostPerBlock,
      slotMiB: round(slabMiB / slots, 4),
      mostPerBlock,
      ...sequence,
    },
    engines: [summarize(phobos), summarize(others)],
  }
}

const DATES = committedDates as Record<string, string>

export function loadSqueezeRelease(root = repoRoot()): ElasticCacheData {
  const { file } = SQUEEZE_RELEASE
  return parseSqueezeRun(file, readFileSync(resolve(root, file), "utf8"), DATES[file], SQUEEZE_RELEASE)
}
