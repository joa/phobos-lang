// What the front page's tiles-and-lanes scene draws, read from the compiler's traces so the picture
// matches what was compiled: the gemm's tile sizes, thread count and k steps, the lanes matvec's
// per-lane runs and the xor masks its sum<over ctx.lane> lowers to, and the staged matvec's tile
// copy into shared memory with the runs its lanes then read from that tile. Pure; Astro frontmatter
// calls it and passes the result to the page as JSON, so no trace ships to the browser.

import type { Trace } from "../../../lib/viz/trace"

export interface TilesData {
  kernel: string
  chip: string
  M: number
  N: number
  K: number
  tileM: number
  tileN: number
  tileK: number
  threads: number
  // The k loop's steps for CTA (0, 0), in order.
  kts: number[]
}

export interface LanesData {
  kernel: string
  chip: string
  K: number
  lanes: number
  // Elements each lane reads per k step, and where lane 0's run starts in each step.
  runElems: number
  kSteps: number[]
  // The xor mask of each stage of the collective, in order.
  masks: number[]
}

export interface MixedData {
  kernel: string
  chip: string
  K: number
  threads: number
  lanes: number
  warps: number
  // The tile copy: elements staged, elements per thread per pass, the threads that take part, and
  // whether a barrier follows it.
  tileElems: number
  copyWidth: number
  copyThreads: number
  copyBarrier: boolean
  // Per k step, where each lane of warp 0 starts its run in the shared tile and in its row of W.
  runElems: number
  kSteps: number[]
  tileRuns: number[][]
  weightRuns: number[][]
  masks: number[]
}

export interface CodeExcerpt {
  lines: string[]
  // Line index (0-based) of each beat moment within `lines`.
  marks: Record<string, number>
}

export interface SceneData {
  tiles: TilesData
  lanes: LanesData
  mixed: MixedData
  tilesCode: CodeExcerpt
  lanesCode: CodeExcerpt
  mixedCode: CodeExcerpt
}

export function tilesData(trace: Trace): TilesData {
  const firstCta = trace.ctas[0]
  const kts = trace.steps
    .filter(step => step.cta.join() === firstCta.join() && typeof step.iter?.kt === "number")
    .map(step => step.iter.kt as number)
  return {
    kernel: trace.name,
    chip: trace.compiler.chip,
    M: trace.dims.M,
    N: trace.dims.N,
    K: trace.dims.K,
    tileM: trace.params.TILE_M,
    tileN: trace.params.TILE_N,
    tileK: trace.params.TILE_K,
    threads: trace.launch.threads,
    kts: [...new Set(kts)],
  }
}

// The mask of an xor stage, checked against every lane: partners[i] must be i ^ mask.
export function xorMask(partners: number[]): number {
  const mask = partners[0] ^ 0
  partners.forEach((partner, lane) => {
    if (partner !== (lane ^ mask)) throw new Error(`lane ${lane} pairs with ${partner}, not ${lane ^ mask}`)
  })
  return mask
}

export function lanesData(trace: Trace): LanesData {
  const firstCta = trace.ctas[0].join()
  const steps = trace.steps.filter(step => step.cta.join() === firstCta)
  const collective = steps.find(step => step.lanes?.kind === "collective" && step.lanes.stages)
  if (!collective?.lanes?.stages) throw new Error(`${trace.name} has no collective`)
  const loads = steps.filter(step => step.lanes?.kind === "lane_load" && step.lanes.runs)
  if (loads.length === 0) throw new Error(`${trace.name} has no lane loads`)
  const lane0 = (step: (typeof loads)[number]) => step.lanes!.runs!.find(run => run.lane === 0)!.access.box
  const box = lane0(loads[0])
  const runElems = box[1][1] - box[1][0]
  const kSteps = [...new Set(loads.map(step => lane0(step)[1][0]))]
  return {
    kernel: trace.name,
    chip: trace.compiler.chip,
    K: trace.dims.K,
    lanes: trace.launch.lanes,
    runElems,
    kSteps,
    masks: collective.lanes.stages.map(xorMask),
  }
}

export function mixedData(trace: Trace): MixedData {
  const firstCta = trace.ctas[0].join()
  const steps = trace.steps.filter(step => step.cta.join() === firstCta)
  const kindOf = new Map(trace.ops.map(op => [op.id, op.kind]))
  const copy = steps.find(step => kindOf.get(step.op) === "stage" && step.map && step.writes?.some(w => w.buffer !== undefined))
  if (!copy?.map) throw new Error(`${trace.name} stages no tile`)
  const tileBuffer = copy.writes!.find(w => w.buffer !== undefined)!.buffer!
  const collective = steps.find(step => step.lanes?.kind === "collective" && step.lanes.stages)
  if (!collective?.lanes?.stages) throw new Error(`${trace.name} has no collective`)
  const loads = steps.filter(step => step.lanes?.kind === "lane_load" && step.lanes.runs)
  const fromTile = loads.filter(step => step.lanes!.runs!.every(run => run.access.buffer === tileBuffer))
  const fromWeights = loads.filter(step => step.lanes!.runs!.every(run => run.access.tensor === "W"))
  if (fromTile.length === 0 || fromTile.length !== fromWeights.length)
    throw new Error(`${trace.name}: lanes read ${fromTile.length} runs of the tile and ${fromWeights.length} of W`)
  const starts = (step: (typeof loads)[number]) =>
    [...step.lanes!.runs!].sort((a, b) => a.lane - b.lane).map(run => run.access.box[1][0])
  const box = fromTile[0].lanes!.runs![0].access.box
  const elems = copy.map.shape.reduce((product, extent) => product * extent, 1)
  return {
    kernel: trace.name,
    chip: trace.compiler.chip,
    K: trace.params.K,
    threads: trace.launch.threads,
    lanes: trace.launch.lanes,
    warps: trace.launch.warps,
    tileElems: elems,
    copyWidth: copy.map.width,
    copyThreads: Math.min(copy.map.threads, elems / copy.map.width),
    copyBarrier: copy.barrier?.kept === true,
    runElems: box[1][1] - box[1][0],
    kSteps: fromTile.map(step => step.iter.k),
    tileRuns: fromTile.map(starts),
    weightRuns: fromWeights.map(starts),
    masks: collective.lanes.stages.map(xorMask),
  }
}

// Source lines picked in order, each the first line after the previous pick whose trimmed text starts
// with its prefix, dedented by the first pick's indent; a line indented less starts at column 0. A
// mark names the first picked line containing its text.
export function excerpt(source: string, prefixes: string[], marks: Record<string, string>): CodeExcerpt {
  const all = source.split("\n")
  const picked: string[] = []
  let from = 0
  for (const prefix of prefixes) {
    const index = all.findIndex((line, at) => at >= from && line.trim().startsWith(prefix))
    if (index < 0) throw new Error(`no line starting with "${prefix}"`)
    picked.push(all[index])
    from = index + 1
  }
  const indentOf = (line: string) => line.length - line.trimStart().length
  const indent = indentOf(picked[0])
  const lines = picked.map(line => line.slice(Math.min(indent, indentOf(line))).trimEnd())
  const found: Record<string, number> = {}
  for (const [name, text] of Object.entries(marks)) {
    const index = lines.findIndex(line => line.includes(text))
    if (index < 0) throw new Error(`no line with "${text}"`)
    found[name] = index
  }
  return { lines, marks: found }
}

export function sceneData(gemm: Trace, matvec: Trace, staged: Trace): SceneData {
  return {
    tiles: tilesData(gemm),
    lanes: lanesData(matvec),
    mixed: mixedData(staged),
    tilesCode: excerpt(gemm.source.text, ["for kt", "var a", "var b", "acc += dot", "}"], {
      a: "var a =",
      b: "var b =",
      dot: "acc += dot(",
    }),
    lanesCode: excerpt(matvec.source.text, ["for k in", "let at", "acc += sum", "}", "let total"], {
      at: "let at =",
      acc: "acc += sum(",
      sum: "sum<over ctx.lane>",
    }),
    mixedCode: excerpt(
      staged.source.text,
      ["var xs", "if row", "for k in", "let at", "acc += sum", "}", "}", "let total"],
      { stage: "var xs =", at: "let at =", acc: "acc += sum(", sum: "sum<over ctx.lane>" }
    ),
  }
}
