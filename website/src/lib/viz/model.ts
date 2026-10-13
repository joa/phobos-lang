// The visualizer's model of one trace: statement steps per CTA, the state at any step, derived facts
// (lines touched, coalescing) and the readout sentence. Pure, so it runs in Node and in vitest.
//
// Program order is the op table's array order; op ids are not sorted. A `for` or `if` op is its
// header: its body follows it, each body closing with a `yield` that carries the header's span, so a
// loop's extent is header to last yield. A ragged loop has two bodies (whole tiles, then the
// remainder), and a buffer allocated in one body lives to the end of that body, not the other.

import type {
  Access,
  Barrier,
  Buffer,
  Cta,
  LaneEvent,
  Op,
  Span,
  Step,
  Tensor,
  ThreadMap,
  Trace,
  Uniformity,
} from "./trace"

export const LINE_BYTES = 128
const KEYFRAME_EVERY = 8

export type Box = [number, number][]

export interface Statement {
  index: number
  cta: Cta
  iter: Record<string, number>
  span: Span
  /** The raw trace steps the statement groups, in order. */
  steps: Step[]
  /** For lockstep between variants: the statement's source text and iteration, not its line number. */
  key: string
}

export interface LineFacts {
  /** Distinct 128-byte lines warp 0 touches in its first pass over the access. */
  lines: number
  /** The fewest lines the same bytes could take. */
  idealLines: number
  requestedBytes: number
  /** Every line touched, in address order, with the bytes warp 0 uses of it. */
  touched: { line: number; usedBytes: number }[]
  /** For each lane of warp 0, the index into `touched` of each line its elements land in. */
  laneLines: number[][]
  /** For each lane of warp 0, the bytes of its first line it uses, as [from, to) within the line. */
  laneBytes: [number, number][]
}

export interface ThreadFacts {
  elements: number
  activeThreads: number
  threads: number
  perThread: number
  passes: number
  vectorBytes: number
  elem: string
  async: boolean
}

export interface MappedAccess {
  access: Access
  map: ThreadMap
  threads: ThreadFacts
  lines: LineFacts | null
  /** Elements of the box warp 0 takes in its first pass, per lane, as [row, col] pairs. */
  laneElements: [number, number][][]
}

export interface ValueKind {
  opId: number
  name: string
  kind: Uniformity
  line: number
}

export interface StepState {
  statement: Statement
  reads: Access[]
  writes: Access[]
  liveBuffers: number[]
  /** Regions of each out or inout tensor written so far by this CTA, this step included. */
  written: Record<string, Box[]>
  mapped: MappedAccess | null
  lanes: { opId: number; event: LaneEvent }[]
  barriers: { opId: number; kept: boolean }[]
  values: ValueKind[]
}

interface Keyframe {
  written: Record<string, Box[]>
}

export class KernelModel {
  readonly trace: Trace
  readonly lines: string[]
  #opById = new Map<number, Op>()
  #orderOf = new Map<number, number>()
  #endOf = new Map<number, number>()
  #bodies = new Map<number, [number, number][]>()
  #tensorByName = new Map<string, Tensor>()
  #statements: Statement[][] = []
  #keyframes: Keyframe[][] = []
  #values: ValueKind[]

  constructor(trace: Trace) {
    this.trace = trace
    this.lines = trace.source.text.replace(/\r\n?/g, "\n").split("\n")
    trace.ops.forEach((op, index) => {
      this.#opById.set(op.id, op)
      this.#orderOf.set(op.id, index)
    })
    for (const tensor of trace.tensors) this.#tensorByName.set(tensor.name, tensor)
    this.#indexRegions()
    this.#values = this.#namedValues()
    for (const cta of trace.ctas) {
      const statements = this.#groupStatements(cta)
      this.#statements.push(statements)
      this.#keyframes.push(this.#buildKeyframes(statements))
    }
  }

  get name(): string {
    return this.trace.name
  }

  get chip(): string {
    return this.trace.compiler.chip
  }

  get ctas(): Cta[] {
    return this.trace.ctas
  }

  get tensors(): Tensor[] {
    return this.trace.tensors
  }

  get buffers(): Buffer[] {
    return this.trace.buffers
  }

  get barriers(): Barrier[] {
    return this.trace.barriers ?? []
  }

  get sharedPeakBytes(): number {
    const placed = this.trace.buffers.map(buffer => (buffer.offsetBytes ?? 0) + buffer.bytes)
    return this.trace.sharedPeakBytes ?? Math.max(0, ...placed)
  }

  get opCount(): number {
    return this.trace.ops.length
  }

  /** Every named value of the kernel (`%row.16`), in program order, with its kind. */
  get values(): ValueKind[] {
    return this.#values
  }

  get hasLaneEvents(): boolean {
    return this.trace.steps.some(step => step.lanes !== undefined)
  }

  op(id: number): Op | undefined {
    return this.#opById.get(id)
  }

  tensor(name: string): Tensor | undefined {
    return this.#tensorByName.get(name)
  }

  buffer(id: number): Buffer | undefined {
    return this.trace.buffers.find(buffer => buffer.id === id)
  }

  /** The op's position in program order. */
  orderOf(opId: number): number {
    return this.#orderOf.get(opId) ?? -1
  }

  /** Where the op ends in program order: a `for` or `if` ends at its last yield. */
  endOf(opId: number): number {
    return this.#endOf.get(opId) ?? this.orderOf(opId)
  }

  /** A buffer's live interval in program order. */
  liveRange(buffer: Buffer): [number, number] {
    const start = this.orderOf(buffer.firstOp)
    const bodies = this.#bodies.get(buffer.lastOp)
    const enclosing = bodies?.find(([from, to]) => start >= from && start <= to)
    return [start, enclosing ? enclosing[1] : this.endOf(buffer.lastOp)]
  }

  /** Which body of a loop with several (a ragged loop: whole tiles, then the remainder) holds the op. */
  bodyOf(opId: number): number {
    const order = this.orderOf(opId)
    let body = 0
    for (const bodies of this.#bodies.values()) {
      if (bodies.length < 2) continue
      const index = bodies.findIndex(([from, to]) => order >= from && order <= to)
      if (index >= 0) body = index
    }
    return body
  }

  /** Where a barrier sits in program order: after its op, past the whole loop for a `for`. */
  barrierAt(barrier: Barrier): number {
    return this.endOf(barrier.afterOp)
  }

  ctaIndex(cta: Cta): number {
    return this.trace.ctas.findIndex(other => other.every((value, axis) => value === cta[axis]))
  }

  statements(ctaIndex: number): Statement[] {
    return this.#statements[ctaIndex] ?? []
  }

  stepCount(ctaIndex = 0): number {
    return this.statements(ctaIndex).length
  }

  stateAt(ctaIndex: number, stepIndex: number): StepState {
    const statements = this.statements(ctaIndex)
    const index = Math.max(0, Math.min(statements.length - 1, stepIndex))
    const statement = statements[index]
    const keyIndex = Math.floor(index / KEYFRAME_EVERY)
    const written = cloneWritten(this.#keyframes[ctaIndex][keyIndex].written)
    for (let at = keyIndex * KEYFRAME_EVERY; at <= index; at += 1) this.#applyWrites(written, statements[at])
    const reads = statement.steps.flatMap(step => step.reads ?? [])
    const writes = statement.steps.flatMap(step => step.writes ?? [])
    const position = this.orderOf(statement.steps[statement.steps.length - 1].op)
    const liveBuffers = this.trace.buffers
      .filter(buffer => {
        const [from, to] = this.liveRange(buffer)
        return position >= from && position <= to
      })
      .map(buffer => buffer.id)
    const mappedStep = statement.steps.find(step => step.map)
    const lanes = statement.steps.flatMap(step => (step.lanes ? [{ opId: step.op, event: step.lanes }] : []))
    const barriers = statement.steps.flatMap(step => (step.barrier ? [{ opId: step.op, kept: step.barrier.kept }] : []))
    const line = statement.span.line
    return {
      statement,
      reads,
      writes,
      liveBuffers,
      written,
      mapped: mappedStep ? this.mappedAccess(mappedStep) : null,
      lanes,
      barriers,
      values: this.statementValues(line),
    }
  }

  /** The values a source line computes, with their kinds; constants are left out. */
  statementValues(line: number): ValueKind[] {
    const named = this.#values.filter(value => value.line === line)
    if (named.length > 0) return named
    return this.trace.ops
      .filter(op => op.span?.line === line && op.uniformity && !QUIET_KINDS.has(op.kind))
      .map(op => ({ opId: op.id, name: op.kind, kind: op.uniformity as Uniformity, line }))
  }

  /** The access a thread map describes: the first one whose box has the map's shape. */
  mappedAccess(step: Step): MappedAccess | null {
    const map = step.map
    if (!map) return null
    const accesses = [...(step.reads ?? []), ...(step.writes ?? [])]
    // Several accesses can have the map's shape; the vector width tells which element type it moves.
    const shaped = accesses.filter(candidate => sameShape(boxShape(candidate.box), map.shape))
    const fits = (candidate: Access) => this.#elemOf(candidate).bytes * map.width === map.vectorBytes
    const access =
      shaped.find(candidate => candidate.tensor && fits(candidate)) ??
      shaped.find(candidate => candidate.tensor) ??
      shaped.find(fits) ??
      shaped[0]
    if (!access) return null
    const elements = map.shape.reduce((product, extent) => product * extent, 1)
    const perPass = map.threads * map.width
    const passes = Math.max(1, Math.ceil(elements / perPass))
    const activeThreads = Math.min(map.threads, Math.ceil(elements / map.width))
    const laneElements: [number, number][][] = []
    const cols = map.shape[map.shape.length - 1]
    for (let lane = 0; lane < this.trace.launch.lanes; lane += 1) {
      const owned: [number, number][] = []
      for (let offset = 0; offset < map.width; offset += 1) {
        const flat = lane * map.width + offset
        if (flat >= elements) break
        owned.push([Math.floor(flat / cols), flat % cols])
      }
      laneElements.push(owned)
    }
    const elem = this.#elemOf(access)
    return {
      access,
      map,
      threads: {
        elements,
        activeThreads,
        threads: map.threads,
        perThread: map.width,
        passes,
        vectorBytes: map.vectorBytes,
        elem: elem.name,
        async: map.async ?? false,
      },
      lines: access.tensor ? this.#warpLines(access, laneElements) : null,
      laneElements,
    }
  }

  /** The 128-byte lines warp 0 touches with a lane event's runs. */
  laneRunLines(event: LaneEvent): LineFacts | null {
    const runs = event.runs ?? []
    if (runs.length === 0 || !runs[0].access.tensor) return null
    const laneElements: [number, number][][] = runs.map(run => boxElements(run.access.box))
    const origin = runs.map(run => run.access)
    return this.#linesFor(origin[0].tensor ?? "", laneElements)
  }

  /** Each CTA's accesses at the same step index, for the all-CTAs view. */
  accessesAcross(stepIndex: number): { ctaIndex: number; reads: Access[]; writes: Access[] }[] {
    return this.trace.ctas.map((_, ctaIndex) => {
      const statements = this.statements(ctaIndex)
      const statement = statements[Math.min(stepIndex, statements.length - 1)]
      return {
        ctaIndex,
        reads: statement?.steps.flatMap(step => step.reads ?? []) ?? [],
        writes: statement?.steps.flatMap(step => step.writes ?? []) ?? [],
      }
    })
  }

  /** The step of another CTA (or of a variant's model) that matches a statement key, or null. */
  findByKey(ctaIndex: number, key: string): number | null {
    const index = this.statements(ctaIndex).findIndex(statement => statement.key === key)
    return index < 0 ? null : index
  }

  /** The same statement one loop iteration later (+1) or earlier (-1), else the loop's edge. */
  iterationStep(ctaIndex: number, stepIndex: number, direction: 1 | -1): number {
    const statements = this.statements(ctaIndex)
    const current = statements[stepIndex]
    if (!current || Object.keys(current.iter).length === 0) {
      return Math.max(0, Math.min(statements.length - 1, stepIndex + direction))
    }
    const sameSpan = (statement: Statement) =>
      statement.span.line === current.span.line && statement.span.col === current.span.col
    for (let at = stepIndex + direction; at >= 0 && at < statements.length; at += direction) {
      if (sameSpan(statements[at])) return at
    }
    // No other iteration: leave the loop, to the first statement past it or before it.
    for (let at = stepIndex + direction; at >= 0 && at < statements.length; at += direction) {
      if (Object.keys(statements[at].iter).length === 0) return at
    }
    return direction > 0 ? statements.length - 1 : 0
  }

  readout(ctaIndex: number, stepIndex: number): string {
    return describe(this, this.stateAt(ctaIndex, stepIndex))
  }

  ctaName(cta: Cta): string {
    return ctaName(cta, this.trace.launch.grid)
  }

  bufferName(id: number): string {
    return this.buffer(id)?.name ?? `buffer ${id}`
  }

  #indexRegions(): void {
    const ops = this.trace.ops
    ops.forEach((op, index) => {
      if (op.kind !== "for" && op.kind !== "if") return
      const yields: number[] = []
      ops.forEach((other, at) => {
        if (at > index && other.kind === "yield" && other.span && op.span && sameSpan(other.span, op.span)) {
          yields.push(at)
        }
      })
      if (yields.length === 0) return
      this.#endOf.set(op.id, yields[yields.length - 1])
      const bodies: [number, number][] = []
      let from = index + 1
      for (const at of yields) {
        bodies.push([from, at])
        from = at + 1
      }
      this.#bodies.set(op.id, bodies)
    })
  }

  #namedValues(): ValueKind[] {
    const values: ValueKind[] = []
    const seen = new Set<string>()
    for (const op of this.trace.ops) {
      if (!op.span || !op.uniformity) continue
      const match = /^%([A-Za-z_]\w*)\.\d+\s*:/.exec(op.ir)
      if (!match || seen.has(match[1])) continue
      seen.add(match[1])
      values.push({ opId: op.id, name: match[1], kind: op.uniformity, line: op.span.line })
    }
    return values
  }

  #groupStatements(cta: Cta): Statement[] {
    const statements: Statement[] = []
    const sameCta = (step: Step) => step.cta.every((value, axis) => value === cta[axis])
    for (const step of this.trace.steps) {
      if (!sameCta(step)) continue
      const span = this.#opById.get(step.op)?.span
      if (!span) continue
      const last = statements[statements.length - 1]
      if (last && sameSpan(last.span, span) && sameIter(last.iter, step.iter)) {
        last.steps.push(step)
        continue
      }
      const text = (this.lines[span.line - 1] ?? "").trim()
      statements.push({
        index: statements.length,
        cta,
        iter: step.iter,
        span,
        steps: [step],
        key: `${text}|${JSON.stringify(sortedIter(step.iter))}`,
      })
    }
    return statements
  }

  #buildKeyframes(statements: Statement[]): Keyframe[] {
    const keyframes: Keyframe[] = [{ written: {} }]
    const written: Record<string, Box[]> = {}
    statements.forEach((statement, index) => {
      if (index > 0 && index % KEYFRAME_EVERY === 0) keyframes.push({ written: cloneWritten(written) })
      this.#applyWrites(written, statement)
    })
    return keyframes
  }

  #applyWrites(written: Record<string, Box[]>, statement: Statement): void {
    for (const step of statement.steps) {
      for (const access of step.writes ?? []) {
        if (!access.tensor) continue
        const boxes = (written[access.tensor] ??= [])
        if (!boxes.some(box => sameBox(box, access.box) || contains(box, access.box))) boxes.push(access.box)
      }
    }
  }

  #elemOf(access: Access): { name: string; bytes: number } {
    if (access.tensor) {
      const tensor = this.tensor(access.tensor)
      if (tensor) return { name: tensor.elem, bytes: tensor.elemBytes }
    }
    if (access.buffer !== undefined) {
      const buffer = this.buffer(access.buffer)
      if (buffer?.elem) return { name: buffer.elem, bytes: elemBytesOf(buffer.elem) }
    }
    const first = this.trace.tensors[0]
    return { name: first?.elem ?? "f32", bytes: first?.elemBytes ?? 4 }
  }

  #warpLines(access: Access, laneElements: [number, number][][]): LineFacts | null {
    const origin = access.box.map(range => range[0])
    const absolute = laneElements.map(elements =>
      elements.map(([row, col]): [number, number] =>
        origin.length === 1 ? [0, origin[0] + col] : [origin[0] + row, origin[1] + col]
      )
    )
    return this.#linesFor(access.tensor ?? "", absolute)
  }

  // Assumes the tensor is dense and row-major and starts on a 128-byte boundary, as an allocation does.
  #linesFor(tensorName: string, laneElements: [number, number][][]): LineFacts | null {
    const tensor = this.tensor(tensorName)
    if (!tensor) return null
    const cols = tensor.shape[tensor.shape.length - 1]
    const lineBytes = new Map<number, number>()
    const laneLineNumbers: number[][] = []
    const laneBytes: [number, number][] = []
    let requestedBytes = 0
    for (const elements of laneElements) {
      const lines: number[] = []
      let span: [number, number] = [0, 0]
      for (const [row, col] of elements) {
        const address = (row * cols + col) * tensor.elemBytes
        const line = Math.floor(address / LINE_BYTES)
        lineBytes.set(line, (lineBytes.get(line) ?? 0) + tensor.elemBytes)
        requestedBytes += tensor.elemBytes
        if (!lines.includes(line)) lines.push(line)
        const within = address - line * LINE_BYTES
        if (line === lines[0]) {
          const end = within + tensor.elemBytes
          span = span[1] === 0 ? [within, end] : [Math.min(span[0], within), Math.max(span[1], end)]
        }
      }
      laneLineNumbers.push(lines)
      laneBytes.push(span)
    }
    const touched = [...lineBytes.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([line, usedBytes]) => ({ line, usedBytes }))
    const slotOf = new Map(touched.map((entry, slot) => [entry.line, slot]))
    return {
      lines: touched.length,
      idealLines: Math.ceil(requestedBytes / LINE_BYTES),
      requestedBytes,
      touched,
      laneLines: laneLineNumbers.map(lines => lines.map(line => slotOf.get(line) ?? 0)),
      laneBytes,
    }
  }
}

const QUIET_KINDS = new Set(["const", "yield", "dim", "assume_align"])

export function sameSpan(a: Span, b: Span): boolean {
  return a.line === b.line && a.col === b.col && a.endLine === b.endLine && a.endCol === b.endCol
}

function sameIter(a: Record<string, number>, b: Record<string, number>): boolean {
  const keys = Object.keys(a)
  return keys.length === Object.keys(b).length && keys.every(key => a[key] === b[key])
}

function sortedIter(iter: Record<string, number>): [string, number][] {
  return Object.entries(iter).sort((a, b) => a[0].localeCompare(b[0]))
}

export function boxShape(box: Box): number[] {
  return box.map(([from, to]) => to - from)
}

function sameShape(a: number[], b: number[]): boolean {
  const trim = (shape: number[]) => {
    const kept = [...shape]
    while (kept.length > 1 && kept[0] === 1) kept.shift()
    return kept
  }
  const left = trim(a)
  const right = trim(b)
  return left.length === right.length && left.every((extent, axis) => extent === right[axis])
}

function sameBox(a: Box, b: Box): boolean {
  return a.length === b.length && a.every(([from, to], axis) => from === b[axis][0] && to === b[axis][1])
}

function contains(outer: Box, inner: Box): boolean {
  return (
    outer.length === inner.length && outer.every(([from, to], axis) => from <= inner[axis][0] && to >= inner[axis][1])
  )
}

function cloneWritten(written: Record<string, Box[]>): Record<string, Box[]> {
  return Object.fromEntries(Object.entries(written).map(([name, boxes]) => [name, [...boxes]]))
}

// Every element of a box, row-major, as [row, col]; a 1-D box has row 0.
function boxElements(box: Box): [number, number][] {
  const elements: [number, number][] = []
  if (box.length === 1) {
    for (let col = box[0][0]; col < box[0][1]; col += 1) elements.push([0, col])
    return elements
  }
  for (let row = box[0][0]; row < box[0][1]; row += 1) {
    for (let col = box[1][0]; col < box[1][1]; col += 1) elements.push([row, col])
  }
  return elements
}

function elemBytesOf(elem: string): number {
  const bits = /(\d+)$/.exec(elem)
  return bits ? Math.max(1, Number(bits[1]) / 8) : 4
}

/** `A[0:32, 4:8]`, with an extent of one printed as the index alone: `y[0, 3]`. */
export function formatBox(name: string, box: Box): string {
  return `${name}[${box.map(([from, to]) => (to - from === 1 ? `${from}` : `${from}:${to}`)).join(", ")}]`
}

/** `CTA 3` on a one-dimensional grid, `CTA (1, 0)` on a two-dimensional one. */
export function ctaName(cta: Cta, grid: Cta): string {
  const axes = grid[2] > 1 ? 3 : grid[1] > 1 ? 2 : 1
  return axes === 1 ? `CTA ${cta[0]}` : `CTA (${cta.slice(0, axes).join(", ")})`
}

export function iterLabel(iter: Record<string, number>): string {
  return sortedIter(iter)
    .map(([name, value]) => `${name} = ${value}`)
    .join(", ")
}

function bytesRange([from, to]: [number, number]): string {
  return `bytes ${from} to ${to - 1}`
}

function describeAccess(model: KernelModel, access: Access): string {
  if (access.tensor !== undefined) {
    const masked = access.maskedElems ? ` (${access.maskedElems} masked off)` : ""
    return formatBox(access.tensor, access.box) + masked
  }
  const name = model.bufferName(access.buffer ?? -1)
  return access.bytes ? `shared ${name} (${bytesRange(access.bytes)})` : `shared ${name}`
}

// How each op kind reads in a sentence: `into` joins reads to writes, `lead` alone names the op.
const KIND_PHRASES: Record<string, { verb: string; into?: string }> = {
  materialize: { verb: "copies", into: "into" },
  copy: { verb: "copies", into: "into" },
  stage: { verb: "stages", into: "into" },
  chain: { verb: "converts", into: "into" },
  fill: { verb: "fills" },
  fused: { verb: "computes elementwise from", into: "into" },
  dot_into: { verb: "multiplies", into: "into" },
  dot_fused: { verb: "multiplies", into: "into" },
  reduce: { verb: "reduces", into: "into" },
  accumulate: { verb: "adds", into: "into" },
  scaled_add: { verb: "computes alpha * acc + beta * c_old from", into: "into" },
  gemm_dot: { verb: "runs a GEMM k step" },
  gemm_store: { verb: "runs the GEMM epilogue" },
  store: { verb: "stores", into: "into" },
  load: { verb: "loads", into: "into" },
}

function describeStep(model: KernelModel, step: Step): string {
  if (step.lanes) return describeLanes(model, step.lanes)
  const op = model.op(step.op)
  const kind = op?.kind ?? "op"
  const reads = (step.reads ?? []).map(access => describeAccess(model, access))
  const writes = (step.writes ?? []).map(access => describeAccess(model, access))
  const phrase = KIND_PHRASES[kind] ?? { verb: kind.replace(/_/g, " ") }
  const on = op?.instr ? ` on ${op.instr}` : ""
  if (phrase.into && reads.length > 0 && writes.length > 0) {
    return `${phrase.verb} ${joinList(reads)} ${phrase.into} ${joinList(writes)}${on}`
  }
  if (reads.length === 0 && writes.length > 0 && (phrase.into || phrase.verb === "fills")) {
    return `${phrase.verb} ${joinList(writes)}${on}`
  }
  let clause = `${phrase.verb}${on}`
  if (reads.length > 0) clause += `: reads ${joinList(reads)}`
  if (writes.length > 0) clause += `${reads.length > 0 ? " and" : ":"} writes ${joinList(writes)}`
  return clause
}

function describeLanes(model: KernelModel, event: LaneEvent): string {
  const lanes = model.trace.launch.lanes
  if (event.kind === "collective" && event.stages) {
    const distances = event.stages.map(stage => stage[0])
    const pieces = event.width < lanes ? `, in pieces of ${event.width} lanes` : ""
    return `the lane collective folds warp 0's ${lanes} values in ${event.stages.length} ${event.mode ?? "xor"} shuffle stages (${distances.join(", ")})${pieces}, and every lane ends with the result`
  }
  if ((event.kind === "lane_load" || event.kind === "lane_store") && event.runs && event.runs.length > 0) {
    const first = event.runs[0].access
    const runLength = boxShape(first.box).reduce((product, extent) => product * extent, 1)
    const tensor = first.tensor ? model.tensor(first.tensor) : undefined
    const elem = tensor?.elem ?? "value"
    const all = event.runs.map(run => run.access.box)
    const union: Box = all[0].map((_, axis) => [
      Math.min(...all.map(box => box[axis][0])),
      Math.max(...all.map(box => box[axis][1])),
    ])
    const verb = event.kind === "lane_load" ? "load" : "store"
    const vector = event.vectorBytes ? ` as one ${event.vectorBytes}-byte vector` : ""
    const lines = model.laneRunLines(event)
    const lineText = lines
      ? `; ${plural(lines.lines, "line")} of ${LINE_BYTES} bytes for ${lines.requestedBytes} bytes`
      : ""
    return `warp 0's ${event.runs.length} lanes each ${verb} ${runLength} ${elem} of ${formatBox(first.tensor ?? "?", union)}${vector}${lineText}`
  }
  if (event.kind === "shuffle")
    return `a ${event.mode ?? ""} shuffle moves one value between lanes of warp 0`.replace("  ", " ")
  return `${event.kind.replace(/_/g, " ")} across warp 0`
}

function describeThreads(facts: ThreadFacts, elemBytes: number): string {
  const who =
    facts.activeThreads < facts.threads
      ? `${facts.activeThreads} of ${facts.threads} threads`
      : `${facts.threads} threads`
  const each =
    facts.passes > 1
      ? `${facts.perThread} ${facts.elem} each per pass, ${facts.passes} passes`
      : `${facts.perThread} ${facts.elem} each`
  const how = facts.async
    ? `, ${facts.vectorBytes}-byte cp.async copies`
    : facts.vectorBytes > elemBytes
      ? `, ${facts.vectorBytes}-byte vectors`
      : ""
  return `${who}, ${each}${how}`
}

function describe(model: KernelModel, state: StepState): string {
  const { statement } = state
  const iter = iterLabel(statement.iter)
  const who = `${model.ctaName(statement.cta)}${iter ? `, ${iter}` : ""}`
  const clauses = statement.steps.map(step => describeStep(model, step))
  let sentence = `${who}: ${clauses.join("; then ")}`
  if (state.mapped) {
    const tensor = state.mapped.access.tensor ? model.tensor(state.mapped.access.tensor) : undefined
    const elemBytes = tensor?.elemBytes ?? elemBytesOf(state.mapped.threads.elem)
    const several = state.reads.length + state.writes.length > 1 && state.mapped.access.tensor
    const lead = several ? `${state.mapped.access.tensor}: ` : ""
    sentence += `; ${lead}${describeThreads(state.mapped.threads, elemBytes)}`
  }
  sentence += "."
  const lines = state.mapped?.lines
  if (lines && lines.lines > lines.idealLines) {
    sentence += ` Warp 0 touches ${plural(lines.lines, "line")} of ${LINE_BYTES} bytes for ${lines.requestedBytes} bytes; ${lines.idealLines} would hold them.`
  }
  const kept = state.barriers.filter(barrier => barrier.kept).length
  const elided = state.barriers.length - kept
  if (state.barriers.length === 1) sentence += kept ? " A barrier follows." : " The barrier after it is elided."
  else if (state.barriers.length > 1)
    sentence += ` ${state.barriers.length} barriers follow its ops: ${kept} kept, ${elided} elided.`
  return sentence
}

function joinList(items: string[]): string {
  if (items.length <= 1) return items.join("")
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? "" : "s"}`
}
