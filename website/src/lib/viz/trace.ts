// The trace format: the contract between the compiler's exporter
// (`cargo run -p phobos-lang --features trace --example site -- traces`) and the
// visualizer. Version 1. Additions are optional fields only; anything else bumps
// `schema`.
//
// Conventions:
// - Every box is half-open per dimension: [[row0, row1], [col0, col1]].
// - Line and column numbers are 1-based, as the lexer counts them. A span's end
//   is exclusive, and a compound statement's span is its header, up to its `{`.
// - `sha256` is over the source text with CRLF normalized to LF. For a variant
//   it is over the original text (normalized the same way) followed by the
//   variant as compact JSON, `add` before `drop`, such as `{"add":"@padstage"}`.
//   For a fence it is over `source.text` exactly: the fence body with the
//   opening fence's indentation stripped, lines joined by LF, one trailing LF.
// - The grid is not in the kernel: the host picks it. It comes from
//   `website/kernels.json` or a fence's `grid` meta.
// - Placement fields (`Buffer.offsetBytes`, `barriers`) are optional so a future
//   in-browser build without the MLIR emitter can still produce a trace.

export type Uniformity = "uniform" | "warp" | "varying"

export interface Trace {
  schema: 1
  /** Trace name, also the file stem: `<kernel>.<chip>.json`. */
  name: string
  /** The kernel's name in the source. */
  kernel: string
  source: Source
  compiler: { commit: string; chip: string; indexBits: number }
  /** Tile constants as compiled (from `@autotune` overrides). */
  params: Record<string, number>
  /** Runtime dimensions the trace was evaluated with. */
  dims: Record<string, number>
  launch: Launch
  tensors: Tensor[]
  ops: Op[]
  buffers: Buffer[]
  barriers?: Barrier[]
  sharedPeakBytes?: number
  /** CTAs traced in full; every other CTA appears only in `summaries`. */
  ctas: Cta[]
  steps: Step[]
  summaries?: Summary[]
}

export interface Source {
  origin: "file" | "fence"
  /** Repo-relative path: the `.ph` file, or the docs page holding the fence. */
  path: string
  /** For a fence, its `trace` name; absent for a file. */
  fence?: string
  /** The text exactly as compiled (after a variant was applied). */
  text: string
  sha256: string
  /** Lines the code panel hides (for example `@cluster`), 1-based. */
  hiddenLines?: number[]
  variant?: { add?: string; drop?: string }
}

export interface Launch {
  grid: Cta
  threads: number
  lanes: number
  warps: number
}

export type Cta = [number, number, number]

export interface Tensor {
  name: string
  elem: string
  elemBytes: number
  shape: number[]
  role: "in" | "out" | "inout"
}

export interface Span {
  line: number
  col: number
  endLine: number
  endCol: number
}

export interface Op {
  id: number
  /** A short stable kind, such as "slice_copy", "dot", "store", "shuffle", "lane_load". */
  kind: string
  span?: Span
  /** The op printed by `ir/print.rs`. */
  ir: string
  /** How far the op's result may differ between threads (`ir/uniform.rs`). */
  uniformity?: Uniformity
  /** The instruction family a contraction lowered to: "wmma", "mma.sync" or "fma". */
  instr?: string
}

export interface Buffer {
  id: number
  /** The source name of the tile, or the pool's name for emitter scratch. */
  name: string
  bytes: number
  offsetBytes?: number
  /** First and last op (by id) the buffer is live across. */
  firstOp: number
  lastOp: number
  /** Scratch the emitter allocates and the graph never sees (dot staging, the matmul slab). */
  scratch: boolean
  elem?: string
  shape?: number[]
}

export interface Barrier {
  /** The op whose trailing barrier this is. */
  afterOp: number
  kept: boolean
}

/** A region of a tensor or a shared buffer. */
export interface Access {
  tensor?: string
  buffer?: number
  box: [number, number][]
  /** For shared buffers, the byte range touched. */
  bytes?: [number, number]
  /** Elements masked off by bounds checks inside `box`. */
  maskedElems?: number
}

/**
 * How one access is spread over the CTA's threads, as the emitter does it: the
 * region is walked in row-major order, `width` contiguous elements per thread per
 * pass, thread `t` starting at element `t * width`, the CTA advancing by
 * `threads * width` elements per pass.
 */
export interface ThreadMap {
  shape: number[]
  width: number
  threads: number
  /** Vector access width in bytes (1 for a load per element). */
  vectorBytes: number
  /** cp.async staging, when the copy is asynchronous. */
  async?: boolean
}

export interface LaneEvent {
  kind: "shuffle" | "collective" | "exchange" | "lane_load" | "lane_store" | "dp4a"
  mode?: "xor" | "down" | "up" | "from"
  /** Lanes per piece (`width w`), 32 for the whole warp. */
  width: number
  /** For a shuffle: the source lane of every lane of warp 0. */
  partners?: number[]
  /** For a collective: each stage's partners, one array of `lanes` per stage. */
  stages?: number[][]
  /** For lane loads and stores: each lane's run within warp 0. */
  runs?: { lane: number; access: Access }[]
  vectorBytes?: number
  alignBytes?: number
  /** For an exchange: the CTA barriers it takes. */
  barriers?: number
}

export interface Step {
  op: number
  cta: Cta
  /** Loop variables by name at this step. */
  iter: Record<string, number>
  reads?: Access[]
  writes?: Access[]
  map?: ThreadMap
  lanes?: LaneEvent
  barrier?: { kept: boolean }
}

export interface Summary {
  cta: Cta
  op: number
  iter: Record<string, number>
  reads?: Access[]
  writes?: Access[]
}
