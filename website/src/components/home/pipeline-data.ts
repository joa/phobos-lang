// The shape of src/data/pipeline/gemm_fp32.json, which scripts/pipeline-excerpts.mjs writes from real
// compiler output for examples/gemm_fp32.ph. `stages` is the text excerpt per station; `structure` is
// the same output as data to draw each station from. Types and pure helpers only.

export interface PipelineData {
  /** Station id to a verbatim excerpt of the compiler's output. */
  stages: Record<string, string>
  structure: PipelineStructure
}

export interface PipelineStructure {
  source: SourceView
  ast: AstView
  graph: GraphView
  /** One entry per trace; the `recommended` one is the one to animate. */
  passes: PassesView[]
  mlir: MlirView
  ptx: PtxView
  gpu: GpuView
}

export type SourceRole = "attribute" | "signature" | "code" | "blank" | "loop-header" | "loop-body" | "loop-end"

export interface SourceView {
  path: string
  /** Line numbers are 1-based. */
  lines: { n: number; text: string; role: SourceRole }[]
  loop: { header: number; body: number[]; end: number }
}

export interface AstNode {
  /** The AST variant, such as For, Var, Index, Span, Binary, Call; Kernel, Attributes, Params at the top. */
  kind: string
  /** Short: a name, an operator, a literal. */
  label: string
  /** The field of the parent this node fills, such as start, end, step, body, value, target, len. */
  role?: string
  /** Source line of a statement. */
  line?: number
  /** Number of descendants pruned away below this node. */
  folded?: number
  /** For a folded node, its source spelling. */
  text?: string
  children?: AstNode[]
}

export interface AstView {
  root: AstNode
  nodeCount: number
  /** The kind of the subtree kept whole. */
  focus: string
  pruned: string
}

export type GraphNodeKind = "param" | "op" | "math" | "guard" | "for" | "block_arg" | "yield" | "for_result"

export interface GraphNode {
  /** The IR's value name (`%A.5`), `#n` for an op without a result, or a prefixed id for a synthetic node. */
  id: string
  kind: GraphNodeKind
  /** The IR op, or a description for a synthetic node ("index math", "if trap", "block arg", "for result"). */
  op: string
  label: string
  /** The result type, short (`@div[..]` and `{align ..}` dropped); null for ops without a result. */
  type: string | null
  /** The id of the region the node lives in (the `for`), null at the kernel's top level. */
  region: string | null
  /** Constants inlined as operands. */
  literals?: string[]
  /** For `math` and `guard`: the IR values folded into this node. */
  folds?: string[]
  layer: number
  row: number
  /** Layout coordinates in units of the layout's `columnWidth` and `rowHeight`; the node's center. */
  x: number
  y: number
}

export interface GraphEdge {
  from: string
  to: string
  /** `back`: the loop-carried value, from the yield to the block argument; not part of the layering. */
  kind: "data" | "back"
}

export interface GraphRegion {
  id: string
  kind: "for"
  /** The header node, left of the box, and the result node, right of it. */
  header: string
  result: string
  label: string
  /** The box around the region's nodes, in layout coordinates, half a cell of margin on each side. */
  box: {
    x: number
    y: number
    width: number
    height: number
    firstLayer: number
    lastLayer: number
    firstRow: number
    lastRow: number
  }
}

export interface GraphView {
  kernel: string
  nodes: GraphNode[]
  edges: GraphEdge[]
  regions: GraphRegion[]
  pruned: { opsPrinted: number; foldedOps: number; foldedIds: string[]; note: string }
  layout: {
    method: string
    direction: "left-to-right"
    columnWidth: number
    rowHeight: number
    layers: number
    rows: number
    /** Extent of the node centers: x in [0, width], y in [0, height]. */
    width: number
    height: number
  }
}

export interface PassBuffer {
  id: number
  name: string
  bytes: number
  offsetBytes: number
  /** Live from op `firstOp` to op `lastOp`, by trace op id. */
  firstOp: number
  lastOp: number
  /** Emitter scratch, such as dot staging, which the graph never sees. */
  scratch: boolean
  elem?: string
  shape?: number[]
}

export interface PassesView {
  /** The trace file stem under src/data/traces. */
  trace: string
  /** The change from the kernel as written the trace was compiled with, null for none. */
  variant: { add?: string; drop?: string } | null
  chip: string
  params: Record<string, number>
  peakBytes: number
  buffers: PassBuffer[]
  barriers: { afterOp: number; opKind: string; kept: boolean }[]
  barrierCounts: { kept: number; elided: number }
  /**
   * The trace's ops by id, which is the timeline: a loop's id comes after its body's, so a range
   * of ids reaching the loop's id is live to the loop's end.
   */
  ops: { id: number; kind: string; parent?: number }[]
  loops: { id: number; var: string | null; trips: number | null; firstOp: number; lastOp: number }[]
  recommended: boolean
}

export type MlirRole = "module" | "kernel" | "guard" | "copy" | "contract" | "loop" | "branch"

export interface MlirRegion {
  /** The op that owns the region, such as scf.for, scf.if, gpu.func; `else` for an scf.if's second region. */
  op: string
  /** 1-based line in the printed module. */
  line: number
  role: MlirRole
  /** The region's own ops by name, nested regions' excluded. */
  ops: Record<string, number>
  /** All ops in the region, nested ones included. */
  total: number
  iterArgs?: boolean
  resultType?: string
  children: MlirRegion[]
}

export interface MlirView {
  /** The gpu.func's region tree. */
  root: MlirRegion
  dialects: Record<string, number>
  ops: number
  constants: number
  loops: { line: number; depth: number; role: MlirRole; iterArgs: boolean }[]
  sharedBytes: number
  /** At most three real lines, verbatim. */
  labels: string[]
  note: string
}

export interface PtxFamily {
  name: string
  group: "memory" | "float" | "convert" | "integer" | "move" | "control" | "sync"
  count: number
}

export interface PtxView {
  target: string
  entry: string
  maxntid: string
  maxThreads: number
  params: number
  sharedBytes: number
  /** `.reg` declarations by type: virtual registers, not what ptxas allocates. */
  registers: Record<string, number>
  registersVirtual: true
  instructions: number
  blocks: number
  families: PtxFamily[]
  opcodes: Record<string, number>
  note: string
}

export interface GpuView {
  trace: string
  chip: string
  grid: [number, number, number]
  threads: number
  warps: number
  lanes: number
  ctas: number
  dims: Record<string, number>
  params: Record<string, number>
  sharedBytesPerCta: number
  note: string
}

export function countAstNodes(node: AstNode): number {
  return 1 + (node.children ?? []).reduce((sum, child) => sum + countAstNodes(child), 0)
}

/** The passes view to animate: the recommended one, else the first. */
export function recommendedPasses(structure: PipelineStructure): PassesView {
  return structure.passes.find(view => view.recommended) ?? structure.passes[0]
}

/** Buffers live at op `opId`. */
export function liveBuffers(view: PassesView, opId: number): PassBuffer[] {
  return view.buffers.filter(buffer => buffer.firstOp <= opId && opId <= buffer.lastOp)
}
