// Writes src/data/pipeline/gemm_fp32.json, the data of the front page's compiler scene, from real
// compiler output for examples/gemm_fp32.ph. Produce the inputs with the toolchain first (no GPU):
//
//   cargo run -p phobos-lang --example ir examples/gemm_fp32.ph > ir.txt
//   cargo run -p phobos-lang --example emit examples/gemm_fp32.ph > mlir.txt
//   cargo run -p phobos-compile -- examples/gemm_fp32.ph --chip sm_75 -o gemm.ptx
//   a `{:#?}` dump of phobos_lang::parse for the AST > ast.txt
//
// The tree has no example for the AST dump; a scratch crate depending on phobos-lang by path (with
// the workspace's [patch.crates-io] and .cargo/config.toml) does it with this main:
//
//   for k in phobos_lang::parse(&std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap()).unwrap() {
//       println!("{k:#?}");
//   }
//
// then: node scripts/pipeline-excerpts.mjs DIR   (DIR holds ast.txt, ir.txt, mlir.txt, gemm.ptx)
//
// The file carries two things. `stages` is the text excerpt per station, each a contiguous slice of
// that output cut at anchors. `structure` is the same output parsed into data to draw: the source's
// lines, a pruned AST tree, the graph IR as a laid-out DAG, the shared-memory buffers and barriers of
// the visualizer's traces, the MLIR region tree, the PTX instruction mix and the launch. Its shape is
// the `PipelineData` type in src/components/home/pipeline-data.ts.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs"
import { join } from "node:path"
import { format, resolveConfig } from "prettier"

const dir = process.argv[2]
if (!dir) throw new Error("usage: node scripts/pipeline-excerpts.mjs DIR")

const readText = name => readFileSync(join(dir, name), "utf8").replace(/\r\n?/g, "\n")
const read = name => readText(name).split("\n")
const repo = new URL("../../", import.meta.url)
const readRepo = path => readFileSync(new URL(path, repo), "utf8").replace(/\r\n?/g, "\n")

function slice(lines, startPattern, count) {
  const start = lines.findIndex(line => startPattern.test(line))
  if (start < 0) throw new Error(`anchor ${startPattern} not found`)
  return dedent(lines.slice(start, start + count))
}

function dedent(lines) {
  const indents = lines.filter(line => line.trim()).map(line => line.match(/^\s*/)[0].length)
  const strip = Math.min(...indents)
  return lines.map(line => line.slice(strip).replace(/\s+$/, "")).join("\n")
}

// Rust's {:#?} spreads a one-field tuple over three lines; join them back into Int(0)
function compactDebug(text) {
  let previous
  do {
    previous = text
    text = text.replace(/\(\n\s*([^(){}[\]\n]+),\n\s*\)/g, "($1)")
  } while (text !== previous)
  return text
}

const SOURCE_PATH = "examples/gemm_fp32.ph"
// The first is the one the scene shows: without @aligned the kernel stages its tiles in shared
// memory, so the strip has buffers to pack and barriers to elide. The second is the kernel as written.
const TRACES = ["gemm_fp32_no_aligned.sm_75", "gemm_fp32.sm_75"]
const source = readRepo(SOURCE_PATH).split("\n")
const traces = TRACES.map(file => ({ file, ...JSON.parse(readRepo(`website/src/data/traces/${file}.json`)) }))
const trace = traces.find(t => t.file === "gemm_fp32.sm_75")

const ptx = read("gemm.ptx")
const ptxLines = [
  ...ptx.filter(line => /^\.target|^\.visible \.entry|^\.maxntid/.test(line)),
  "...",
  ptx.find(line => /ld\.global\.v4/.test(line)),
  ptx.find(line => /bar\.sync/.test(line)),
  ptx.find(line => /ld\.shared\.v4/.test(line)),
  ptx.find(line => /fma\.rn\.f32/.test(line)),
].map(line => line.replace(/\t+/g, " ").trim())

const buffers = trace.buffers.map(
  buffer =>
    `${buffer.name.padEnd(6)} ${String(buffer.bytes).padStart(4)} B at ${String(buffer.offsetBytes).padStart(4)}` +
    `${buffer.scratch ? "  dot staging" : ""}`
)
const barriers = trace.barriers ?? []
const kept = barriers.filter(barrier => barrier.kept).length
const passes = [
  `// TILE_M=${trace.params.TILE_M} TILE_N=${trace.params.TILE_N} TILE_K=${trace.params.TILE_K}, ${trace.compiler.chip}`,
  ...buffers,
  `peak ${trace.sharedPeakBytes} B of shared memory`,
  `barriers: ${kept} kept, ${barriers.length - kept} elided`,
].join("\n")

const stages = {
  source: slice(source, /^\s*for kt in/, 5),
  ast:
    compactDebug(slice(read("ast.txt"), /^\s*For \{/, 60))
      .split("\n")
      .slice(0, 14)
      .join("\n") + "\n...",
  graph: slice(read("ir.txt"), /= for %/, 10),
  passes,
  mlir: slice(read("mlir.txt"), /= scf\.for .*iter_args/, 9) + "\n...",
  ptx: ptxLines.join("\n"),
}

// AST: a parser for Rust's {:#?} output, then a display tree pruned around the `for` loop.

function parseDebug(text) {
  const tokens = text.match(/"(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?(?:e-?\d+)?|[A-Za-z_#][\w#.]*|[{}()[\],:]/g)
  let at = 0
  const peek = () => tokens[at]
  const take = expected => {
    const token = tokens[at++]
    if (expected && token !== expected) throw new Error(`ast dump: expected ${expected}, got ${token}`)
    return token
  }
  function list(close) {
    const items = []
    while (peek() !== close) {
      items.push(value())
      if (peek() === ",") take(",")
    }
    take(close)
    return items
  }
  function value() {
    const token = take()
    if (token === "[") return { t: "list", items: list("]") }
    if (token.startsWith('"')) return { t: "str", v: JSON.parse(token) }
    if (/^-?\d/.test(token)) return { t: "num", v: Number(token) }
    if (peek() === "(") {
      take("(")
      return { t: "tuple", name: token, items: list(")") }
    }
    if (peek() === "{") {
      take("{")
      const fields = {}
      while (peek() !== "}") {
        const key = take()
        take(":")
        fields[key] = value()
        if (peek() === ",") take(",")
      }
      take("}")
      return { t: "struct", name: token, fields }
    }
    return { t: "ident", name: token }
  }
  const values = []
  while (at < tokens.length) values.push(value())
  return values
}

const BINARY = {
  Add: "+",
  Sub: "-",
  Mul: "*",
  Div: "/",
  Rem: "%",
  Lt: "<",
  Le: "<=",
  Gt: ">",
  Ge: ">=",
  Eq: "==",
  Ne: "!=",
}
const ASSIGN = { Set: "=", Add: "+=", Sub: "-=", Mul: "*=" }
const LEAVES = ["Var", "Sym", "Ident"]

function some(v) {
  if (!v || (v.t === "ident" && v.name === "None")) return null
  return v.t === "tuple" && v.name === "Some" ? v.items[0] : v
}

function typeText(v) {
  if (v.t === "ident") return v.name.toLowerCase()
  if (v.t === "tuple" && v.name === "Scalar") return typeText(v.items[0])
  if (v.t === "tuple" && (v.name === "Tensor" || v.name === "Tile")) {
    const dims = v.items[1].items.map(exprText).join(", ")
    return `${v.name.toLowerCase()}<${typeText(v.items[0])}>[${dims}]`
  }
  return v.name ?? "?"
}

// The source spelling of an expression, for a folded node's text.
function exprText(v) {
  if (!v) return ""
  if (v.t === "num") return String(v.v)
  if (v.t === "str") return v.v
  if (v.t === "ident") return v.name
  if (v.t === "tuple") {
    if (LEAVES.includes(v.name)) return v.items[0].v
    if (v.name === "Int") return String(v.items[0].v)
    if (v.name === "Float") return Number.isInteger(v.items[0].v) ? v.items[0].v.toFixed(1) : String(v.items[0].v)
    return v.items.map(exprText).join(", ")
  }
  const f = v.fields
  if (v.name === "Binary") return `${exprText(f.lhs)} ${BINARY[f.op.name] ?? f.op.name} ${exprText(f.rhs)}`
  if (v.name === "Call") return `${f.callee.v.replace(/^#/, "")}(${f.args.items.map(exprText).join(", ")})`
  if (v.name === "Index") return `${exprText(f.base)}[${f.subs.items.map(exprText).join(", ")}]`
  if (v.name === "Span") return `${exprText(f.start)} :+ ${exprText(f.len)}`
  return v.name ?? "?"
}

function stmtText(v) {
  const f = v.fields
  if (v.name === "Let" || v.name === "Var") return `${v.name.toLowerCase()} ${f.name.v} = ${exprText(some(f.value))}`
  if (v.name === "Assign") return `${exprText(f.target)} ${ASSIGN[f.op.name] ?? f.op.name} ${exprText(f.value)}`
  if (v.name === "For")
    return `for ${f.var.v} in range(${exprText(f.start)}, ${exprText(f.end)}, ${exprText(some(f.step))})`
  return v.name
}

function countNodes(node) {
  return 1 + (node.children ?? []).reduce((sum, child) => sum + countNodes(child), 0)
}

function foldNode(node, text) {
  const hidden = countNodes(node) - 1
  const folded = { kind: node.kind, label: node.label, text }
  if (node.role) folded.role = node.role
  if (node.line) folded.line = node.line
  if (hidden) folded.folded = hidden
  return folded
}

function astNode(v, role) {
  const node = toAstNode(v)
  return role ? { ...node, role } : node
}

function toAstNode(v) {
  if (v.t === "num") return { kind: "Int", label: String(v.v) }
  if (v.t === "tuple") {
    if (LEAVES.includes(v.name)) return { kind: v.name, label: v.items[0].v }
    if (v.name === "Int" || v.name === "Float") return { kind: v.name, label: exprText(v) }
    return { kind: v.name, label: exprText(v), children: v.items.filter(i => i.t !== "str").map(i => astNode(i)) }
  }
  if (v.t !== "struct") return { kind: v.name ?? v.t, label: exprText(v) }
  const f = v.fields
  const line = f.span?.fields?.line?.v
  const withLine = node => (line ? { ...node, line } : node)
  switch (v.name) {
    case "Binary":
      return { kind: "Binary", label: BINARY[f.op.name] ?? f.op.name, children: [astNode(f.lhs), astNode(f.rhs)] }
    case "Call":
      return { kind: "Call", label: f.callee.v.replace(/^#/, ""), children: f.args.items.map(a => astNode(a)) }
    case "Index":
      return { kind: "Index", label: "[ ]", children: [astNode(f.base), ...f.subs.items.map(s => astNode(s))] }
    case "Span":
      return { kind: "Span", label: ":+", children: [astNode(f.start, "start"), astNode(f.len, "len")] }
    case "Let":
    case "Var": {
      const value = some(f.value)
      return withLine({ kind: v.name, label: f.name.v, children: value ? [astNode(value, "value")] : [] })
    }
    case "Assign":
      return withLine({
        kind: "Assign",
        label: ASSIGN[f.op.name] ?? f.op.name,
        children: [astNode(f.target, "target"), astNode(f.value, "value")],
      })
    case "For": {
      const step = some(f.step)
      return withLine({
        kind: "For",
        label: f.var.v,
        children: [
          astNode(f.start, "start"),
          astNode(f.end, "end"),
          ...(step ? [astNode(step, "step")] : []),
          ...f.body.items.map(s => astNode(s, "body")),
        ],
      })
    }
    default:
      return { kind: v.name, label: exprText(v) }
  }
}

function buildAst(kernel) {
  const f = kernel.fields
  const attrs = f.attrs.items.map(a => `@${a.fields.name.v}`)
  const params = f.params.items.map(p => `${p.fields.name.v}: ${typeText(p.fields.ty)}`)
  const children = [
    { kind: "Attributes", label: attrs.join(" "), text: attrs.join(" "), folded: attrs.length },
    {
      kind: "Params",
      label: f.params.items.map(p => p.fields.name.v).join(", "),
      text: params.join(", "),
      folded: params.length,
    },
    ...f.body.items.map(stmt => {
      const node = astNode(stmt, "body")
      return stmt.name === "For" ? node : foldNode(node, stmtText(stmt))
    }),
  ]
  const root = { kind: "Kernel", label: f.name.v, children }
  return {
    root,
    nodeCount: countNodes(root),
    focus: "For",
    pruned:
      "Source spans, absent options (ty: None, unroll: None) and the attribute and parameter lists are dropped; " +
      "every statement outside the `for` is one node with its subtree folded (`folded` counts the hidden nodes, " +
      "`text` spells it out).",
  }
}

// Source: the kernel's lines, with the loop marked by the AST's spans.

function buildSource(ast) {
  const loop = ast.root.children.find(c => c.kind === "For")
  const header = loop.line
  const body = loop.children.filter(c => c.role === "body").map(c => c.line)
  const braces = line => (line.match(/\{/g) ?? []).length - (line.match(/\}/g) ?? []).length
  let open = 0
  let end = header
  for (let n = header; n <= source.length; n += 1) {
    open += braces(source[n - 1])
    if (open === 0) {
      end = n
      break
    }
  }
  const kernelLine = source.findIndex(l => /^\s*kernel /.test(l)) + 1
  const signatureEnd = source.findIndex((l, i) => i >= kernelLine - 1 && /\{\s*$/.test(l)) + 1
  const lines = source
    .map((text, i) => ({ n: i + 1, text: text.replace(/\s+$/, "") }))
    .filter(({ n, text }) => text || n < source.length)
    .map(({ n, text }) => {
      let role = "code"
      if (!text.trim()) role = "blank"
      else if (text.startsWith("@")) role = "attribute"
      else if (n >= kernelLine && n <= signatureEnd) role = "signature"
      else if (n === header) role = "loop-header"
      else if (body.includes(n)) role = "loop-body"
      else if (n === end) role = "loop-end"
      return { n, text, role }
    })
  return { path: SOURCE_PATH, lines, loop: { header, body, end } }
}

// Graph IR: the `ir` example's print parsed into a DAG, folded, then laid out in layers.

function splitTop(text) {
  const parts = []
  let level = 0
  let current = ""
  for (const ch of text) {
    if ("([{<".includes(ch)) level += 1
    if (")]}>".includes(ch)) level -= 1
    if (ch === "," && level === 0) {
      parts.push(current.trim())
      current = ""
    } else current += ch
  }
  if (current.trim()) parts.push(current.trim())
  return parts
}

const shortType = type => type.replace(/@div\[[^\]]*\]/, "").replace(/\{[^}]*\}/, "")
const valueName = id => id.match(/^%([A-Za-z_]\w*)\.\d+$/)?.[1] ?? id

function parseIr(lines) {
  const ops = []
  const regions = [null]
  let kernel = null
  const typed = text => {
    const [, id, type] = text.match(/^(%[\w.]+): (.*)$/)
    return { id, type }
  }
  for (const raw of lines) {
    const line = raw.trim()
    if (!line || line.startsWith("//")) continue
    let m
    if ((m = line.match(/^kernel (\w+)\((.*)\) \{$/))) {
      kernel = m[1]
      for (const param of splitTop(m[2])) ops.push({ ...typed(param), op: "param", operands: [], region: null })
      continue
    }
    if (line === "}") {
      regions.pop()
      continue
    }
    if ((m = line.match(/^\^\((.*)\)$/))) {
      for (const arg of splitTop(m[1]))
        ops.push({ ...typed(arg), op: "block_arg", operands: [], region: regions.at(-1) })
      continue
    }
    const opens = line.endsWith("{")
    const body = opens ? line.slice(0, -1).trim() : line
    let op
    if ((m = body.match(/^(%[\w.]+): (.+?) = (\w+)(.*)$/))) op = { id: m[1], type: m[2], op: m[3], rest: m[4].trim() }
    else {
      m = body.match(/^(\w+)(.*)$/)
      op = { id: `#${ops.length}`, type: null, op: m[1], rest: m[2].trim() }
    }
    // Operands are value names; a `[32, 4]` shape or an `alpha=s` flag is not one.
    op.operands = op.rest.match(/%[\w.]+/g) ?? []
    op.region = regions.at(-1)
    ops.push(op)
    if (opens) regions.push(op.id)
  }
  return { kernel, ops }
}

const FOLDED_OPS = new Set([
  "const",
  "mul",
  "add",
  "sub",
  "div",
  "rem",
  "ne",
  "eq",
  "lt",
  "le",
  "gt",
  "ge",
  "min",
  "max",
])
const EXPR_OPS = {
  mul: "*",
  add: "+",
  sub: "-",
  div: "/",
  rem: "%",
  ne: "!=",
  eq: "==",
  lt: "<",
  le: "<=",
  gt: ">",
  ge: ">=",
}

function buildGraph(ir) {
  const byId = new Map(ir.ops.map(op => [op.id, op]))
  const order = new Map(ir.ops.map((op, i) => [op.id, i]))
  const loop = ir.ops.find(op => op.op === "for")
  const isFolded = op => FOLDED_OPS.has(op.op)
  const isGuard = op =>
    op.op === "if" && ir.ops.filter(c => c.region === op.id).every(c => c.op === "trap" || c.op === "yield")
  const regionOf = op => (op.region === loop.id ? loop.id : null)

  const nodes = []
  const edges = []
  const folded = []
  const edge = (from, to, kind = "data") => edges.push({ from, to, kind })

  // An expression over folded ops; its leaves are the kept values it reads.
  function expr(id, leaves) {
    const op = byId.get(id)
    if (!op || !isFolded(op)) {
      leaves.add(id)
      return valueName(id)
    }
    folded.push(id)
    if (op.op === "const") return op.rest
    if (op.op in EXPR_OPS) return op.operands.map(o => expr(o, leaves)).join(` ${EXPR_OPS[op.op]} `)
    return `${op.op}(${op.operands.map(o => expr(o, leaves)).join(", ")})`
  }
  function foldedUnder(id) {
    const out = []
    const walk = v => {
      const op = byId.get(v)
      if (!op || !isFolded(op) || out.includes(v)) return
      out.push(v)
      op.operands.forEach(walk)
    }
    walk(id)
    return out
  }

  // Reads of a kept op: a constant becomes a literal on it, folded arithmetic one `index math` node.
  function connect(op, nodeId) {
    const literals = []
    for (const id of op.operands) {
      const src = byId.get(id)
      if (src?.op === "const") {
        literals.push(src.rest)
        folded.push(id)
      } else if (src && isFolded(src)) {
        const leaves = new Set()
        const math = {
          id: `math:${id}`,
          kind: "math",
          op: "index math",
          label: expr(id, leaves),
          type: src.type,
          region: regionOf(op),
          folds: foldedUnder(id),
        }
        nodes.push(math)
        for (const leaf of leaves) edge(leaf, math.id)
        edge(math.id, nodeId)
      } else edge(id, nodeId)
    }
    return literals
  }

  for (const op of ir.ops) {
    if (isFolded(op) || op.op === "trap" || op.op === "block_arg") continue
    if (op.op === "yield" && op.region !== loop.id) continue
    if (op.op === "if" && isGuard(op)) {
      const leaves = new Set()
      const label = `trap if ${expr(op.operands[0], leaves)}`
      const id = `guard:${op.id}`
      nodes.push({
        id,
        kind: "guard",
        op: "if trap",
        label,
        type: null,
        region: null,
        folds: foldedUnder(op.operands[0]),
      })
      for (const leaf of leaves) edge(leaf, id)
      continue
    }
    if (op.op === "for") {
      const [start, end, step, init] = op.operands
      const args = ir.ops.filter(o => o.op === "block_arg" && o.region === op.id)
      const name = id => (byId.get(id)?.op === "const" ? byId.get(id).rest : valueName(id))
      const header = {
        id: `for:${op.id}`,
        kind: "for",
        op: "for",
        label: `for ${valueName(args[0].id)} in ${name(start)}..${name(end)} step ${name(step)}`,
        type: null,
        region: null,
      }
      nodes.push(header)
      for (const id of [start, end, step, init]) {
        if (byId.get(id)?.op === "const") folded.push(id)
        else edge(id, header.id)
      }
      args.forEach((arg, i) => {
        const label = i === 0 ? valueName(arg.id) : `${valueName(arg.id)} carried`
        nodes.push({ id: arg.id, kind: "block_arg", op: "block arg", label, type: shortType(arg.type), region: op.id })
        edge(header.id, arg.id)
      })
      continue
    }
    const kind = op.op === "param" ? "param" : op.op === "yield" ? "yield" : "op"
    const node = {
      id: op.id,
      kind,
      op: op.op,
      label: opLabel(op),
      type: op.type && shortType(op.type),
      region: regionOf(op),
    }
    nodes.push(node)
    const literals = connect(op, node.id)
    if (literals.length) node.literals = literals
  }

  // The loop's value after its last trip: the yield's, one layer after the body.
  const loopYield = nodes.find(n => n.kind === "yield" && n.region === loop.id)
  const args = ir.ops.filter(o => o.op === "block_arg" && o.region === loop.id)
  nodes.push({
    id: loop.id,
    kind: "for_result",
    op: "for result",
    label: valueName(loop.id),
    type: shortType(loop.type),
    region: null,
  })
  edge(loopYield.id, loop.id)
  edge(loopYield.id, args.at(-1).id, "back")

  const lastBody = Math.max(...ir.ops.filter(o => o.region === loop.id).map(o => order.get(o.id)))
  const position = node =>
    node.kind === "for_result" ? lastBody + 0.5 : order.get(node.id.replace(/^(math|guard|for):/, ""))
  nodes.sort((a, b) => position(a) - position(b))

  const ids = new Set(nodes.map(n => n.id))
  const seen = new Set()
  const unique = edges.filter(e => {
    const key = `${e.from}>${e.to}>${e.kind}`
    if (!ids.has(e.from) || !ids.has(e.to)) throw new Error(`graph: dangling edge ${key}`)
    const fresh = !seen.has(key)
    seen.add(key)
    return fresh
  })
  const foldedIds = [...new Set(folded)].sort((a, b) => order.get(a) - order.get(b))
  return {
    kernel: ir.kernel,
    nodes,
    edges: unique,
    regions: [
      {
        id: loop.id,
        kind: "for",
        header: `for:${loop.id}`,
        result: loop.id,
        label: nodes.find(n => n.kind === "for").label,
      },
    ],
    pruned: {
      opsPrinted: ir.ops.length,
      foldedOps: foldedIds.length,
      foldedIds,
      note:
        "Constants are inlined into their readers (as `literals`, or into a label); index arithmetic (mul, rem, ne) " +
        "is folded into one `index math` node per reader, labelled with its expression; each `if { trap }` " +
        "divisibility check is one `guard` node. The `for` is drawn as a header node, its block arguments, the " +
        "body and a result node; the yield's edge back to the carried argument is `back` and not layered.",
    },
  }

  function opLabel(op) {
    switch (op.op) {
      case "param":
        return valueName(op.id)
      case "slice":
        return `slice ${valueName(op.operands[0])}${op.rest.match(/^\[[^\]]*\]/)?.[0] ?? ""}`
      case "dim":
        return `${valueName(op.id)} = dim ${op.rest.match(/^\d+/)?.[0]} ${valueName(op.operands[0])}`
      case "program_id":
        return `${valueName(op.id)} = program_id ${op.rest}`
      case "assume_align":
        return `assume_align ${valueName(op.operands[0])}`
      default:
        return op.op
    }
  }
}

const COLUMN_UNITS = 170
const ROW_UNITS = 56

// Longest-path layering pulled right to each node's earliest reader, then barycentric rows with
// the loop body kept in one band. Left to right: x is the layer.
function layoutGraph(graph) {
  const { nodes, edges } = graph
  const region = graph.regions[0]
  const preds = new Map(nodes.map(n => [n.id, []]))
  const succs = new Map(nodes.map(n => [n.id, []]))
  for (const e of edges.filter(e => e.kind !== "back")) {
    preds.get(e.to).push(e.from)
    succs.get(e.from).push(e.to)
  }
  const inRegion = n => n.region === region.id
  const layer = new Map(nodes.map(n => [n.id, 0]))
  for (let changed = true; changed;) {
    changed = false
    for (const n of nodes) {
      let l = Math.max(0, ...preds.get(n.id).map(p => layer.get(p) + 1))
      if (inRegion(n)) l = Math.max(l, layer.get(region.header) + 1)
      if (l !== layer.get(n.id)) {
        layer.set(n.id, l)
        changed = true
      }
    }
  }
  // Block arguments stay at the loop's first layer, where the header hands them in.
  for (const n of [...nodes].sort((a, b) => layer.get(b.id) - layer.get(a.id))) {
    if (n.kind === "block_arg") continue
    const next = succs.get(n.id).map(s => layer.get(s))
    if (next.length) layer.set(n.id, Math.max(layer.get(n.id), Math.min(...next) - 1))
  }
  const layerCount = Math.max(...layer.values()) + 1
  const regionLayers = nodes.filter(inRegion).map(n => layer.get(n.id))
  const regionFirst = Math.min(...regionLayers)
  const regionLast = Math.max(...regionLayers)

  const row = new Map(nodes.map((n, i) => [n.id, i]))
  const ends = new Set([region.header, region.result])
  let band = { top: 0, height: 0 }
  // The loop's header and result line up with its band; everything else sits by its operands, or by
  // its readers when it has none.
  const bary = n => {
    if (ends.has(n.id) && band.height) return band.top + (band.height - 1) / 2
    const around = preds.get(n.id).length ? preds.get(n.id) : succs.get(n.id)
    return around.length ? around.reduce((s, id) => s + row.get(id), 0) / around.length : row.get(n.id)
  }
  const layers = Array.from({ length: layerCount }, (_, l) => nodes.filter(n => layer.get(n.id) === l))
  for (let sweep = 0; sweep < 16; sweep += 1) {
    const center = band.top + (band.height - 1) / 2
    const groups = layers.map((members, l) => {
      const ranked = members.map(n => ({ n, b: bary(n) })).sort((a, b) => a.b - b.b)
      if (l < regionFirst || l > regionLast) return { banded: false, ranked }
      return {
        banded: true,
        above: ranked.filter(r => !inRegion(r.n) && r.b <= center).map(r => r.n),
        inside: ranked.filter(r => inRegion(r.n)).map(r => r.n),
        below: ranked.filter(r => !inRegion(r.n) && r.b > center).map(r => r.n),
      }
    })
    const banded = groups.filter(g => g.banded)
    band = {
      top: Math.max(0, ...banded.map(g => g.above.length)),
      height: Math.max(...banded.map(g => g.inside.length)),
    }
    for (const g of groups) {
      if (g.banded) {
        g.above.forEach((n, i) => row.set(n.id, band.top - g.above.length + i))
        const offset = Math.floor((band.height - g.inside.length) / 2)
        g.inside.forEach((n, i) => row.set(n.id, band.top + offset + i))
        g.below.forEach((n, i) => row.set(n.id, band.top + band.height + i))
      } else {
        // Each node as near its barycenter as the order allows.
        let previous = -Infinity
        for (const { n, b } of g.ranked) {
          previous = Math.max(previous + 1, Math.round(b))
          row.set(n.id, previous)
        }
      }
    }
  }
  // Rows no node uses are dropped; the order of the rest is kept, so the band stays clear.
  const used = [...new Set(row.values())].sort((a, b) => a - b)
  for (const n of nodes) {
    n.layer = layer.get(n.id)
    n.row = used.indexOf(row.get(n.id))
    n.x = n.layer * COLUMN_UNITS
    n.y = n.row * ROW_UNITS
  }
  const rows = used.length
  const bandRows = nodes.filter(inRegion).map(n => n.row)
  const [firstRow, lastRow] = [Math.min(...bandRows), Math.max(...bandRows)]
  region.box = {
    x: (regionFirst - 0.5) * COLUMN_UNITS,
    y: (firstRow - 0.5) * ROW_UNITS,
    width: (regionLast - regionFirst + 1) * COLUMN_UNITS,
    height: (lastRow - firstRow + 1) * ROW_UNITS,
    firstLayer: regionFirst,
    lastLayer: regionLast,
    firstRow,
    lastRow,
  }
  graph.layout = {
    method:
      "longest-path layering, each node pulled right to just before its earliest reader; barycentric rows, " +
      "the loop body in one band of rows that other nodes in its layers sit above or below",
    direction: "left-to-right",
    columnWidth: COLUMN_UNITS,
    rowHeight: ROW_UNITS,
    layers: layerCount,
    rows,
    width: (layerCount - 1) * COLUMN_UNITS,
    height: (rows - 1) * ROW_UNITS,
  }
  return graph
}

// Passes: the visualizer's traces, as buffers on the shared-memory strip and barriers on the ops.

function buildPasses(t) {
  const ops = t.ops
    .map(op => ({ id: op.id, kind: op.kind, ...(op.parent !== undefined ? { parent: op.parent } : {}) }))
    .sort((a, b) => a.id - b.id)
  const kindOf = id => t.ops.find(op => op.id === id)?.kind
  const loops = t.ops
    .filter(op => op.kind === "for")
    .map(op => {
      const body = t.ops.filter(o => o.parent === op.id).map(o => o.id)
      return {
        id: op.id,
        var: op.loop?.var ?? null,
        trips: op.loop?.trips ?? null,
        firstOp: Math.min(...body),
        lastOp: op.id,
      }
    })
  const barriers = (t.barriers ?? []).map(b => ({ afterOp: b.afterOp, opKind: kindOf(b.afterOp), kept: b.kept }))
  return {
    trace: t.file,
    variant: t.source.variant ?? null,
    chip: t.compiler.chip,
    params: t.params,
    peakBytes: t.sharedPeakBytes,
    buffers: t.buffers.map(b => ({
      id: b.id,
      name: b.name,
      bytes: b.bytes,
      offsetBytes: b.offsetBytes,
      firstOp: b.firstOp,
      lastOp: b.lastOp,
      scratch: b.scratch,
      ...(b.elem ? { elem: b.elem } : {}),
      ...(b.shape ? { shape: b.shape } : {}),
    })),
    barriers,
    barrierCounts: { kept: barriers.filter(b => b.kept).length, elided: barriers.filter(b => !b.kept).length },
    ops,
    loops,
  }
}

// MLIR: the region tree with op counts by dialect, not the text.

function buildMlir(lines) {
  const root = { op: "module", line: 1, role: "module", ops: {}, total: 0, children: [] }
  const stack = [root]
  const dialects = {}
  let constants = 0
  let opCount = 0
  lines.forEach((raw, i) => {
    const line = raw.trim()
    if (!line || line.startsWith("#") || line === "module {") return
    if (line === "}") {
      stack.pop()
      return
    }
    if (line === "} else {") {
      stack.pop()
      const branch = { op: "else", line: i + 1, role: "branch", ops: {}, total: 0, children: [] }
      stack.at(-1).children.push(branch)
      stack.push(branch)
      return
    }
    const name = line.replace(/^%[\w#:]+(, %[\w#]+)* = /, "").match(/^([a-z_]+\.[\w.]+)/)?.[1]
    if (!name) return
    const dialect = name.split(".")[0]
    dialects[dialect] = (dialects[dialect] ?? 0) + 1
    opCount += 1
    if (name === "arith.constant") constants += 1
    const parent = stack.at(-1)
    for (const frame of stack) frame.total += 1
    if (line.endsWith("{")) {
      const node = { op: name, line: i + 1, role: "", ops: {}, total: 0, children: [] }
      if (/iter_args/.test(line)) node.iterArgs = true
      const result = line.match(/-> \(?([^){]+)\)? \{$/)
      if (result && name.startsWith("scf.")) node.resultType = result[1].trim()
      parent.children.push(node)
      stack.push(node)
    } else parent.ops[name] = (parent.ops[name] ?? 0) + 1
  })
  const has = (node, op) => node.ops[op] || node.children.some(c => has(c, op))
  const tag = node => {
    node.children.forEach(tag)
    if (node.role) return
    if (node.op === "gpu.module") node.role = "module"
    else if (node.op === "gpu.func") node.role = "kernel"
    else if (node.ops["llvm.intr.trap"]) node.role = "guard"
    else if (node.ops["vector.contract"]) node.role = "contract"
    else if (
      node.op === "scf.for" &&
      !has(node, "vector.contract") &&
      (has(node, "memref.store") || has(node, "vector.store"))
    )
      node.role = "copy"
    else if (node.op === "scf.for") node.role = "loop"
    else if (node.children.length && node.children.every(c => c.role === "copy")) node.role = "copy"
    else node.role = "branch"
  }
  tag(root)
  const fn = root.children.find(c => c.op === "gpu.module").children.find(c => c.op === "gpu.func")
  const loops = []
  const walk = (node, depth) => {
    if (node.op === "scf.for") loops.push({ line: node.line, depth, role: node.role, iterArgs: !!node.iterArgs })
    node.children.forEach(c => walk(c, depth + (node.op === "scf.for" ? 1 : 0)))
  }
  walk(fn, 0)
  const LABELS = [/= gpu\.block_id\s+x$/, /= vector\.broadcast %cst : f32 to vector/, /^gpu\.barrier$/]
  const trimmed = lines.map(l => l.trim())
  const labels = LABELS.map(pattern => trimmed.find(l => pattern.test(l))).filter(Boolean)
  const shared = trimmed.find(l => /^memref\.global .*_shared : memref<\d+xi8, 3>/.test(l))
  return {
    root: fn,
    dialects,
    ops: opCount,
    constants,
    loops,
    sharedBytes: Number(shared?.match(/memref<(\d+)xi8, 3>/)[1] ?? 0),
    labels,
    note:
      "Static op counts in the printed module. A region's `ops` counts its own ops by name, `total` includes " +
      "nested regions; `role` is read off the region's contents. `labels` are real lines, verbatim.",
  }
}

// PTX: the static instruction mix by family, the declared registers and the launch bound.

const PTX_FAMILIES = [
  ["ld.global", "memory", /^ld\.global/],
  ["st.global", "memory", /^st\.global/],
  ["ld.shared", "memory", /^ld\.shared/],
  ["st.shared", "memory", /^st\.shared/],
  ["ld.param", "memory", /^ld\.param/],
  ["fma", "float", /^fma\./],
  ["mul.f", "float", /^mul\.rn?\.f/],
  ["add.f", "float", /^add\.rn?\.f/],
  ["cvt", "convert", /^cvt\./],
  ["int", "integer", /^(add|sub|mul|mad|shl|shr|and|or|xor|min|max|div|rem|neg|not)\./],
  ["mov", "move", /^mov\./],
  ["setp", "control", /^setp\./],
  ["bra", "control", /^bra/],
  ["bar.sync", "sync", /^bar\./],
  ["trap", "control", /^trap$/],
  ["ret", "control", /^ret$/],
]

function buildPtx(lines) {
  const opcodes = {}
  let labels = 0
  for (const line of lines) {
    if (/^\$L\w+:/.test(line)) labels += 1
    if (!/^\t+[a-z@]/.test(line)) continue
    const opcode = line
      .trim()
      .replace(/^@!?%p\d+\s+/, "")
      .split(/[\s;]/)[0]
    opcodes[opcode] = (opcodes[opcode] ?? 0) + 1
  }
  const familyOf = op => PTX_FAMILIES.find(([, , pattern]) => pattern.test(op))
  const unclassified = Object.keys(opcodes).filter(op => !familyOf(op))
  if (unclassified.length) throw new Error(`ptx: opcodes in no family: ${unclassified.join(", ")}`)
  const families = PTX_FAMILIES.map(([name, group]) => ({
    name,
    group,
    count: Object.entries(opcodes)
      .filter(([op]) => familyOf(op)[0] === name)
      .reduce((sum, [, n]) => sum + n, 0),
  }))
  const registers = {}
  for (const m of lines.join("\n").matchAll(/\.reg \.(\w+)\s+%\w+<(\d+)>/g)) registers[m[1]] = Number(m[2])
  const maxntid = lines.find(l => /^\.maxntid/.test(l)).trim()
  return {
    target: lines.find(l => /^\.target/.test(l)).trim(),
    entry: lines.find(l => /^\.visible \.entry/.test(l)).match(/entry (\w+)/)[1],
    maxntid,
    maxThreads: Number(maxntid.match(/\d+/)[0]),
    params: lines.filter(l => /^\s*\.param /.test(l)).length,
    sharedBytes: Number(lines.find(l => /^\s*\.shared .*\[\d+\]/.test(l))?.match(/\[(\d+)\]/)[1] ?? 0),
    registers,
    registersVirtual: true,
    instructions: Object.values(opcodes).reduce((sum, n) => sum + n, 0),
    blocks: labels + 1,
    families,
    opcodes,
    note:
      "Static counts over the PTX text, not executed instructions. `registers` are the `.reg` declarations, " +
      "virtual registers before ptxas allocates; the physical count needs ptxas.",
  }
}

const kernel = parseDebug(readText("ast.txt")).find(v => v.name === "Kernel")
const ast = buildAst(kernel)
const graph = layoutGraph(buildGraph(parseIr(read("ir.txt"))))
const ptxData = buildPtx(ptx)
const structure = {
  source: buildSource(ast),
  ast,
  graph,
  passes: traces.map((t, i) => ({ ...buildPasses(t), recommended: i === 0 })),
  mlir: buildMlir(read("mlir.txt")),
  ptx: ptxData,
  gpu: {
    trace: trace.file,
    chip: trace.compiler.chip,
    grid: trace.launch.grid,
    threads: trace.launch.threads,
    warps: trace.launch.warps,
    lanes: trace.launch.lanes,
    ctas: trace.launch.grid.reduce((a, b) => a * b, 1),
    dims: trace.dims,
    params: trace.params,
    sharedBytesPerCta: trace.sharedPeakBytes,
    note: "The grid is the host's choice for the trace's dims (website/kernels.json), not part of the kernel.",
  },
}

const out = new URL("website/src/data/pipeline/", repo)
mkdirSync(out, { recursive: true })
const file = new URL("gemm_fp32.json", out)
const options = { ...(await resolveConfig(file)), filepath: file.pathname }
writeFileSync(file, await format(JSON.stringify({ stages, structure }), options))
console.log(
  `stages: ${Object.keys(stages).join(", ")}\n` +
    `ast ${ast.nodeCount} nodes; graph ${graph.nodes.length} nodes, ${graph.edges.length} edges, ` +
    `${graph.layout.layers} layers x ${graph.layout.rows} rows, ${graph.pruned.foldedOps} ops folded; ` +
    `ptx ${ptxData.instructions} instructions`
)
