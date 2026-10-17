// The standard library as the docs present it, read from its Phobos sources: each file's functions,
// their parameters, the comment above each as its documentation, and the source text. Pure, so the
// tests and the pages parse the same text the compiler embeds. A function with no comment of its
// own that directly follows another (no blank line between) shares that one's documentation, as
// `exp2` and `log2` do.
//
// `evaluator` runs an expression function on numbers, which is how the activation and random-number
// figures draw what the shipped source computes and how the tests check the formulas against
// reference values.

export type Kind = "expression" | "statement"

export interface LibraryFunction {
  file: string
  name: string
  params: string[]
  kind: Kind
  // Paragraphs of the comment above the definition, or of the definition it shares one with.
  doc: string[]
  // The first function of the run that shares `doc`, or its own name.
  docOf: string
  source: string
  // 1-based, in the file.
  line: number
}

export interface LibraryFile {
  file: string
  // The comment that opens the file.
  intro: string[]
  functions: LibraryFunction[]
}

// The files `phobos-lang/src/stdlib.rs` embeds, in its order: ("std/math.ph", "stdlib/math.ph").
export function shippedFiles(stdlibRs: string): { file: string; path: string }[] {
  return [...stdlibRs.matchAll(/\("([^"]+\.ph)",\s*include_str!\("([^"]+)"\)\)/g)].map(match => ({
    file: match[1],
    path: match[2],
  }))
}

const FN_HEAD = /^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(=|\{)/

// A comment block's lines as paragraphs: an empty `//` line separates two.
function paragraphs(lines: string[]): string[] {
  const out: string[] = []
  let current: string[] = []
  for (const line of lines) {
    const text = line.replace(/^\/\/ ?/, "")
    if (text.trim() === "") {
      if (current.length > 0) out.push(current.join(" "))
      current = []
    } else current.push(text.trim())
  }
  if (current.length > 0) out.push(current.join(" "))
  return out
}

export function parseLibrary(file: string, text: string): LibraryFile {
  const lines = text.replace(/\r\n?/g, "\n").split("\n")
  const functions: LibraryFunction[] = []
  let comment: string[] = []
  let intro: string[] | null = null
  // The function that ended on the line before, for sharing its documentation.
  let previous: LibraryFunction | null = null
  let index = 0
  while (index < lines.length) {
    const line = lines[index]
    const trimmed = line.trim()
    if (trimmed.startsWith("//")) {
      comment.push(trimmed)
      previous = null
      index += 1
      continue
    }
    if (trimmed === "") {
      if (intro === null && comment.length > 0 && functions.length === 0) intro = paragraphs(comment)
      comment = []
      previous = null
      index += 1
      continue
    }
    const head = FN_HEAD.exec(line)
    if (!head) throw new Error(`${file}:${index + 1}: expected a function, found \`${trimmed}\``)
    const start = index
    if (head[3] === "{") {
      let depth = 0
      do {
        for (const ch of lines[index]) depth += ch === "{" ? 1 : ch === "}" ? -1 : 0
        index += 1
      } while (depth > 0 && index < lines.length)
      if (depth !== 0) throw new Error(`${file}:${start + 1}: unbalanced braces`)
    } else index += 1
    const name = head[1]
    const own = paragraphs(comment)
    const shared = own.length === 0 && previous !== null
    const fn: LibraryFunction = {
      file,
      name,
      params: head[2]
        .split(",")
        .map(param => param.trim())
        .filter(Boolean),
      kind: head[3] === "=" ? "expression" : "statement",
      doc: shared ? previous!.doc : own,
      docOf: shared ? previous!.docOf : name,
      source: lines.slice(start, index).join("\n"),
      line: start + 1,
    }
    functions.push(fn)
    comment = []
    previous = fn
  }
  return { file, intro: intro ?? [], functions }
}

// Whether a name is the library's own helper rather than part of what it offers.
export function isHelper(name: string): boolean {
  return name.startsWith("_")
}

// Expression evaluation.

// A value as the device holds it: a float, in double precision here, or an `i32`, which wraps as
// the device's does.
interface Value {
  v: number
  int: boolean
}

// An `i32` argument, as `uniform(seed, counter)` takes: `f("uniform", int(7), int(0))`.
export interface IntArg {
  int: number
}

export const int = (value: number): IntArg => ({ int: value })

type BinOp = "+" | "-" | "*" | "/" | "%" | "&" | "|" | "^" | "<<" | ">>" | "<" | "<=" | ">" | ">=" | "==" | "!=" | "&&" | "||"

type Node =
  | { kind: "num"; value: Value }
  | { kind: "var"; name: string }
  | { kind: "unary"; op: "-" | "~" | "!"; arg: Node }
  | { kind: "bin"; op: BinOp; lhs: Node; rhs: Node }
  | { kind: "call"; callee: string; args: Node[] }

// Loosest first, as SPEC.md's precedence row lists them.
const LEVELS: BinOp[][] = [
  ["||"],
  ["&&"],
  ["==", "!="],
  ["<", "<=", ">", ">="],
  ["|"],
  ["^"],
  ["&"],
  ["<<", ">>"],
  ["+", "-"],
  ["*", "/", "%"],
]

function tokenize(text: string): string[] {
  return text.match(/\d+\.?\d*(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z0-9_]*|<<|>>|<=|>=|==|!=|&&|\|\||[()+\-*/%&|^~!<>,]|\S/g) ?? []
}

function parseExpression(text: string): Node {
  const tokens = tokenize(text)
  let at = 0
  const peek = () => tokens[at]
  const take = (want?: string) => {
    const token = tokens[at]
    if (want !== undefined && token !== want) throw new Error(`expected \`${want}\`, found \`${token}\` in ${text}`)
    at += 1
    return token
  }
  const primary = (): Node => {
    const token = take()
    if (token === undefined) throw new Error(`unexpected end of ${text}`)
    if (token === "(") {
      const inner = level(0)
      take(")")
      return inner
    }
    if (token === "-" || token === "~" || token === "!") return { kind: "unary", op: token, arg: primary() }
    if (/^\d/.test(token)) return { kind: "num", value: { v: Number(token), int: !/[.eE]/.test(token) } }
    if (/^[A-Za-z_]/.test(token)) {
      if (peek() !== "(") return { kind: "var", name: token }
      take("(")
      const args: Node[] = []
      if (peek() !== ")") {
        args.push(level(0))
        while (peek() === ",") {
          take(",")
          args.push(level(0))
        }
      }
      take(")")
      return { kind: "call", callee: token, args }
    }
    throw new Error(`unexpected \`${token}\` in ${text}`)
  }
  function level(depth: number): Node {
    if (depth === LEVELS.length) return primary()
    let lhs = level(depth + 1)
    while ((LEVELS[depth] as string[]).includes(peek())) {
      const op = take() as BinOp
      lhs = { kind: "bin", op, lhs, rhs: level(depth + 1) }
    }
    return lhs
  }
  const node = level(0)
  if (at !== tokens.length) throw new Error(`trailing \`${tokens[at]}\` in ${text}`)
  return node
}

// The built-ins on floats an expression function may call, exactly where the hardware is: `exp`,
// `log`, `tanh` and `sqrt` are approximate on the device.
const BUILTINS: Record<string, (...args: number[]) => number> = {
  exp: Math.exp,
  log: Math.log,
  tanh: Math.tanh,
  sqrt: Math.sqrt,
  rsqrt: x => 1 / Math.sqrt(x),
  abs: Math.abs,
  sin: Math.sin,
  cos: Math.cos,
  floor: Math.floor,
  ceil: Math.ceil,
}

const float = (v: number): Value => ({ v, int: false })
const i32 = (v: number): Value => ({ v: v | 0, int: true })
const bool = (b: boolean): Value => ({ v: b ? 1 : 0, int: true })

function binary(op: BinOp, a: Value, b: Value): Value {
  const ints = a.int && b.int
  switch (op) {
    case "+":
      return ints ? i32(a.v + b.v) : float(a.v + b.v)
    case "-":
      return ints ? i32(a.v - b.v) : float(a.v - b.v)
    case "*":
      return ints ? i32(Math.imul(a.v, b.v)) : float(a.v * b.v)
    case "/":
      return ints ? i32(Math.trunc(a.v / b.v)) : float(a.v / b.v)
    case "%":
      return ints ? i32(a.v % b.v) : float(a.v % b.v)
    case "<":
      return bool(a.v < b.v)
    case "<=":
      return bool(a.v <= b.v)
    case ">":
      return bool(a.v > b.v)
    case ">=":
      return bool(a.v >= b.v)
    case "==":
      return bool(a.v === b.v)
    case "!=":
      return bool(a.v !== b.v)
    case "&&":
      return bool(a.v !== 0 && b.v !== 0)
    case "||":
      return bool(a.v !== 0 || b.v !== 0)
  }
  if (!ints) throw new Error(`\`${op}\` takes integers`)
  switch (op) {
    case "&":
      return i32(a.v & b.v)
    case "|":
      return i32(a.v | b.v)
    case "^":
      return i32(a.v ^ b.v)
    case "<<":
      return i32(a.v << b.v)
    case ">>":
      return i32(a.v >> b.v)
  }
}

// An evaluator of the expression functions of `files`: floats in double precision, `i32` values
// wrapping as on the device. A function that reduces a tile, as `rowmean` does, has no value on a
// number and fails when called.
export function evaluator(files: LibraryFile[]): (name: string, ...args: (number | IntArg)[]) => number {
  const bodies = new Map<string, { params: string[]; body: Node }>()
  for (const fn of files.flatMap(file => file.functions)) {
    if (fn.kind !== "expression") continue
    const text = fn.source.slice(fn.source.indexOf("=") + 1)
    bodies.set(fn.name, { params: fn.params, body: parseExpression(text) })
  }
  const run = (node: Node, scope: Map<string, Value>): Value => {
    switch (node.kind) {
      case "num":
        return node.value
      case "var": {
        const value = scope.get(node.name)
        if (value === undefined) throw new Error(`unbound \`${node.name}\``)
        return value
      }
      case "unary": {
        const x = run(node.arg, scope)
        if (node.op === "!") return bool(x.v === 0)
        if (node.op === "~") return binary("^", x, i32(-1))
        return x.int ? i32(-x.v) : float(-x.v)
      }
      case "bin":
        return binary(node.op, run(node.lhs, scope), run(node.rhs, scope))
      case "call":
        return call(
          node.callee,
          node.args.map(arg => run(arg, scope))
        )
    }
  }
  const call = (name: string, args: Value[]): Value => {
    if (name === "where") return args[0].v !== 0 ? args[1] : args[2]
    if (name === "f32" || name === "f64") return float(args[0].v)
    if (name === "i32") return i32(Math.trunc(args[0].v))
    if (name === "min" || name === "max") {
      const [a, b] = args
      const pick = Number.isNaN(a.v) ? b : Number.isNaN(b.v) ? a : (name === "min" ? a.v <= b.v : a.v >= b.v) ? a : b
      return a.int && b.int ? pick : float(pick.v)
    }
    const builtin = BUILTINS[name]
    if (builtin) return float(builtin(...args.map(arg => arg.v)))
    const fn = bodies.get(name)
    if (!fn) throw new Error(`\`${name}\` is not an expression function or a built-in the evaluator knows`)
    if (fn.params.length !== args.length) throw new Error(`\`${name}\` takes ${fn.params.length} arguments`)
    return run(fn.body, new Map(fn.params.map((param, i) => [param, args[i]])))
  }
  return (name, ...args) =>
    call(
      name,
      args.map(arg => (typeof arg === "number" ? float(arg) : i32(arg.int)))
    ).v
}
