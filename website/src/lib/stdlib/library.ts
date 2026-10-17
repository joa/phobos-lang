// The standard library as the docs present it, read from its Phobos sources: each file's functions,
// their parameters, the comment above each as its documentation, and the source text. Pure, so the
// tests and the pages parse the same text the compiler embeds. A function with no comment of its
// own that directly follows another (no blank line between) shares that one's documentation, as
// `exp2` and `log2` do.
//
// `evaluate` runs an expression function on numbers, which is how the activation figure plots what
// the shipped source computes and how the tests check the formulas against reference values.

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

type Node =
  | { kind: "num"; value: number }
  | { kind: "var"; name: string }
  | { kind: "neg"; arg: Node }
  | { kind: "bin"; op: "+" | "-" | "*" | "/"; lhs: Node; rhs: Node }
  | { kind: "call"; callee: string; args: Node[] }

function tokenize(text: string): string[] {
  const tokens = text.match(/\d+\.?\d*(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z0-9_]*|[()+\-*/,]|\S/g) ?? []
  return tokens
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
      const inner = sum()
      take(")")
      return inner
    }
    if (token === "-") return { kind: "neg", arg: primary() }
    if (/^\d/.test(token)) return { kind: "num", value: Number(token) }
    if (/^[A-Za-z_]/.test(token)) {
      if (peek() !== "(") return { kind: "var", name: token }
      take("(")
      const args: Node[] = []
      if (peek() !== ")") {
        args.push(sum())
        while (peek() === ",") {
          take(",")
          args.push(sum())
        }
      }
      take(")")
      return { kind: "call", callee: token, args }
    }
    throw new Error(`unexpected \`${token}\` in ${text}`)
  }
  const product = (): Node => {
    let lhs = primary()
    while (peek() === "*" || peek() === "/") {
      const op = take() as "*" | "/"
      lhs = { kind: "bin", op, lhs, rhs: primary() }
    }
    return lhs
  }
  function sum(): Node {
    let lhs = product()
    while (peek() === "+" || peek() === "-") {
      const op = take() as "+" | "-"
      lhs = { kind: "bin", op, lhs, rhs: product() }
    }
    return lhs
  }
  const node = sum()
  if (at !== tokens.length) throw new Error(`trailing \`${tokens[at]}\` in ${text}`)
  return node
}

// The built-ins an expression function may call, exactly where the hardware is: `exp`, `log`,
// `tanh` and `sqrt` are approximate on the device.
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
  min: (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.min(a, b)),
  max: (a, b) => (Number.isNaN(a) ? b : Number.isNaN(b) ? a : Math.max(a, b)),
  f32: x => x,
}

// An evaluator of the expression functions of `files`, in double precision.
export function evaluator(files: LibraryFile[]): (name: string, ...args: number[]) => number {
  const bodies = new Map<string, { params: string[]; body: Node }>()
  for (const fn of files.flatMap(file => file.functions)) {
    if (fn.kind !== "expression") continue
    const text = fn.source.slice(fn.source.indexOf("=") + 1)
    bodies.set(fn.name, { params: fn.params, body: parseExpression(text) })
  }
  const run = (node: Node, scope: Map<string, number>): number => {
    switch (node.kind) {
      case "num":
        return node.value
      case "var": {
        const value = scope.get(node.name)
        if (value === undefined) throw new Error(`unbound \`${node.name}\``)
        return value
      }
      case "neg":
        return -run(node.arg, scope)
      case "bin": {
        const [a, b] = [run(node.lhs, scope), run(node.rhs, scope)]
        return node.op === "+" ? a + b : node.op === "-" ? a - b : node.op === "*" ? a * b : a / b
      }
      case "call":
        return call(node.callee, node.args.map(arg => run(arg, scope)))
    }
  }
  const call = (name: string, args: number[]): number => {
    const builtin = BUILTINS[name]
    if (builtin) return builtin(...args)
    const fn = bodies.get(name)
    if (!fn) throw new Error(`\`${name}\` is not an expression function or a built-in the evaluator knows`)
    if (fn.params.length !== args.length) throw new Error(`\`${name}\` takes ${fn.params.length} arguments`)
    return run(fn.body, new Map(fn.params.map((param, i) => [param, args[i]])))
  }
  return (name, ...args) => call(name, args)
}
