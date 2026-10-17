import { readdirSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { evaluator, isHelper, parseLibrary, shippedFiles } from "../src/lib/stdlib/library"
import { FUNCTIONS, LIBRARY } from "../src/lib/stdlib/sources"

const DOCS = join(import.meta.dirname, "../src/content/docs/docs/stdlib")
const STDLIB_DIR = join(import.meta.dirname, "../../phobos-lang/src/stdlib")

describe("standard library sources", () => {
  it("covers every file in phobos-lang/src/stdlib", () => {
    const onDisk = readdirSync(STDLIB_DIR).filter(name => name.endsWith(".ph"))
    expect(LIBRARY.map(file => file.file.replace("std/", "")).sort()).toEqual(onDisk.sort())
  })

  it("documents every function and opens every file with a comment", () => {
    for (const file of LIBRARY) expect(file.intro.length, file.file).toBeGreaterThan(0)
    const undocumented = FUNCTIONS.filter(fn => fn.doc.length === 0).map(fn => fn.name)
    expect(undocumented).toEqual([])
  })

  it("has a page entry for every public function", () => {
    const pages = readdirSync(DOCS)
      .filter(name => name.endsWith(".mdx"))
      .map(name => readFileSync(join(DOCS, name), "utf8"))
      .join("\n")
    const listed = new Set([...pages.matchAll(/<Fn\s+names?="([^"]+)"/g)].flatMap(match => match[1].split(/\s+/)))
    const missing = FUNCTIONS.filter(fn => !isHelper(fn.name) && !listed.has(fn.name)).map(fn => fn.name)
    expect(missing).toEqual([])
    const unknown = [...listed].filter(name => !FUNCTIONS.some(fn => fn.name === name))
    expect(unknown).toEqual([])
  })

  it("reads the embedded file list from stdlib.rs", () => {
    const rs = 'pub const FILES = &[\n    ("std/a.ph", include_str!("stdlib/a.ph")),\n    ("std/b.ph", include_str!("stdlib/b.ph")),\n];'
    expect(shippedFiles(rs)).toEqual([
      { file: "std/a.ph", path: "stdlib/a.ph" },
      { file: "std/b.ph", path: "stdlib/b.ph" },
    ])
  })

  it("parses kinds, parameters and shared documentation", () => {
    const file = parseLibrary(
      "std/t.ph",
      "// Intro.\n\n// Twice.\nfn double(x) = x * 2.0\nfn triple(x) = x * 3.0\n\n// In place.\n//\n// Second.\nfn clear(t, n) {\n  t = 0.0\n}\n"
    )
    expect(file.intro).toEqual(["Intro."])
    expect(file.functions.map(fn => [fn.name, fn.kind, fn.params, fn.doc, fn.docOf])).toEqual([
      ["double", "expression", ["x"], ["Twice."], "double"],
      ["triple", "expression", ["x"], ["Twice."], "double"],
      ["clear", "statement", ["t", "n"], ["In place.", "Second."], "clear"],
    ])
    expect(file.functions[2].source).toBe("fn clear(t, n) {\n  t = 0.0\n}")
  })
})

describe("the library's expression functions", () => {
  const f = evaluator(LIBRARY)

  // References: the closed forms, in double precision.
  const cases: [string, number[], number, number][] = [
    ["relu", [-1.5], 0, 0],
    ["relu6", [7], 6, 0],
    ["leaky_relu", [-2, 0.01], -0.02, 1e-15],
    ["hardtanh", [-3], -1, 0],
    ["elu", [-1, 1], Math.exp(-1) - 1, 1e-15],
    ["selu", [-1], -1.1113307378125625, 1e-12],
    ["selu", [2], 2.101401974710961, 1e-12],
    ["celu", [-1, 2], 2 * (Math.exp(-0.5) - 1), 1e-15],
    ["sigmoid", [0], 0.5, 0],
    ["log_sigmoid", [0], -Math.LN2, 1e-15],
    ["log_sigmoid", [-50], -50, 1e-12],
    ["hardsigmoid", [1], 1 / 6 + 0.5, 1e-15],
    ["silu", [1], 0.7310585786300049, 1e-15],
    ["swish", [1, 2], 1 / (1 + Math.exp(-2)), 1e-15],
    ["hardswish", [1], 2 / 3, 1e-15],
    ["erf", [0.5], 0.5204998778130465, 1.5e-7],
    ["erf", [-1.2], -0.9103139782296353, 1.5e-7],
    ["erf", [3], 0.9999779095030014, 1.5e-7],
    ["erf", [0], 0, 1e-9],
    ["gelu", [1], 0.8413447460685429, 2e-7],
    ["gelu", [-0.5], -0.15426876936299344, 2e-7],
    ["gelu_tanh", [1], 0.8411919906082768, 1e-12],
    ["quick_gelu", [1], 1 / (1 + Math.exp(-1.702)), 1e-15],
    ["softplus", [0], Math.LN2, 1e-15],
    ["softplus", [40], 40, 1e-12],
    ["mish", [1], 0.8650983882673103, 1e-12],
    ["softsign", [3], 0.75, 0],
    ["squared_relu", [-2], 0, 0],
    ["squared_relu", [3], 9, 0],
    ["swiglu", [1, 2], 2 * 0.7310585786300049, 1e-15],
    ["reglu", [-1, 2], 0, 0],
    ["clamp", [5, -1, 1], 1, 0],
    ["lerp", [2, 4, 0.25], 2.5, 0],
    ["exp2", [3], 8, 1e-12],
    ["log2", [8], 3, 1e-12],
    ["sign", [-0.25], -1, 0],
    ["sign", [0], 0, 0],
    ["rcp", [4], 0.25, 0],
  ]

  it.each(cases)("%s(%s)", (name, args, want, tolerance) => {
    expect(Math.abs(f(name, ...args) - want)).toBeLessThanOrEqual(tolerance)
  })

  it("keeps erf within its stated bound everywhere", () => {
    // erf by its series, which converges fast for |x| <= 3.
    const reference = (x: number) => {
      let term = x
      let total = x
      for (let n = 1; n < 80; n += 1) {
        term *= (-x * x) / n
        total += term / (2 * n + 1)
      }
      return (2 / Math.sqrt(Math.PI)) * total
    }
    let worst = 0
    for (let x = -3; x <= 3; x += 0.01) worst = Math.max(worst, Math.abs(f("erf", x) - reference(x)))
    expect(worst).toBeLessThan(1.5e-7)
  })
})
