// The traces against what the site shows: every traced kernel's trace is fresh (its source hash
// matches the file or fence as it is now), every kernel a page shows with a trace has one, every
// attribute has a page, and every trace has the shape trace.schema.json describes.

import { createHash } from "node:crypto"
import { existsSync, readdirSync, readFileSync } from "node:fs"
import { join, relative } from "node:path"
import { describe, expect, it } from "vitest"
import attributes from "../src/data/attributes.json"
import schema from "../src/data/trace.schema.json"
import kernels from "../kernels.json"
import type { Trace } from "../src/lib/viz/trace"

const WEBSITE = join(import.meta.dirname, "..")
const REPO = join(WEBSITE, "..")
const TRACES = join(WEBSITE, "src/data/traces")
const DOCS = join(WEBSITE, "src/content/docs/docs")
const REGENERATE = "cargo run -p phobos-lang --features trace --example site -- traces"

const traces: { file: string; trace: Trace }[] = readdirSync(TRACES)
  .filter(file => file.endsWith(".json"))
  .map(file => ({ file, trace: JSON.parse(readFileSync(join(TRACES, file), "utf8")) as Trace }))

function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex")
}

function lf(text: string): string {
  return text.replace(/\r\n?/g, "\n")
}

function walk(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : entry.name.endsWith(".mdx") ? [join(dir, entry.name)] : []
  )
}

const pages = walk(DOCS).map(path => ({ path, text: lf(readFileSync(path, "utf8")) }))

interface Fence {
  page: string
  trace: string
  body: string
}

// Whole-kernel fences with a trace meta: the body with the fence's indentation stripped, joined by
// LF, one trailing LF, as the exporter hashes it.
function traceFences(): Fence[] {
  const fences: Fence[] = []
  for (const page of pages) {
    const lines = page.text.split("\n")
    for (let index = 0; index < lines.length; index += 1) {
      const open = /^(\s*)```phobos\s+(.*)$/.exec(lines[index])
      const trace = open && /(?:^|\s)trace="([^"]+)"/.exec(open[2])
      if (!open || !trace) continue
      const indent = open[1].length
      const body: string[] = []
      for (index += 1; index < lines.length && !/^\s*```\s*$/.test(lines[index]); index += 1) {
        body.push(lines[index].slice(Math.min(indent, lines[index].length - lines[index].trimStart().length)))
      }
      fences.push({
        page: relative(REPO, page.path).replace(/\\/g, "/"),
        trace: trace[1],
        body: body.join("\n") + "\n",
      })
    }
  }
  return fences
}

function expectedHash(trace: Trace): string {
  const source = trace.source
  if (source.origin === "fence") return sha256(source.text)
  const text = lf(readFileSync(join(REPO, source.path), "utf8"))
  const variant = source.variant
  if (!variant) return sha256(text)
  const parts = [
    ...(variant.add !== undefined ? [`"add":${JSON.stringify(variant.add)}`] : []),
    ...(variant.drop !== undefined ? [`"drop":${JSON.stringify(variant.drop)}`] : []),
  ]
  return sha256(`${text}{${parts.join(",")}}`)
}

describe("trace freshness", () => {
  it.each(traces.map(entry => [entry.file, entry.trace] as const))("%s matches its source", (_file, trace) => {
    if (trace.source.origin === "file") expect(existsSync(join(REPO, trace.source.path))).toBe(true)
    expect(trace.source.sha256, `stale trace; regenerate with ${REGENERATE}`).toBe(expectedHash(trace))
  })

  it("names each trace file after its trace and chip", () => {
    for (const { file, trace } of traces) expect(file).toBe(`${trace.name}.${trace.compiler.chip}.json`)
  })
})

describe("kernels the site shows", () => {
  const traced = new Set(traces.map(({ trace }) => trace.name))

  it("has a trace for every <Kernel file> on a docs page", () => {
    const files = pages.flatMap(page =>
      [...page.text.matchAll(/<Kernel\b[^>]*\bfile="([^"]+)"/g)].map(match => match[1])
    )
    expect(files.length).toBeGreaterThan(0)
    for (const file of files) {
      const kernel = kernels.kernels.find(entry => entry.path === file && !("variant" in entry))
      expect(kernel, `${file} is not in kernels.json`).toBeDefined()
      expect(traced.has(kernel!.name), `no trace for ${file}; generate it with ${REGENERATE}`).toBe(true)
    }
  })

  it("has a trace for every trace name a page uses", () => {
    const names = pages.flatMap(page =>
      [...page.text.matchAll(/\b(?:trace|beforeTrace|afterTrace)="([^"]+)"/g)].map(match => match[1])
    )
    expect(names.length).toBeGreaterThan(0)
    for (const name of names) expect(traced.has(name), `no trace named ${name}`).toBe(true)
  })

  it("has a fresh trace for every trace fence", () => {
    const fences = traceFences()
    expect(fences.length).toBeGreaterThan(0)
    for (const fence of fences) {
      const match = traces.find(({ trace }) => trace.name === fence.trace)?.trace
      expect(match, `no trace for the fence ${fence.trace} in ${fence.page}`).toBeDefined()
      expect(match!.source.origin).toBe("fence")
      expect(match!.source.path).toBe(fence.page)
      expect(match!.source.sha256, `stale trace ${fence.trace}; regenerate with ${REGENERATE}`).toBe(sha256(fence.body))
    }
  })

  it("has a page for every attribute", () => {
    for (const name of attributes) expect(existsSync(join(DOCS, "attributes", `${name}.mdx`)), name).toBe(true)
  })
})

// A structural check against trace.schema.json: types, required and unknown fields, enums, consts,
// patterns and tuple lengths, following $ref. Enough for the subset of JSON Schema the file uses.
type SchemaNode = Record<string, unknown>

function validate(value: unknown, node: SchemaNode, path: string, errors: string[]): void {
  if (typeof node.$ref === "string") {
    const target = node.$ref.replace("#/$defs/", "")
    validate(value, (schema.$defs as Record<string, SchemaNode>)[target], path, errors)
    return
  }
  if ("const" in node && value !== node.const) errors.push(`${path}: expected ${String(node.const)}`)
  if (Array.isArray(node.enum) && !node.enum.includes(value)) errors.push(`${path}: ${String(value)} not in enum`)
  const type = node.type
  if (type === "object") {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      errors.push(`${path}: not an object`)
      return
    }
    const record = value as Record<string, unknown>
    const properties = (node.properties ?? {}) as Record<string, SchemaNode>
    for (const key of (node.required ?? []) as string[]) if (!(key in record)) errors.push(`${path}.${key}: missing`)
    for (const [key, child] of Object.entries(record)) {
      if (properties[key]) validate(child, properties[key], `${path}.${key}`, errors)
      else if (node.additionalProperties === false) errors.push(`${path}.${key}: unknown field`)
      else if (typeof node.additionalProperties === "object")
        validate(child, node.additionalProperties as SchemaNode, `${path}.${key}`, errors)
    }
  } else if (type === "array") {
    if (!Array.isArray(value)) {
      errors.push(`${path}: not an array`)
      return
    }
    if (typeof node.minItems === "number" && value.length < node.minItems) errors.push(`${path}: too short`)
    if (typeof node.maxItems === "number" && value.length > node.maxItems) errors.push(`${path}: too long`)
    if (node.items)
      value.forEach((item, index) => validate(item, node.items as SchemaNode, `${path}[${index}]`, errors))
  } else if (type === "integer") {
    if (!Number.isInteger(value)) errors.push(`${path}: not an integer`)
  } else if (type === "number") {
    if (typeof value !== "number") errors.push(`${path}: not a number`)
  } else if (type === "string") {
    if (typeof value !== "string") errors.push(`${path}: not a string`)
    else if (typeof node.pattern === "string" && !new RegExp(node.pattern).test(value)) errors.push(`${path}: pattern`)
  } else if (type === "boolean") {
    if (typeof value !== "boolean") errors.push(`${path}: not a boolean`)
  }
}

describe("trace schema", () => {
  it.each(traces.map(entry => [entry.file, entry.trace] as const))("%s validates", (_file, trace) => {
    const errors: string[] = []
    validate(trace, schema as unknown as SchemaNode, "trace", errors)
    expect(errors.slice(0, 10)).toEqual([])
  })

  it("catches a wrong field", () => {
    const errors: string[] = []
    const broken = { ...traces[0].trace, schema: 2, extra: true }
    validate(broken, schema as unknown as SchemaNode, "trace", errors)
    expect(errors.length).toBeGreaterThanOrEqual(2)
  })
})
