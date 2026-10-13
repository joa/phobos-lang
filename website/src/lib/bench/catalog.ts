// What the engine runs, read from the tree at build time: the models the README lists as tested, and
// the quantized formats phobos-gguf decodes, one file per format in phobos-gguf/src/quant/.

import { readdirSync, readFileSync } from "node:fs"
import { resolve } from "node:path"
import { repoRoot } from "./load"

export interface TestedModel {
  name: string
  url: string
  note: string
}

// The README's "Inference has been tested with" list, GGUF only.
export function testedModels(root = repoRoot()): TestedModel[] {
  const readme = readFileSync(resolve(root, "README.md"), "utf8").replace(/\r\n?/g, "\n")
  const start = readme.indexOf("Inference has been tested with")
  if (start < 0) throw new Error("README.md has no tested-models list")
  const block = readme.slice(start).split("\n\n")[1] ?? ""
  return block
    .split("\n")
    .map(line => line.match(/^- \[([^\]]+)\]\(([^)]+)\)\s*(.*)$/))
    .filter((found): found is RegExpMatchArray => found !== null)
    .map(([, name, url, rest]) => ({ name, url, note: rest.replace(/^\(|\)$/g, "").trim() }))
    .filter(model => !/onnx/i.test(model.name + model.note))
}

// A README name for a benchmarked model file, matched without regard to case.
export function displayName(model: string, tested: TestedModel[]): string {
  return tested.find(entry => entry.name.toLowerCase() === model.toLowerCase())?.name ?? model
}

export interface FormatFamily {
  family: string
  formats: string[]
}

const FAMILIES: [string, RegExp][] = [
  ["Block formats", /^Q\d_[01]$/],
  ["K-quants", /^Q\d_K$/],
  ["I-quants", /^IQ/],
  ["Ternary", /^PTQ/],
]

export function quantFormats(root = repoRoot()): string[] {
  const dir = resolve(root, "phobos-gguf/src/quant")
  const names: string[] = []
  for (const file of readdirSync(dir).sort()) {
    if (!file.endsWith(".rs")) continue
    const text = readFileSync(resolve(dir, file), "utf8")
    const found = text.match(/pub static SPEC: Spec = Spec \{\s*name: "([^"]+)"/)
    if (found) names.push(found[1])
  }
  return names
}

export function formatFamilies(formats: string[]): FormatFamily[] {
  const order = (name: string) => name.replace(/\d+/g, digits => digits.padStart(2, "0"))
  return FAMILIES.map(([family, pattern]) => ({
    family,
    formats: formats.filter(name => pattern.test(name)).sort((a, b) => order(a).localeCompare(order(b))),
  })).filter(group => group.formats.length > 0)
}
