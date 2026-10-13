// Writes src/data/pipeline/gemm_fp32.json, the excerpts of the front page's compiler scene, from
// real compiler output for examples/gemm_fp32.ph. Produce the inputs with the toolchain first:
//
//   cargo run -p phobos-lang --example ir examples/gemm_fp32.ph > ir.txt
//   cargo run -p phobos-lang --example emit examples/gemm_fp32.ph > mlir.txt
//   cargo run -p phobos-compile -- examples/gemm_fp32.ph --chip sm_75 -o gemm.ptx
//   a `{:#?}` dump of phobos_lang::parse for the AST > ast.txt
//
// then: node scripts/pipeline-excerpts.mjs DIR   (DIR holds ast.txt, ir.txt, mlir.txt, gemm.ptx)
//
// Every excerpt is a contiguous slice of that output, cut at anchors, so it stays real when the
// compiler changes; only the passes excerpt is a summary, of the trace the visualizer uses.

import { readFileSync, writeFileSync, mkdirSync } from "node:fs"
import { join } from "node:path"

const dir = process.argv[2]
if (!dir) throw new Error("usage: node scripts/pipeline-excerpts.mjs DIR")

const read = name => readFileSync(join(dir, name), "utf8").replace(/\r\n?/g, "\n").split("\n")
const repo = new URL("../../", import.meta.url)

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

const source = readFileSync(new URL("examples/gemm_fp32.ph", repo), "utf8").replace(/\r\n?/g, "\n").split("\n")
const trace = JSON.parse(readFileSync(new URL("website/src/data/traces/gemm_fp32.sm_75.json", repo), "utf8"))

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
  ast: compactDebug(slice(read("ast.txt"), /^\s*For \{/, 60)).split("\n").slice(0, 14).join("\n") + "\n...",
  graph: slice(read("ir.txt"), /= for %/, 10),
  passes,
  mlir: slice(read("mlir.txt"), /= scf\.for .*iter_args/, 9) + "\n...",
  ptx: ptxLines.join("\n"),
}

const out = new URL("website/src/data/pipeline/", repo)
mkdirSync(out, { recursive: true })
writeFileSync(new URL("gemm_fp32.json", out), JSON.stringify({ stages }, null, 2) + "\n")
console.log(Object.keys(stages).join(", "))
