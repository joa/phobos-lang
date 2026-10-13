// Captures the --help of the four release binaries into src/data/cli/<tool>.txt,
// which the CLI reference page renders. The output is committed, so the Pages
// build needs no Rust. ONNX options are dropped from phobos-cli's help, since
// the site covers GGUF only.
//
//     node scripts/capture-cli.mjs [BIN_DIR]
//
// BIN_DIR defaults to target/release of the main checkout (the worktree's
// common git directory's parent), which is where `cargo build --release` puts
// the binaries. Nothing is run but `--help`.

import { execFileSync, spawnSync } from "node:child_process"
import { existsSync, mkdirSync, writeFileSync } from "node:fs"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

export const TOOLS = ["phobos-cli", "phobos-compile", "phobos-bench", "phobos-cache"]

const WEBSITE = fileURLToPath(new URL("../", import.meta.url))
const OUT_DIR = join(WEBSITE, "src", "data", "cli")

function defaultBinDir() {
  const common = execFileSync("git", ["rev-parse", "--path-format=absolute", "--git-common-dir"], {
    cwd: WEBSITE,
    encoding: "utf8",
  }).trim()
  return join(dirname(common), "target", "release")
}

/** Removes an option and its continuation lines from a help text. */
export function dropOption(text, flag) {
  const lines = text.split("\n")
  const out = []
  let dropping = false
  for (const line of lines) {
    const option = /^\s{1,8}-/.test(line)
    if (option) dropping = new RegExp(`^\\s{1,8}(?:-\\w, )?${flag}\\b`).test(line)
    else if (dropping && !/^\s{10,}\S/.test(line)) dropping = false
    if (!dropping) out.push(line)
  }
  return out.join("\n")
}

/** Makes phobos-cli's help GGUF only. */
export function dropOnnx(text) {
  let help = dropOption(text, "--onnx")
  help = dropOption(help, "--tokenizer")
  help = help.replace("(--gguf FILE | --onnx DIR)", "--gguf FILE")
  help = help.replace(
    "One of --gguf or --onnx is required; there is no default model.",
    "--gguf is required; there is no default model."
  )
  if (/onnx/i.test(help)) throw new Error("phobos-cli help still mentions ONNX; update dropOnnx")
  return help
}

/** Normalizes a captured help text: LF line ends, no `Error: ` prefix, no trailing blank lines. */
export function clean(text) {
  return `${text
    .replace(/\r\n?/g, "\n")
    .replace(/^Error: /, "")
    .replace(/[ \t]+$/gm, "")
    .trimEnd()}\n`
}

function capture(binDir, tool) {
  const exe = join(binDir, process.platform === "win32" ? `${tool}.exe` : tool)
  if (!existsSync(exe)) throw new Error(`${exe} not found; build it with cargo build --release -p ${tool}`)
  // Some tools print usage to stderr or exit non-zero on --help; both streams count.
  const run = spawnSync(exe, ["--help"], { encoding: "utf8", timeout: 30_000 })
  if (run.error) throw run.error
  const text = clean(`${run.stdout ?? ""}${run.stderr ?? ""}`)
  return tool === "phobos-cli" ? dropOnnx(text) : text
}

function main() {
  const binDir = resolve(process.argv[2] ?? defaultBinDir())
  mkdirSync(OUT_DIR, { recursive: true })
  for (const tool of TOOLS) {
    const text = capture(binDir, tool)
    writeFileSync(join(OUT_DIR, `${tool}.txt`), text)
    console.log(`${tool}: ${text.split("\n").length - 1} lines`)
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main()
