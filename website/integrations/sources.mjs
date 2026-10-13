// Renders the repository's normative root files (SPEC.md, ENV.md, BENCHMARKS.md,
// WINDOWS.md) in place as Starlight pages. The files stay the single source: this
// integration copies them into a gitignored content folder on every start, injects
// front matter, tags Phobos code fences and rewrites repository-relative links to
// site routes or GitHub URLs. In dev it watches the root files and rewrites on change.

import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs"
import { dirname, join, posix, resolve } from "node:path"
import { fileURLToPath } from "node:url"

export const REPO_URL = "https://github.com/joa/phobos-lang"
export const BLOB_URL = `${REPO_URL}/blob/main/`
export const TREE_URL = `${REPO_URL}/tree/main/`

/** The root files rendered as pages, with the route each one gets. */
export const SOURCES = [
  { file: "SPEC.md", route: "/docs/reference/spec/", label: "Language spec", order: 1 },
  { file: "ENV.md", route: "/docs/reference/environment/", label: "Environment variables", order: 2 },
  { file: "BENCHMARKS.md", route: "/docs/reference/benchmarks/", label: "Benchmarks", order: 3 },
  {
    file: "WINDOWS.md",
    route: "/docs/reference/windows/",
    label: "Building on Windows",
    title: "Building on Windows",
    order: 5,
  },
]

const REPO_ROOT = fileURLToPath(new URL("../../", import.meta.url))
const OUT_DIR = fileURLToPath(new URL("../src/content/docs/docs/reference/generated/", import.meta.url))

/** The slug Starlight routes a page to, from its route. */
export function slugOf(route) {
  return route.replace(/^\/|\/$/g, "")
}

/**
 * Splits a markdown text into front matter and body. The title comes from the
 * first line if it is an H1, which is then removed; otherwise `fallbackTitle`
 * is used and the body is left whole.
 *
 * @param {string} text
 * @param {{ title?: string, label?: string, order: number, route: string, source: string }} page
 */
export function frontMatter(text, { title: fallbackTitle, label, order, route, source }) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n")
  let first = 0
  while (first < lines.length && lines[first].trim() === "") first++
  const h1 = /^#\s+(.+?)\s*#*\s*$/.exec(lines[first] ?? "")
  let title = fallbackTitle
  let body = lines
  if (h1) {
    title = h1[1]
    body = lines.slice(first + 1)
  }
  if (!title) throw new Error(`${source}: no H1 and no fallback title`)
  while (body.length > 0 && body[0].trim() === "") body = body.slice(1)
  const yaml = [
    "---",
    `title: ${JSON.stringify(title)}`,
    `slug: ${JSON.stringify(slugOf(route))}`,
    `description: ${JSON.stringify(`${source}, rendered from the repository.`)}`,
    "sidebar:",
    `  label: ${JSON.stringify(label ?? title)}`,
    `  order: ${order}`,
    `editUrl: ${JSON.stringify(BLOB_URL + source)}`,
    "---",
    "",
    `<!-- Generated from ${source} by integrations/sources.mjs. Edit ${source}, not this file. -->`,
  ]
  return `${yaml.join("\n")}\n\n${body.join("\n")}`
}

/** Whether a fence body is a Phobos program: an optional attribute run, then a kernel or fn header. */
export function isPhobosCode(body) {
  return /^(?:\s*(?:\/\/[^\n]*|@\w+[^\n]*)\n)*\s*(?:kernel\s+\w+\s*\(|fn\s+\w+\s*\()/.test(body)
}

/**
 * Calls `outside` on every stretch of text that is not inside a fenced code
 * block and `inside` on every fence (opening line, body, closing line).
 */
function mapFences(text, { outside = s => s, inside = (open, body, close) => open + body + close }) {
  const fence = /^([ \t]*)(`{3,}|~{3,})([^\n]*)\n([\s\S]*?)^\1\2[ \t]*$/gm
  let out = ""
  let last = 0
  for (const match of text.matchAll(fence)) {
    out += outside(text.slice(last, match.index))
    const [, indent, marks, info, body] = match
    out += inside(`${indent}${marks}${info}\n`, body, `${indent}${marks}`)
    last = match.index + match[0].length
  }
  return out + outside(text.slice(last))
}

/** Retags `plain` and untagged fences that hold Phobos code as `phobos`, and `ebnf` as `text`. Others keep their tag. */
export function tagFences(text) {
  return mapFences(text, {
    inside(open, body, close) {
      const m = /^([ \t]*)(`{3,}|~{3,})([^\n]*)\n$/.exec(open)
      const info = m[3].trim()
      const [lang = "", ...meta] = info.split(/\s+/)
      let tag = lang
      if ((lang === "" || lang === "plain" || lang === "text") && isPhobosCode(body)) tag = "phobos"
      // The highlighter has no EBNF grammar; plain text says so without a build warning.
      else if (lang === "ebnf") tag = "text"
      if (tag !== lang) open = `${m[1]}${m[2]}${[tag, ...meta].join(" ")}\n`
      return open + body + close
    },
  })
}

/**
 * Lifts the headings outside code fences so the shallowest is an H2, the level under the page
 * title. A file that starts at H3, as WINDOWS.md does, would otherwise skip a level.
 */
export function liftHeadings(text) {
  const heading = /^(#{2,6})(?=\s)/gm
  let shallowest = 7
  mapFences(text, {
    outside(chunk) {
      for (const [, marks] of chunk.matchAll(heading)) shallowest = Math.min(shallowest, marks.length)
      return chunk
    },
  })
  if (shallowest === 7 || shallowest === 2) return text
  const lift = shallowest - 2
  return mapFences(text, { outside: chunk => chunk.replace(heading, marks => marks.slice(lift)) })
}

/**
 * Resolves one link target found in a root file. Returns the rewritten target,
 * plus the repository path of an image to copy beside the page when it is one.
 */
export function rewriteTarget(target, { image = false, isDir = () => false } = {}) {
  if (/^[a-z][a-z0-9+.-]*:/i.test(target) || target.startsWith("#") || target.startsWith("/")) {
    return { target }
  }
  const hashAt = target.indexOf("#")
  const path = hashAt < 0 ? target : target.slice(0, hashAt)
  const hash = hashAt < 0 ? "" : target.slice(hashAt)
  const normalized = posix
    .normalize(path.replace(/\\/g, "/"))
    .replace(/^(\.\/)+/, "")
    .replace(/\/$/, "")
  if (normalized === "" || normalized === ".") return { target: REPO_URL + hash }
  if (normalized.startsWith("../")) throw new Error(`link leaves the repository: ${target}`)
  const source = SOURCES.find(s => s.file === normalized)
  if (source) return { target: source.route + hash }
  if (image) return { target: `./${normalized}`, copy: normalized }
  return { target: (isDir(normalized) ? TREE_URL : BLOB_URL) + normalized + hash }
}

/**
 * Rewrites inline links and images outside code fences and inline code.
 * Returns the text and the repository paths of the images it refers to.
 */
export function rewriteLinks(text, options = {}) {
  const copies = new Set()
  const rewritten = mapFences(text, {
    outside(chunk) {
      // Inline code is set aside first, so a link whose label is code is still a link.
      const spans = []
      const masked = chunk.replace(/`+[^`]*`+/g, span => `\u0000${spans.push(span) - 1}\u0000`)
      const linked = masked.replace(
        /(!?)\[([^\]]*)\]\(\s*<?([^)\s>]+)>?((?:\s+"[^"]*")?)\s*\)/g,
        (_, bang, label, target, title) => {
          const result = rewriteTarget(target, { ...options, image: bang === "!" })
          if (result.copy) copies.add(result.copy)
          return `${bang}[${label}](${result.target}${title})`
        }
      )
      return linked.replace(/\u0000(\d+)\u0000/g, (_, i) => spans[Number(i)])
    },
  })
  return { text: rewritten, copies: [...copies] }
}

/** The page text for one root file, and the images it needs beside it. */
export function render(source, text, options = {}) {
  let body = text.replace(/\r\n?/g, "\n")
  if (source.file === "SPEC.md") body = tagFences(body)
  const { text: linked, copies } = rewriteLinks(body, options)
  return { page: frontMatter(liftHeadings(linked), { ...source, source: source.file }), copies }
}

function isRepoDir(root) {
  return path => {
    try {
      return statSync(join(root, path)).isDirectory()
    } catch {
      return false
    }
  }
}

/** Writes every page into `outDir`, replacing what was there. */
export function generate({ root = REPO_ROOT, outDir = OUT_DIR, logger } = {}) {
  rmSync(outDir, { recursive: true, force: true })
  mkdirSync(outDir, { recursive: true })
  for (const source of SOURCES) writeOne(source, { root, outDir, logger })
}

function writeOne(source, { root, outDir, logger }) {
  const from = join(root, source.file)
  if (!existsSync(from)) {
    logger?.warn(`${source.file} not found, skipped`)
    return
  }
  const { page, copies } = render(source, readFileSync(from, "utf8"), { isDir: isRepoDir(root) })
  writeFileSync(join(outDir, `${slugOf(source.route).split("/").pop()}.md`), page)
  for (const copy of copies) {
    const target = join(outDir, copy)
    mkdirSync(dirname(target), { recursive: true })
    copyFileSync(join(root, copy), target)
  }
}

/** The Astro integration. */
export default function sources() {
  return {
    name: "phobos-sources",
    hooks: {
      "astro:config:setup": ({ logger }) => {
        generate({ logger })
      },
      "astro:server:setup": ({ server, logger }) => {
        // Lowercased, since the watcher and Node can disagree on a Windows drive letter's case.
        const key = file => resolve(file).toLowerCase()
        const watched = new Map(SOURCES.map(source => [key(join(REPO_ROOT, source.file)), source]))
        server.watcher.add(SOURCES.map(source => join(REPO_ROOT, source.file)))
        server.watcher.on("change", file => {
          const source = watched.get(key(file))
          if (!source) return
          writeOne(source, { root: REPO_ROOT, outDir: OUT_DIR, logger })
          logger.info(`${source.file} rewritten`)
        })
      },
    },
  }
}
