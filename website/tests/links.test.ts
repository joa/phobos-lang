// Every internal link and anchor in the built site resolves. It reads dist/ as the last build left it
// and builds nothing, so it is skipped until a build has run. starlight-links-validator checks the
// Markdown pages at build time; this also covers the custom pages and links drawn by components.

import { existsSync, readFileSync, readdirSync, statSync } from "node:fs"
import { join, relative, sep } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const DIST = fileURLToPath(new URL("../dist/", import.meta.url))
const SITE_ORIGIN = "https://phobos-lang.dev"

function htmlFiles(dir: string): string[] {
  return readdirSync(dir).flatMap(name => {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) return htmlFiles(path)
    return name.endsWith(".html") ? [path] : []
  })
}

// The URL path a built file is served at.
function routeOf(file: string): string {
  const path = "/" + relative(DIST, file).split(sep).join("/")
  return path.endsWith("/index.html") ? path.slice(0, -"index.html".length) : path
}

// The built file a URL path is served from, or null.
function fileFor(pathname: string): string | null {
  const decoded = decodeURIComponent(pathname)
  const candidates = decoded.endsWith("/")
    ? [decoded + "index.html"]
    : [decoded, decoded + "/index.html", decoded + ".html"]
  for (const candidate of candidates) {
    const file = join(DIST, ...candidate.split("/"))
    if (existsSync(file) && statSync(file).isFile()) return file
  }
  return null
}

function idsOf(html: string): Set<string> {
  return new Set([...html.matchAll(/\s(?:id|name)="([^"]+)"/g)].map(match => match[1]))
}

function hrefsOf(html: string): string[] {
  return [...html.matchAll(/<a\s[^>]*?href="([^"]*)"/g)].map(match => match[1].replaceAll("&amp;", "&"))
}

describe.skipIf(!existsSync(join(DIST, "index.html")))("links in dist/", () => {
  it("resolve to a built page and an id on it", () => {
    const pages = new Map(htmlFiles(DIST).map(file => [file, readFileSync(file, "utf8")]))
    const idCache = new Map<string, Set<string>>()
    const ids = (file: string) => {
      let found = idCache.get(file)
      if (!found) idCache.set(file, (found = idsOf(pages.get(file) ?? readFileSync(file, "utf8"))))
      return found
    }
    const broken: string[] = []
    for (const [file, html] of pages) {
      const base = new URL(routeOf(file), SITE_ORIGIN)
      for (const href of hrefsOf(html)) {
        if (href === "" || /^(mailto|javascript|data):/.test(href)) continue
        const url = new URL(href, base)
        if (url.origin !== SITE_ORIGIN) continue
        const target = fileFor(url.pathname)
        if (!target) {
          broken.push(`${routeOf(file)}: ${href} (no page)`)
          continue
        }
        const hash = decodeURIComponent(url.hash.slice(1))
        if (hash !== "" && hash !== "top" && target.endsWith(".html") && !ids(target).has(hash)) {
          broken.push(`${routeOf(file)}: ${href} (no #${hash})`)
        }
      }
    }
    expect(broken).toEqual([])
  })
})
