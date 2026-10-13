// axe-core over every route in dist/, in both themes, against a running server (astro preview).
//
//   node scripts/axe.mjs [--base URL] [--json FILE] [ROUTE...]
//
// With no routes, every index.html under dist/ is a route. Prints each violation once with the
// routes and themes it occurs on; exits 1 if there is any. Uses puppeteer-core with the system
// Chrome; set CHROME_PATH to point elsewhere.

import { existsSync, readdirSync, statSync, writeFileSync } from "node:fs"
import { join, relative, sep } from "node:path"
import { fileURLToPath } from "node:url"
import { AxePuppeteer } from "@axe-core/puppeteer"
import puppeteer from "puppeteer-core"

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
]
const THEMES = ["dark", "light"]
const DIST = fileURLToPath(new URL("../dist/", import.meta.url))
const SETTLE_MS = 600

function parseArgs(argv) {
  const options = { base: "http://localhost:4408", json: null, routes: [] }
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (arg === "--base") options.base = argv[++index]
    else if (arg === "--json") options.json = argv[++index]
    else options.routes.push(arg)
  }
  return options
}

function distRoutes(dir = DIST) {
  return readdirSync(dir).flatMap(name => {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) return distRoutes(path)
    if (name !== "index.html") return []
    return ["/" + relative(DIST, dir).split(sep).join("/") + (dir === DIST ? "" : "/")].map(route =>
      route.replace(/\/+/g, "/")
    )
  })
}

const wait = durationMs => new Promise(resolve => setTimeout(resolve, durationMs))

const options = parseArgs(process.argv.slice(2))
const routes = options.routes.length > 0 ? options.routes : [...distRoutes(), "/404.html"]
const executablePath = CHROME_CANDIDATES.find(candidate => candidate && existsSync(candidate))
const browser = await puppeteer.launch({ executablePath, headless: true })
const found = new Map()
const incomplete = new Map()
try {
  for (const route of routes) {
    for (const theme of THEMES) {
      const page = await browser.newPage()
      await page.setViewport({ width: 1280, height: 800 })
      await page.evaluateOnNewDocument(value => {
        try {
          localStorage.setItem("starlight-theme", value)
        } catch {}
      }, theme)
      await page.goto(new URL(route, options.base).href, { waitUntil: "networkidle0" })
      // Figures start as they near the viewport; scroll once through the page so all of them are live.
      await page.evaluate(async () => {
        for (let y = 0; y < document.body.scrollHeight; y += innerHeight) {
          scrollTo(0, y)
          await new Promise(resolve => setTimeout(resolve, 60))
        }
        scrollTo(0, 0)
      })
      await wait(SETTLE_MS)
      const results = await new AxePuppeteer(page).analyze()
      for (const [bucket, list] of [
        [found, results.violations],
        [incomplete, results.incomplete],
      ]) {
        for (const violation of list) {
          const entry = bucket.get(violation.id) ?? { help: violation.help, impact: violation.impact, where: [] }
          entry.where.push({ route, theme, targets: violation.nodes.slice(0, 4).map(node => node.target.join(" ")) })
          bucket.set(violation.id, entry)
        }
      }
      await page.close()
    }
  }
} finally {
  await browser.close()
}

console.log(`${routes.length} routes x ${THEMES.length} themes`)
for (const [label, bucket] of [
  ["violation", found],
  ["incomplete", incomplete],
]) {
  for (const [id, entry] of bucket) {
    console.log(`\n${label} ${id} (${entry.impact}): ${entry.help}, on ${entry.where.length} page loads`)
    for (const { route, theme, targets } of entry.where.slice(0, 6))
      console.log(`  ${route} ${theme}: ${targets.join(" | ")}`)
  }
}
if (options.json)
  writeFileSync(
    options.json,
    JSON.stringify({ violations: Object.fromEntries(found), incomplete: Object.fromEntries(incomplete) }, null, 2)
  )
process.exitCode = found.size > 0 ? 1 : 0
