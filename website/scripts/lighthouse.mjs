// Lighthouse scores of site routes, mobile and desktop, against a running server (astro preview).
//
//   node scripts/lighthouse.mjs [--base URL] [--json FILE] ROUTE...
//
// Prints one line per route and form factor with the four category scores and the core metrics.
// Uses the system Chrome; set CHROME_PATH to point elsewhere.

import { writeFileSync } from "node:fs"
import * as chromeLauncher from "chrome-launcher"
import lighthouse from "lighthouse"
import desktopConfig from "lighthouse/core/config/desktop-config.js"

const DEFAULT_ROUTES = ["/", "/docs/getting-started/", "/inference/", "/viz/gemm_fp32/"]
const CATEGORIES = ["performance", "accessibility", "best-practices", "seo"]
const METRICS = ["first-contentful-paint", "largest-contentful-paint", "total-blocking-time", "cumulative-layout-shift"]

function parseArgs(argv) {
  const options = { base: "http://localhost:4408", json: null, routes: [] }
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (arg === "--base") options.base = argv[++index]
    else if (arg === "--json") options.json = argv[++index]
    else options.routes.push(arg)
  }
  if (options.routes.length === 0) options.routes.push(...DEFAULT_ROUTES)
  return options
}

const options = parseArgs(process.argv.slice(2))
const chrome = await chromeLauncher.launch({ chromeFlags: ["--headless=new"] })
const results = []
try {
  for (const route of options.routes) {
    for (const formFactor of ["mobile", "desktop"]) {
      const flags = { port: chrome.port, output: "json", logLevel: "error", onlyCategories: CATEGORIES }
      const config = formFactor === "desktop" ? desktopConfig : undefined
      const { lhr } = await lighthouse(new URL(route, options.base).href, flags, config)
      const scores = Object.fromEntries(CATEGORIES.map(name => [name, Math.round(lhr.categories[name].score * 100)]))
      const metrics = Object.fromEntries(METRICS.map(name => [name, lhr.audits[name].displayValue]))
      const failing = Object.values(lhr.audits)
        .filter(audit => audit.score !== null && audit.score < 0.9 && audit.scoreDisplayMode !== "informative")
        .map(audit => audit.id)
      results.push({ route, formFactor, scores, metrics, failing })
      console.log(
        `${route} ${formFactor}: ${CATEGORIES.map(name => `${name} ${scores[name]}`).join(", ")} | ` +
          METRICS.map(name => `${name} ${metrics[name]}`).join(", ")
      )
      if (failing.length > 0) console.log(`  below 0.9: ${failing.join(", ")}`)
    }
  }
} finally {
  chrome.kill()
}
if (options.json) writeFileSync(options.json, JSON.stringify(results, null, 2))
