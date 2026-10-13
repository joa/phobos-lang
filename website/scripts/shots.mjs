// Screenshots of site routes in {dark, light} x {1280, 390} x {motion, reduced}, a mid frame and an end
// frame each, for review by eye. Fails on console errors, page errors and horizontal scroll.
//
//   node scripts/shots.mjs [--base URL] [--out DIR] [--only dark,390,reduced] [--steps] [--nojs] ROUTE...
//
// --steps also saves one viewport shot per screen while scrolling down, for scroll-driven scenes.
// --nojs turns JavaScript off (the theme then stays dark), to check the pages' no-script end state.
// Uses puppeteer-core with the system Chrome; set CHROME_PATH to point elsewhere.

import { existsSync, mkdirSync } from "node:fs"
import { join } from "node:path"
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
const WIDTHS_PX = [1280, 390]
const MOTIONS = ["motion", "reduced"]
const HEIGHT_FOR_WIDTH_PX = { 1280: 800, 390: 844 }
const MID_FRAME_MS = 650
const SETTLE_MS = 2200
const SCROLL_STEP_MS = 450

// Dev-server chatter that is not a page error.
const IGNORED_CONSOLE = [/\[vite\]/i, /\[astro\]/i, /Download the React DevTools/i]

function parseArgs(argv) {
  const options = { base: "http://localhost:4401", out: "shots", only: [], steps: false, noJs: false, routes: [] }
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (arg === "--base") options.base = argv[++index]
    else if (arg === "--out") options.out = argv[++index]
    else if (arg === "--only") options.only = argv[++index].split(",")
    else if (arg === "--steps") options.steps = true
    else if (arg === "--nojs") options.noJs = true
    else options.routes.push(arg)
  }
  if (options.routes.length === 0) options.routes.push("/")
  return options
}

function slug(route) {
  const trimmed = route.replace(/^\/+|\/+$/g, "").replace(/[^a-z0-9]+/gi, "-")
  return trimmed === "" ? "home" : trimmed
}

const wait = durationMs => new Promise(resolve => setTimeout(resolve, durationMs))

async function shoot(browser, options, route, theme, widthPx, motion) {
  const page = await browser.newPage()
  const problems = []
  page.on("pageerror", error => problems.push(`pageerror: ${error.message}`))
  page.on("console", message => {
    if (message.type() !== "error") return
    const text = message.text()
    if (!IGNORED_CONSOLE.some(pattern => pattern.test(text))) problems.push(`console.error: ${text}`)
  })
  page.on("requestfailed", request => {
    if (!request.url().includes("/@vite/")) problems.push(`request failed: ${request.url()}`)
  })

  await page.setViewport({ width: widthPx, height: HEIGHT_FOR_WIDTH_PX[widthPx], deviceScaleFactor: 1 })
  await page.emulateMediaFeatures([
    { name: "prefers-reduced-motion", value: motion === "reduced" ? "reduce" : "no-preference" },
  ])
  await page.evaluateOnNewDocument(value => {
    try {
      localStorage.setItem("starlight-theme", value)
    } catch {}
  }, theme)

  if (options.noJs) await page.setJavaScriptEnabled(false)
  const name = `${slug(route)}.${theme}.${widthPx}.${motion}${options.noJs ? ".nojs" : ""}`
  await page.goto(new URL(route, options.base).href, { waitUntil: "networkidle0" })
  await page.evaluate(() => document.fonts.ready)
  await wait(MID_FRAME_MS)
  await page.screenshot({ path: join(options.out, `${name}.mid.png`) })

  const pageHeightPx = await page.evaluate(() => document.documentElement.scrollHeight)
  const viewportPx = HEIGHT_FOR_WIDTH_PX[widthPx]
  let step = 0
  for (let topPx = 0; topPx < pageHeightPx; topPx += Math.round(viewportPx * 0.8)) {
    await page.evaluate(y => window.scrollTo(0, y), topPx)
    await wait(SCROLL_STEP_MS)
    if (options.steps) {
      await page.screenshot({ path: join(options.out, `${name}.step${String(step).padStart(2, "0")}.png`) })
    }
    step += 1
  }
  await page.evaluate(() => window.scrollTo(0, 0))
  await wait(SETTLE_MS)
  await page.screenshot({ path: join(options.out, `${name}.end.png`), fullPage: true })

  const overflowPx = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth
  )
  if (overflowPx > 0) problems.push(`horizontal scroll: ${overflowPx}px`)
  // The dev server reports build errors in an overlay rather than the console.
  if (await page.$("vite-error-overlay")) problems.push("dev server error overlay is showing")
  await page.close()
  return { name, problems }
}

async function main() {
  const options = parseArgs(process.argv.slice(2))
  const executablePath = CHROME_CANDIDATES.find(path => path && existsSync(path))
  if (!executablePath) throw new Error("no Chrome found; set CHROME_PATH")
  mkdirSync(options.out, { recursive: true })

  const browser = await puppeteer.launch({ executablePath, headless: true, args: ["--hide-scrollbars"] })
  let failures = 0
  try {
    for (const route of options.routes) {
      for (const theme of THEMES) {
        for (const widthPx of WIDTHS_PX) {
          for (const motion of MOTIONS) {
            const tags = [theme, String(widthPx), motion]
            if (options.only.length > 0 && !options.only.every(tag => tags.includes(tag))) continue
            const { name, problems } = await shoot(browser, options, route, theme, widthPx, motion)
            if (problems.length === 0) {
              console.log(`ok    ${name}`)
              continue
            }
            failures += 1
            console.log(`FAIL  ${name}`)
            for (const problem of problems) console.log(`      ${problem}`)
          }
        }
      }
    }
  } finally {
    await browser.close()
  }
  if (failures > 0) {
    console.log(`${failures} shot(s) failed`)
    process.exit(1)
  }
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})
