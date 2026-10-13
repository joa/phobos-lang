// Records, for every results/bench*.json and results/agent*.json without a meta.date, and for
// results/results.csv (phobos-kbench's output, which has no date field), the date of the last commit
// that touched it, into src/lib/bench/dates.json. CI checks out one commit, so git history
// is not there at build time; run this after committing a new results file that has no date field.
//
//   node scripts/bench-dates.mjs

import { execFileSync } from "node:child_process"
import { readdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const websiteDir = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const repoRoot = resolve(websiteDir, "..")
const outPath = resolve(websiteDir, "src/lib/bench/dates.json")

const dates = {}
for (const name of readdirSync(resolve(repoRoot, "results")).sort()) {
  const isRun = /^(bench|agent).*\.json$/.test(name)
  if (!isRun && name !== "results.csv") continue
  const file = `results/${name}`
  if (isRun && JSON.parse(readFileSync(resolve(repoRoot, file), "utf8")).meta?.date) continue
  const date = execFileSync("git", ["-C", repoRoot, "log", "-1", "--format=%cI", "--", file], {
    encoding: "utf8",
  }).trim()
  if (!date) throw new Error(`${file} has no meta.date and no commit; commit it first`)
  dates[file] = date
}

writeFileSync(outPath, JSON.stringify(dates, null, 2) + "\n")
console.log(`wrote ${Object.keys(dates).length} dates to ${outPath}`)
