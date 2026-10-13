import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

// WCAG 2.x contrast for every text/background and figure-stroke/background pair the site uses,
// in both themes. tokens.css writes each color as light-dark(dawn, night).

const tokensCss = readFileSync(new URL("../src/styles/tokens.css", import.meta.url), "utf8")

type Theme = "dawn" | "night"

function parseTokens(css: string): Record<Theme, Map<string, string>> {
  const themes = { dawn: new Map<string, string>(), night: new Map<string, string>() }
  for (const match of css.matchAll(/(--[a-z0-9-]+):\s*light-dark\(\s*(#[0-9a-f]{6})\s*,\s*(#[0-9a-f]{6})\s*\)/gi)) {
    themes.dawn.set(match[1], match[2])
    themes.night.set(match[1], match[3])
  }
  return themes
}

function relativeLuminance(hex: string): number {
  const channels = [1, 3, 5].map(index => parseInt(hex.slice(index, index + 2), 16) / 255)
  const [r, g, b] = channels.map(c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4))
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrastRatio(foreground: string, background: string): number {
  const [hi, lo] = [relativeLuminance(foreground), relativeLuminance(background)].sort((a, b) => b - a)
  return (hi + 0.05) / (lo + 0.05)
}

const BODY_TEXT_RATIO = 4.5
const STROKE_RATIO = 3

// Text: body copy, muted labels, links and their hover. --nebula and --mars label figure parts at 11 units,
// which is small text, so they are held to the body ratio too.
const TEXT = ["--star", "--dust", "--brand", "--brand-hi", "--nebula", "--mars"]
const TEXT_BACKGROUNDS = ["--sky-0", "--sky-1", "--sky-2"]

// Figure strokes that carry a state. --sky-line is left out on purpose: it is the graticule and the idle
// hairline, meant to recede, and idle is never told apart by hue alone (the figure dims it as well).
const STROKES = ["--brand", "--nebula", "--star", "--mars", "--dust"]
const STROKE_BACKGROUNDS = ["--sky-0", "--sky-1", "--sky-2"]

// Text drawn on a filled token: the primary button's label sits on --brand.
const FILLED = [{ text: "--sky-0", background: "--brand" }]

const themes = parseTokens(tokensCss)

describe("palette contrast", () => {
  it("parses both themes", () => {
    for (const theme of Object.values(themes)) {
      for (const token of [...TEXT, ...TEXT_BACKGROUNDS, ...STROKES, "--sky-line"]) expect(theme.has(token)).toBe(true)
    }
  })

  for (const themeName of ["night", "dawn"] as const) {
    const theme = themes[themeName]
    const color = (token: string) => theme.get(token) ?? "#000000"

    for (const text of TEXT) {
      for (const background of TEXT_BACKGROUNDS) {
        it(`${themeName}: text ${text} on ${background} >= ${BODY_TEXT_RATIO}`, () => {
          expect(contrastRatio(color(text), color(background))).toBeGreaterThanOrEqual(BODY_TEXT_RATIO)
        })
      }
    }

    for (const stroke of STROKES) {
      for (const background of STROKE_BACKGROUNDS) {
        it(`${themeName}: stroke ${stroke} on ${background} >= ${STROKE_RATIO}`, () => {
          expect(contrastRatio(color(stroke), color(background))).toBeGreaterThanOrEqual(STROKE_RATIO)
        })
      }
    }

    for (const { text, background } of FILLED) {
      it(`${themeName}: text ${text} on filled ${background} >= ${BODY_TEXT_RATIO}`, () => {
        expect(contrastRatio(color(text), color(background))).toBeGreaterThanOrEqual(BODY_TEXT_RATIO)
      })
    }
  }

  it("keeps the idle hairline quieter than any state stroke", () => {
    for (const theme of Object.values(themes)) {
      const idle = contrastRatio(theme.get("--sky-line")!, theme.get("--sky-0")!)
      for (const stroke of STROKES)
        expect(contrastRatio(theme.get(stroke)!, theme.get("--sky-0")!)).toBeGreaterThan(idle)
    }
  })
})
