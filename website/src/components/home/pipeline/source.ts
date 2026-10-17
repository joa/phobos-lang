// The source station: the kernel's accumulator, its k loop and its epilogue, the real lines of
// examples/gemm_fp32.ph with their line numbers, the loop marked. Under every glyph sits one point of
// the cloud, so when the scene moves on the text dissolves into the points that rise into the tree.

import type { SourceView } from "../pipeline-data"
import { structure } from "./data"
import { CHAR_EM, FRAMES, hidden, PARTICLES, round, target, text, type Layout, type Shape, type StationGeometry, type Target, type Tone } from "./types"

/** The lines shown: the accumulator, the loop, and the store a nonzero beta takes. */
export const SOURCE_LINES = [13, 15, 16, 17, 18, 19, 25]

const KEYWORDS = new Set(["var", "let", "for", "in", "kernel"])
const CALLS = new Set(["dot", "range"])

export interface Token {
  text: string
  tone: Tone
}

export function tokenize(line: string): Token[] {
  const tokens: Token[] = []
  for (const match of line.matchAll(/(\s+)|([A-Za-z_][A-Za-z_0-9]*)|(\d+(?:\.\d+)?)|(:\+|\+=|[^\sA-Za-z_0-9])/g)) {
    const [whole, space, word, number] = match
    let tone: Tone = "dust"
    if (space) tone = "line"
    else if (word && KEYWORDS.has(word)) tone = "global"
    else if (word && CALLS.has(word)) tone = "shared"
    else if (word && /^[A-Z_]+$/.test(word)) tone = "hi"
    else if (word) tone = "reg"
    else if (number) tone = "hi"
    tokens.push({ text: whole, tone })
  }
  return tokens
}

interface Row {
  n: number | null
  text: string
  loop: boolean
}

function rows(view: SourceView, wide: boolean): Row[] {
  const out: Row[] = []
  for (const n of SOURCE_LINES) {
    const line = view.lines.find(entry => entry.n === n)
    if (!line) continue
    const loop = line.role.startsWith("loop")
    // The loop keeps its indentation; the store, shown without its `if`, sits flush.
    const textLine = loop ? line.text.replace(/^ {2}/, "") : line.text.trimStart()
    const limit = 48
    if (!wide && textLine.length > limit) {
      const cut = textLine.lastIndexOf(" = ", limit)
      out.push({ n, text: textLine.slice(0, cut + 2), loop })
      out.push({ n: null, text: "    " + textLine.slice(cut + 3), loop })
    } else {
      out.push({ n, text: textLine, loop })
    }
  }
  return out
}

export function sourceStation(layout: Layout): StationGeometry {
  const frame = FRAMES[layout]
  const wide = layout === "wide"
  const fontUnits = wide ? 17 : 12.5
  const charUnits = fontUnits * CHAR_EM
  const pitch = wide ? 40 : 27
  const list = rows(structure.source, wide)
  const codeX = wide ? 230 : 18
  const numberX = wide ? 196 : 0
  const blockHeight = pitch * (list.length - 1)
  const top = (wide ? 312 : 330) - blockHeight / 2

  const shapes: Shape[] = []
  const targets: Target[] = []
  shapes.push(text(wide ? codeX : codeX, top - (wide ? 60 : 48), structure.source.path, "pp-label pp-in", { delayMs: 80 }))

  // The loop's band, behind its lines.
  const loopRows = list.map((row, index) => (row.loop ? index : -1)).filter(index => index >= 0)
  if (loopRows.length > 0) {
    const first = loopRows[0]
    const last = loopRows[loopRows.length - 1]
    const y0 = top + first * pitch - fontUnits * 0.95
    const y1 = top + last * pitch + fontUnits * 0.5
    const x0 = wide ? codeX - 16 : 8
    shapes.push({
      tag: "path",
      attrs: { d: `M${x0} ${round(y0)}V${round(y1)}`, pathLength: 1, class: "pp-wire pp-draw t-global is-strong" },
      delayMs: 320,
    })
    const longest = Math.max(...loopRows.map(index => list[index].text.length))
    const labelX = wide ? codeX + longest * charUnits + 28 : frame.width - 10
    shapes.push(text(labelX, top + first * pitch, "the k loop", "pp-label t-global pp-in", { anchor: wide ? "start" : "end", delayMs: 420 }))
  }

  list.forEach((row, index) => {
    const y = top + index * pitch
    const delayMs = 120 + index * 90
    if (wide && row.n !== null) shapes.push(text(numberX, y, String(row.n), "pp-label is-faint pp-in", { anchor: "end", delayMs }))
    const indent = row.text.match(/^ */)?.[0].length ?? 0
    const tokens = tokenize(row.text.slice(indent))
    shapes.push({
      tag: "text",
      attrs: { x: round(codeX + indent * charUnits), y: round(y), class: `pp-code pp-in${wide ? "" : " is-small"}` },
      spans: tokens.map(token => ({ text: token.text, cls: `t-${token.tone}` })),
      delayMs,
    })
    let column = indent
    for (const token of tokens) {
      for (const char of token.text) {
        if (char !== " ") {
          const x = codeX + (column + 0.5) * charUnits
          targets.push(target(x, y + (wide ? 8 : 6), token.tone === "line" ? "dust" : token.tone, column % 4 === 0 ? 1 : 0, 0, 0.6))
        }
        column += 1
      }
    }
  })

  return { frame, shapes, targets: fill(targets, frame.width, frame.height), streams: [] }
}

/** Pads a station's targets to the cloud's size with hidden points spread over the frame. */
export function fill(targets: Target[], width: number, height: number): Target[] {
  const out = targets.slice(0, PARTICLES)
  for (let index = out.length; index < PARTICLES; index += 1) {
    const golden = index * 0.618034
    out.push(hidden((golden % 1) * width, ((index * 0.381966) % 1) * height))
  }
  return out
}
