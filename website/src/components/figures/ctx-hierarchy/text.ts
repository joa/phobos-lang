// Text measuring for SVG labels in Noto Sans Mono, whose advance is 0.6 em. Pure.

export const MONO_ADVANCE_EM = 0.6

export function textWidthUnits(text: string, fontUnits = 12): number {
  return text.length * fontUnits * MONO_ADVANCE_EM
}

// Breaks text at spaces into lines of at most maxChars characters. A word longer than a line keeps
// a line of its own.
export function wrapText(text: string, maxChars: number): string[] {
  const lines: string[] = []
  let line = ""
  for (const word of text.split(/\s+/).filter(Boolean)) {
    if (line === "") line = word
    else if (line.length + 1 + word.length <= maxChars) line += ` ${word}`
    else {
      lines.push(line)
      line = word
    }
  }
  if (line !== "") lines.push(line)
  return lines
}
