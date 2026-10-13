// Geometry helpers the inference figures share, pure so the tests can check layouts: rectangles, and the
// extent of a mono label, which is what lets a test prove that no two labels overlap.

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface TextAt {
  x: number
  y: number
  text: string
  anchor?: "start" | "middle" | "end"
  // 12 for .chart-label, 11 for .is-small and plate text.
  sizeUnits?: number
}

// Noto Sans Mono advances 0.6 em per glyph.
const MONO_ADVANCE_EM = 0.6

export function textWidthUnits(text: string, sizeUnits = 12): number {
  return text.length * MONO_ADVANCE_EM * sizeUnits
}

// The box a label covers, centered on its y the way dominant-baseline middle draws it.
export function textRect(at: TextAt): Rect {
  const size = at.sizeUnits ?? 12
  const width = textWidthUnits(at.text, size)
  const x = at.anchor === "middle" ? at.x - width / 2 : at.anchor === "end" ? at.x - width : at.x
  return { x, y: at.y - size * 0.6, width, height: size * 1.2 }
}

export function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

export function inside(inner: Rect, outer: Rect): boolean {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  )
}

export function boundsOf(rects: Rect[]): Rect {
  const minX = Math.min(...rects.map(rect => rect.x))
  const minY = Math.min(...rects.map(rect => rect.y))
  const maxX = Math.max(...rects.map(rect => rect.x + rect.width))
  const maxY = Math.max(...rects.map(rect => rect.y + rect.height))
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY }
}

export function center(rect: Rect): { x: number; y: number } {
  return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }
}

// Knuth's multiplicative hash, for picks that look random but repeat on every load.
export function knuthHash(value: number): number {
  return Math.imul(value + 1, 2654435761) >>> 0
}
