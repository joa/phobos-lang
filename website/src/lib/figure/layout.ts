// Layout rules every figure shares. Pure.

export type Layout = "wide" | "narrow"

// Below this container width a figure draws its narrow layout. At 600 px an 11-unit label in the
// 720-unit wide viewBox still renders at about 9 px.
export const NARROW_BELOW_PX = 600

export const VIEW_WIDTH_UNITS: Record<Layout, number> = { wide: 720, narrow: 400 }

// Labels never render below this many viewBox units.
export const MIN_LABEL_UNITS = 11

export function layoutFor(widthPx: number): Layout {
  return widthPx < NARROW_BELOW_PX ? "narrow" : "wide"
}

// The rendered size in CSS px of one viewBox unit at a container width.
export function pxPerUnit(widthPx: number, layout: Layout = layoutFor(widthPx)): number {
  return widthPx / VIEW_WIDTH_UNITS[layout]
}

// Dot positions of a graticule with the given pitch, centered so the margins on both sides match.
export function graticulePoints(widthUnits: number, heightUnits: number, pitchUnits: number): [number, number][] {
  const points: [number, number][] = []
  const insetXUnits = ((widthUnits % pitchUnits) + pitchUnits) / 2
  const insetYUnits = ((heightUnits % pitchUnits) + pitchUnits) / 2
  for (let y = insetYUnits; y < heightUnits; y += pitchUnits) {
    for (let x = insetXUnits; x < widthUnits; x += pitchUnits) points.push([x, y])
  }
  return points
}

// The four L-shaped registration ticks at a plate's corners, as SVG path data.
export function registrationTicksPath(widthUnits: number, heightUnits: number, armUnits = 8): string {
  const w = widthUnits
  const h = heightUnits
  const a = armUnits
  return [`M0 ${a}V0H${a}`, `M${w - a} 0H${w}V${a}`, `M${w} ${h - a}V${h}H${w - a}`, `M${a} ${h}H0V${h - a}`].join("")
}
