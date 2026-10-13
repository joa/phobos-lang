// Star chart primitives. Each creates SVG elements with classes only; chart.css decides every color.
//
// Memory spaces: a space class sets --space, and the state classes (.is-on, .is-hot, .is-warn,
// .is-idle) read it. Global memory is --brand, shared memory --nebula, registers --star; cost is
// --mars and always dashed.

import { graticulePoints, registrationTicksPath } from "./layout"
import type { Polyline } from "./polyline"
import { svgEl, svgText, type Attributes } from "./svg"

export type Space = "global" | "shared" | "reg" | "cost"

export const SPACES: readonly Space[] = ["global", "shared", "reg", "cost"]

export const GRATICULE_PITCH_UNITS = 16
export const NODE_RADIUS_UNITS = 2

export function spaceClass(space: Space | undefined): string {
  return space ? `sp-${space}` : ""
}

function classes(...names: (string | false | null | undefined)[]): string {
  return names.filter(Boolean).join(" ")
}

export interface PlateOptions {
  widthUnits: number
  heightUnits: number
  // Uppercase mono label in the top left corner, such as "FIG / GEMM / SHARED".
  label?: string
  // Prints "schematic" in the top right corner: the drawing illustrates, it does not measure.
  schematic?: boolean
  pitchUnits?: number
}

// The plate every figure sits on: a dotted graticule, corner registration ticks and the plate label.
// The dots are one pattern; its id comes from idPrefix, which must be unique on the page.
export function plate(parent: Element, idPrefix: string, options: PlateOptions): SVGGElement {
  const { widthUnits, heightUnits, label, schematic = false, pitchUnits = GRATICULE_PITCH_UNITS } = options
  const group = svgEl("g", { class: "chart-plate" }, parent)
  const patternId = `${idPrefix}-graticule`
  const defs = svgEl("defs", {}, group)
  const [first] = graticulePoints(widthUnits, heightUnits, pitchUnits)
  const pattern = svgEl(
    "pattern",
    {
      id: patternId,
      width: pitchUnits,
      height: pitchUnits,
      patternUnits: "userSpaceOnUse",
      x: first ? first[0] - pitchUnits / 2 : 0,
      y: first ? first[1] - pitchUnits / 2 : 0,
    },
    defs
  )
  svgEl("circle", { class: "chart-graticule-dot", cx: pitchUnits / 2, cy: pitchUnits / 2, r: 0.75 }, pattern)
  svgEl("rect", { class: "chart-vignette", width: widthUnits, height: heightUnits }, group)
  svgEl("rect", { class: "chart-graticule", width: widthUnits, height: heightUnits, fill: `url(#${patternId})` }, group)
  svgEl("path", { class: "chart-ticks", d: registrationTicksPath(widthUnits, heightUnits) }, group)
  if (label) {
    svgText(label, { class: "chart-plate-text", x: 14, y: 20, "dominant-baseline": "middle" }, group)
  }
  if (schematic) {
    svgText(
      "schematic",
      {
        class: "chart-plate-note",
        x: widthUnits - 14,
        y: 20,
        "text-anchor": "end",
        "dominant-baseline": "middle",
      },
      group
    )
  }
  return group
}

export interface BoxOptions {
  x: number
  y: number
  width: number
  height: number
  space?: Space
  className?: string
}

export function box(parent: Element, options: BoxOptions): SVGRectElement {
  const { x, y, width, height, space, className } = options
  return svgEl("rect", { class: classes("chart-box", spaceClass(space), className), x, y, width, height }, parent)
}

export interface CellGridOptions {
  x: number
  y: number
  rows: number
  cols: number
  cellUnits: number
  gapUnits?: number
  space?: Space
  className?: string
}

// A tensor or tile drawn as cells, row-major: cells[row][col].
export function cellGrid(parent: Element, options: CellGridOptions): SVGRectElement[][] {
  const { x, y, rows, cols, cellUnits, gapUnits = 1, space, className } = options
  const group = svgEl("g", { class: classes("chart-cells", spaceClass(space), className) }, parent)
  const pitchUnits = cellUnits + gapUnits
  const cells: SVGRectElement[][] = []
  for (let row = 0; row < rows; row += 1) {
    const rowCells: SVGRectElement[] = []
    for (let col = 0; col < cols; col += 1) {
      rowCells.push(
        svgEl(
          "rect",
          {
            class: "chart-cell",
            x: x + col * pitchUnits,
            y: y + row * pitchUnits,
            width: cellUnits,
            height: cellUnits,
          },
          group
        )
      )
    }
    cells.push(rowCells)
  }
  return cells
}

export interface Wire {
  // Carries the space and state classes for the hairline and its nodes together.
  group: SVGGElement
  path: SVGPathElement
  nodes: SVGCircleElement[]
}

// A 1-unit hairline along a polyline, with a dot on each end and each vertex and no arrowheads.
export function wire(
  parent: Element,
  line: Polyline,
  { space, className }: { space?: Space; className?: string } = {}
): Wire {
  const group = svgEl("g", { class: classes("chart-wire-group", spaceClass(space), className) }, parent)
  const path = svgEl("path", { class: "chart-wire", d: line.toPathData() }, group)
  const nodes = line.points.map((point, index) =>
    node(group, point.x, point.y, {
      className: index === 0 || index === line.points.length - 1 ? "is-end" : "is-vertex",
    })
  )
  return { group, path, nodes }
}

export function node(
  parent: Element,
  x: number,
  y: number,
  { space, className }: { space?: Space; className?: string } = {}
): SVGCircleElement {
  return svgEl(
    "circle",
    { class: classes("chart-node", spaceClass(space), className), cx: x, cy: y, r: NODE_RADIUS_UNITS },
    parent
  )
}

export interface LabelOptions extends Attributes {
  x: number
  y: number
  anchor?: "start" | "middle" | "end"
  baseline?: "auto" | "middle" | "hanging" | "alphabetic"
  space?: Space
  className?: string
}

// A mono label, sentence case, at least 11 units. Plate labels are the only uppercase text.
export function label(parent: Element, content: string | string[], options: LabelOptions): SVGTextElement {
  const { x, y, anchor = "start", baseline = "middle", space, className, ...rest } = options
  return svgText(
    content,
    {
      ...rest,
      class: classes("chart-label", spaceClass(space), className),
      x,
      y,
      "text-anchor": anchor,
      "dominant-baseline": baseline,
    },
    parent
  )
}

// The soft glow behind whatever is active: one circle filled with its space's pre-blurred radial
// gradient (see ensureGlowDefs), instead of an SVG filter per element.
export function glow(
  parent: Element,
  x: number,
  y: number,
  radiusUnits: number,
  space: Space = "global"
): SVGCircleElement {
  return svgEl("circle", { class: classes("chart-glow", spaceClass(space)), cx: x, cy: y, r: radiusUnits }, parent)
}

export const GLOW_DEFS_ID = "chart-glow-defs"

// The glow gradients, one per space, defined once per document. A gradient's stop-color cannot read a
// property set where it is used, so each space gets its own gradient and CSS picks it by class.
export function glowDefsMarkup(): string {
  const gradients = SPACES.map(
    space =>
      `<radialGradient id="chart-glow-${space}"><stop offset="0" class="chart-glow-stop sp-${space}" stop-opacity="0.42"/>` +
      `<stop offset="0.45" class="chart-glow-stop sp-${space}" stop-opacity="0.12"/>` +
      `<stop offset="1" class="chart-glow-stop sp-${space}" stop-opacity="0"/></radialGradient>`
  ).join("")
  const vignette =
    `<radialGradient id="chart-plate-vignette" cx="0.5" cy="0.45" r="0.6">` +
    `<stop offset="0" class="chart-vignette-stop" stop-opacity="0.07"/>` +
    `<stop offset="1" class="chart-vignette-stop" stop-opacity="0"/></radialGradient>`
  return `<svg id="${GLOW_DEFS_ID}" class="chart-defs" aria-hidden="true" width="0" height="0"><defs>${gradients}${vignette}</defs></svg>`
}

export function ensureGlowDefs(): void {
  if (typeof document === "undefined" || document.getElementById(GLOW_DEFS_ID)) return
  document.body.insertAdjacentHTML("afterbegin", glowDefsMarkup())
}
