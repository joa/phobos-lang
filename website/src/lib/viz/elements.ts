// Creates every element of the visualizer's drawing once per layout and returns references to what
// the steps change. Classes only: chart.css and viz.css decide every color.

import { glow, plate } from "../figure/draw"
import { sweepWire } from "../figure/motion"
import { StarPacket } from "../figure/packet"
import { Polyline } from "../figure/polyline"
import { svgEl, svgText } from "../figure/svg"
import { arcPoints, type Rect, type TextAt, type VizGeometry } from "./geometry"
import type { KernelModel } from "./model"

export const TILE_PACKETS = 12
export const ROUTE_WIRES = 4
export const SLOT_POOL = 32
const WRITTEN_POOL = 6
const CURRENT_POOL = 3

export interface TensorElements {
  name: string
  written: SVGRectElement[]
  current: SVGRectElement[]
  others: SVGRectElement[]
  masked: SVGTextElement
}

export interface SlotElements {
  group: SVGGElement
  outline: SVGRectElement
  used: SVGRectElement
}

export interface ArcElements {
  opId: number
  /** Per stage, one path per lane pair, with the lanes it joins. */
  stages: { path: SVGPathElement; line: Polyline; reverse: Polyline }[][]
}

export interface VizElements {
  tileGroup: SVGGElement
  laneGroup: SVGGElement
  ctaNodes: SVGCircleElement[]
  ctaGlow: SVGCircleElement
  ctaText: SVGTextElement
  tensors: TensorElements[]
  buffers: Map<number, { rect: SVGRectElement; text: SVGTextElement | null }>
  barrierTicks: { group: SVGGElement; index: number }[]
  programCursor: SVGCircleElement
  threadDots: SVGCircleElement[]
  threadText: SVGTextElement
  tileSlots: SlotElements[]
  tileSlotsText: SVGTextElement
  routes: { path: SVGPathElement; sweep: SVGPathElement }[]
  packets: StarPacket[]
  laneNodes: SVGCircleElement[]
  laneWires: SVGPathElement[]
  laneSlots: SlotElements[]
  laneSlotsText: SVGTextElement
  lanePackets: StarPacket[]
  arcs: ArcElements[]
  stageText: SVGTextElement
  idleText: SVGTextElement
  kindNames: SVGTextElement[]
  kindLabels: SVGTextElement[]
  entrance: Element[]
}

export function buildElements(root: SVGGElement, id: string, model: KernelModel, geometry: VizGeometry): VizElements {
  const kernel = model.trace.kernel.toUpperCase()
  plate(root, id, {
    widthUnits: geometry.widthUnits,
    heightUnits: geometry.heightUnits,
    label: `FIG / ${kernel} / ${model.chip.toUpperCase()}`,
  })
  const entrance: Element[] = []
  entrance.push(text(root, geometry.note, "chart-plate-note"))

  const tileGroup = svgEl("g", { class: "viz-level" }, root)
  const laneGroup = svgEl("g", { class: "viz-level viz-off" }, root)
  const tile = geometry.tile
  for (const band of tile.bandLabels) entrance.push(text(tileGroup, band, "chart-plate-text"))

  // Grid: the CTAs as a constellation.
  const gridGroup = svgEl("g", { class: "sp-global" }, tileGroup)
  for (const [from, to] of tile.grid.links) {
    const a = tile.grid.nodes[from]
    const b = tile.grid.nodes[to]
    svgEl("path", { class: "chart-wire", d: `M${a.x} ${a.y}L${b.x} ${b.y}` }, gridGroup)
  }
  const first = tile.grid.nodes[0]
  const ctaGlow = glow(gridGroup, first.x, first.y, 14, "global")
  ctaGlow.classList.add("is-on", "is-breathing", "viz-follow")
  const ctaNodes = tile.grid.nodes.map((node, index) =>
    svgEl(
      "circle",
      {
        class: "chart-node viz-cta",
        cx: node.x,
        cy: node.y,
        r: 3.2,
        "data-group": "cta",
        "data-value": String(index),
      },
      gridGroup
    )
  )
  const area = tile.grid.area
  const ctaText = text(gridGroup, { x: area.x, y: area.y + area.height + 12, text: "" }, "chart-label is-small")

  // Global tensors as cell grids.
  const tensors = tile.tensors.map((place, index) => {
    const group = svgEl("g", { class: "sp-global" }, tileGroup)
    const patternId = `${id}-cells-${index}`
    const pitchX = place.cellCols * place.sx
    const pitchY = place.cellRows * place.sy
    const pattern = svgEl(
      "pattern",
      {
        id: patternId,
        x: place.rect.x,
        y: place.rect.y,
        width: pitchX,
        height: pitchY,
        patternUnits: "userSpaceOnUse",
      },
      svgEl("defs", {}, group)
    )
    svgEl("rect", { class: "chart-cell", width: Math.max(0.6, pitchX - 1), height: Math.max(0.6, pitchY - 1) }, pattern)
    svgEl("rect", { class: "viz-cells", ...rectAttrs(place.rect), fill: `url(#${patternId})` }, group)
    entrance.push(svgEl("rect", { class: "chart-box", ...rectAttrs(place.rect) }, group))
    const others = model.ctas.map(() => svgEl("rect", { class: "chart-box is-outline viz-other is-hidden" }, group))
    const written = range(WRITTEN_POOL).map(() =>
      svgEl("rect", { class: "chart-box is-on viz-written is-hidden" }, group)
    )
    const current = range(CURRENT_POOL).map(() => svgEl("rect", { class: "chart-box is-hot is-hidden" }, group))
    entrance.push(text(group, place.label, "chart-label is-small"))
    entrance.push(text(group, place.note, "chart-label is-small"))
    const masked = text(
      group,
      { x: place.rect.x + place.rect.width, y: place.note.y, text: "", anchor: "end" },
      "chart-label is-small is-warn sp-cost"
    )
    return { name: place.name, written, current, others, masked }
  })

  // Shared memory: a byte ruler with each buffer at its planned offset.
  const shared = tile.shared
  const sharedGroup = svgEl("g", { class: "sp-shared" }, tileGroup)
  entrance.push(svgEl("path", { class: "chart-wire", d: `M${shared.x0} ${shared.axisY}H${shared.x1}` }, sharedGroup))
  svgEl(
    "path",
    { class: "chart-ticks", d: `M${shared.x0} ${shared.axisY - 3}v6M${shared.x1} ${shared.axisY - 3}v6` },
    sharedGroup
  )
  text(sharedGroup, shared.zero, "chart-label is-small")
  text(sharedGroup, shared.peak, "chart-label is-small")
  if (shared.empty) text(sharedGroup, shared.empty, "chart-label is-small")
  const buffers = new Map<number, { rect: SVGRectElement; text: SVGTextElement | null }>()
  for (const placed of shared.buffers) {
    const buffer = model.buffer(placed.id)
    const rect = svgEl("rect", { class: "chart-box", ...rectAttrs(placed.rect) }, sharedGroup)
    const name = buffer?.name ?? ""
    const fits = placed.rect.width >= name.length * 7.6 + 6
    const nameText = fits
      ? text(
          sharedGroup,
          { x: placed.rect.x + 3, y: placed.rect.y + placed.rect.height / 2 + 0.5, text: name },
          "chart-label is-small viz-inside"
        )
      : null
    buffers.set(placed.id, { rect, text: nameText })
  }

  // Barriers in program order.
  const program = tile.program
  const programGroup = svgEl("g", {}, tileGroup)
  entrance.push(svgEl("path", { class: "chart-wire", d: `M${program.x0} ${program.y}H${program.x1}` }, programGroup))
  const barrierTicks = program.barriers.map(barrier => {
    const group = svgEl("g", { class: barrier.kept ? "sp-reg is-on viz-kept" : "viz-elided" }, programGroup)
    svgEl("path", { class: "chart-wire", d: `M${barrier.x} ${program.y - 9}V${program.y + 9}` }, group)
    svgEl("circle", { class: "chart-node", cx: barrier.x, cy: program.y - 9, r: 1.8 }, group)
    return { group, index: barrier.index }
  })
  const programCursor = svgEl(
    "circle",
    { class: "chart-node is-on sp-global viz-follow", cx: 0, cy: program.y, r: 3 },
    programGroup
  )
  text(programGroup, program.note, "chart-label is-small")

  // Threads: the CTA's warps as rows of lanes, and the lines warp 0 touches.
  const threads = tile.threads
  const threadsGroup = svgEl("g", { class: "sp-reg" }, tileGroup)
  const threadDots = threads.dots.map(dot =>
    svgEl("circle", { class: "chart-node", cx: dot.x, cy: dot.y, r: 1.5 }, threadsGroup)
  )
  const threadText = text(threadsGroup, threads.text, "chart-label is-small")
  const tileSlots = slotPool(tileGroup)
  const tileSlotsText = text(tileGroup, threads.slotsText, "chart-label is-small")

  // Routes and packets above both levels' tile parts.
  const routeGroup = svgEl("g", { class: "viz-routes" }, tileGroup)
  const routes = range(ROUTE_WIRES).map(() => {
    const path = svgEl("path", { class: "chart-wire is-hidden", d: "M0 0" }, routeGroup)
    return { path, sweep: sweepWire(routeGroup, "M0 0", "global") }
  })
  const packets = range(TILE_PACKETS).map(() => new StarPacket(routeGroup, { space: "global" }))

  // The lane level.
  const lane = geometry.lane
  for (const band of lane.bandLabels) text(laneGroup, band, "chart-plate-text")
  const foldedGroup = svgEl("g", { class: "sp-reg is-idle" }, laneGroup)
  for (const row of lane.folded) {
    for (const x of row.xs) svgEl("circle", { class: "chart-node", cx: x, cy: row.y, r: 1.1 }, foldedGroup)
  }
  if (lane.foldedNote) text(laneGroup, lane.foldedNote, "chart-label is-small")
  const laneWireGroup = svgEl("g", { class: "sp-global" }, laneGroup)
  const laneWires = lane.lanes.map(() => svgEl("path", { class: "chart-wire is-hidden", d: "M0 0" }, laneWireGroup))
  const laneSlots = slotPool(laneGroup)
  const laneSlotsText = text(laneGroup, lane.slotsText, "chart-label is-small")
  const arcGroup = svgEl("g", { class: "sp-reg" }, laneGroup)
  const arcs = buildArcs(arcGroup, model, lane.lanes, lane.arcMaxUnits)
  const laneNodeGroup = svgEl("g", { class: "sp-reg" }, laneGroup)
  svgEl(
    "path",
    {
      class: "chart-wire",
      d: `M${lane.lanes[0].x} ${lane.lanes[0].y}H${lane.lanes[lane.lanes.length - 1].x}`,
    },
    laneNodeGroup
  )
  const laneNodes = lane.lanes.map(node =>
    svgEl("circle", { class: "chart-node", cx: node.x, cy: node.y, r: 2.6 }, laneNodeGroup)
  )
  for (const laneLabel of lane.laneLabels) text(laneGroup, laneLabel, "chart-label is-small")
  const stageText = text(laneGroup, lane.stageText, "chart-label is-small")
  const idleText = text(laneGroup, lane.idleText, "chart-label is-small")
  const lanePackets = lane.lanes.map(() => new StarPacket(laneGroup, { space: "reg", trailGapUnits: 3, count: 6 }))

  const kindGroup = svgEl("g", { class: "sp-reg" }, laneGroup)
  const kindNames: SVGTextElement[] = []
  const kindLabels: SVGTextElement[] = []
  const warps = model.trace.launch.warps
  const lanes = model.trace.launch.lanes
  model.values.slice(0, lane.kinds.length).forEach((value, index) => {
    const place = lane.kinds[index]
    kindNames.push(text(kindGroup, place.name, "chart-label is-small"))
    kindLabels.push(text(kindGroup, place.kind, "chart-label is-small"))
    const dots = value.kind === "uniform" ? 1 : value.kind === "warp" ? warps : lanes
    const pitch = value.kind === "warp" ? place.pitchUnits * 3 : place.pitchUnits
    for (let dot = 0; dot < dots; dot += 1) {
      svgEl("circle", { class: "chart-node is-on", cx: place.dotsX + dot * pitch, cy: place.dotsY, r: 1.3 }, kindGroup)
    }
  })

  return {
    tileGroup,
    laneGroup,
    ctaNodes,
    ctaGlow,
    ctaText,
    tensors,
    buffers,
    barrierTicks,
    programCursor,
    threadDots,
    threadText,
    tileSlots,
    tileSlotsText,
    routes,
    packets,
    laneNodes,
    laneWires,
    laneSlots,
    laneSlotsText,
    lanePackets,
    arcs,
    stageText,
    idleText,
    kindNames,
    kindLabels,
    entrance,
  }
}

function buildArcs(
  parent: SVGGElement,
  model: KernelModel,
  lanes: { x: number; y: number }[],
  maxUnits: number
): ArcElements[] {
  const seen = new Set<number>()
  const arcs: ArcElements[] = []
  for (const step of model.trace.steps) {
    const event = step.lanes
    if (!event || seen.has(step.op)) continue
    const stages = event.stages ?? (event.partners ? [event.partners] : null)
    if (!stages) continue
    seen.add(step.op)
    arcs.push({
      opId: step.op,
      stages: stages.map(partners => {
        const pairs: { path: SVGPathElement; line: Polyline; reverse: Polyline }[] = []
        partners.forEach((partner, laneIndex) => {
          if (partner === laneIndex || partner < 0 || partner >= lanes.length) return
          // An exchange draws once; a one-way shuffle draws every source.
          if (event.mode === "xor" && partner < laneIndex) return
          const points = arcPoints(lanes[partner], lanes[laneIndex], maxUnits)
          const line = new Polyline(points)
          const path = svgEl(
            "path",
            { class: "chart-wire viz-arc is-hidden", d: line.toPathData(), pathLength: 1 },
            parent
          )
          pairs.push({ path, line, reverse: new Polyline([...points].reverse()) })
        })
        return pairs
      }),
    })
  }
  return arcs
}

function slotPool(parent: SVGGElement): SlotElements[] {
  return range(SLOT_POOL).map(() => {
    const group = svgEl("g", { class: "sp-global is-hidden" }, parent)
    const outline = svgEl("rect", { class: "chart-box" }, group)
    const used = svgEl("rect", { class: "chart-box is-on" }, group)
    return { group, outline, used }
  })
}

export function text(parent: Element, at: TextAt, className: string): SVGTextElement {
  return svgText(
    at.text,
    { class: className, x: at.x, y: at.y, "text-anchor": at.anchor ?? "start", "dominant-baseline": "middle" },
    parent
  )
}

export function rectAttrs(rect: Rect): Record<string, number> {
  const round = (value: number) => Math.round(value * 100) / 100
  return { x: round(rect.x), y: round(rect.y), width: round(rect.width), height: round(rect.height) }
}

export function setRect(element: Element, rect: Rect): void {
  for (const [name, value] of Object.entries(rectAttrs(rect))) element.setAttribute(name, String(value))
}

function range(count: number): number[] {
  return Array.from({ length: count }, (_, index) => index)
}
