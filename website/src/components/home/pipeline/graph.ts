// The graph IR station: the real DAG of gemm_fp32 as the compiler prints it, at the layered layout the
// data carries. Wide flows left to right; narrow turns it to flow down and labels only the nodes that
// have room. Edges are S-curves operand to user, bowed around a node they would cross; the loop-carried
// value goes back over (or beside) the loop's box. Every edge is also a stream the cloud flows along.

import type { GraphNode } from "../pipeline-data"
import type { Point } from "../../../lib/figure/polyline"
import { mapRoute, routePath, sampleRoute, sCurve, type Route } from "./bezier"
import { structure } from "./data"
import {
  FRAMES,
  hidden,
  overlaps,
  PARTICLES,
  round,
  target,
  text,
  textRect,
  type Layout,
  type Rect,
  type Shape,
  type StationGeometry,
  type Stream,
  type Target,
  type Tone,
} from "./types"

export const LABEL_UNITS = 12.5
const LAYER_DELAY_MS = 140
const FIRST_DELAY_MS = 150
const STREAM_LAG_MS = 520
const BACK_LANE_UNITS = 30

interface Projection {
  /** Layout units (x along the flow, y across it) to frame units. */
  map(point: Point): Point
  /** Frame units per layout unit along the flow. */
  alongScale: number
  wide: boolean
}

function projection(layout: Layout): Projection {
  if (layout === "wide") {
    const scale = 0.8
    return { map: p => ({ x: 56 + p.x * scale, y: 92 + (p.y + 80) * scale }), alongScale: scale, wide: true }
  }
  return {
    map: p => ({ x: 14 + (p.y / 56 + 1.64) * 38, y: 104 + (p.x / 170) * 60 }),
    alongScale: 60 / 170,
    wide: false,
  }
}

export function nodeTone(node: GraphNode): Tone {
  const type = node.type ?? ""
  if (node.kind === "guard") return "cost"
  if (node.kind === "math") return "dust"
  if (type.startsWith("gemm") || node.kind === "for" || node.kind === "yield") return "reg"
  if (type === "index") return "dust"
  if (type === "f32") return "reg"
  return "global"
}

/** The label a node gets: the IR's own, shortened only where it would not fit a column. */
export function nodeLabel(node: GraphNode, wide: boolean): string | null {
  if (!wide) {
    if (node.kind === "param") return node.label
    if (node.kind === "for") return "for kt"
    if (node.op === "gemm_dot") return "dot"
    if (node.op === "gemm_store") return "store"
    if (node.kind === "for_result") return "acc"
    if (node.op === "slice") return node.label.replace(/^slice (\w).*/, "slice $1")
    return null
  }
  if (node.kind === "guard") return node.label.replace(/^trap if /, "")
  if (node.kind === "for") return "for kt"
  if (node.kind === "for_result") return "for result"
  if (node.kind === "yield") return "yield"
  return node.label
}

function glyphRadius(node: GraphNode): number {
  return node.kind === "param" ? 6 : node.kind === "for" || node.kind === "for_result" ? 5.5 : 4.5
}

function glyph(node: GraphNode, at: Point, delayMs: number): Shape {
  const tone = nodeTone(node)
  const r = glyphRadius(node)
  const cls = `pp-glyph pp-in t-${tone}`
  const x = round(at.x)
  const y = round(at.y)
  switch (node.kind) {
    case "param":
      return { tag: "circle", attrs: { cx: x, cy: y, r, class: `${cls} is-ring` }, delayMs }
    case "for":
    case "for_result":
      return { tag: "rect", attrs: { x: x - r, y: y - r, width: r * 2, height: r * 2, class: `${cls} is-ring` }, delayMs }
    case "math":
      return { tag: "path", attrs: { d: `M${x} ${y - r}L${x + r} ${y}L${x} ${y + r}L${x - r} ${y}Z`, class: cls }, delayMs }
    case "guard":
      return {
        tag: "path",
        attrs: { d: `M${x} ${round(y - r * 1.1)}L${round(x + r)} ${round(y + r * 0.8)}L${round(x - r)} ${round(y + r * 0.8)}Z`, class: cls },
        delayMs,
      }
    case "yield":
      return { tag: "rect", attrs: { x: x - 3.5, y: y - 3.5, width: 7, height: 7, class: cls }, delayMs }
    case "block_arg":
      return { tag: "circle", attrs: { cx: x, cy: y, r: 3.5, class: cls }, delayMs }
    default:
      return { tag: "circle", attrs: { cx: x, cy: y, r, class: cls }, delayMs }
  }
}

export interface GraphLayout {
  geometry: StationGeometry
  labels: Shape[]
  glyphBoxes: Rect[]
  edgeRoutes: { from: string; to: string; kind: "data" | "back"; route: Route }[]
}

export function graphStation(layout: Layout): GraphLayout {
  const frame = FRAMES[layout]
  const view = structure.graph
  const proj = projection(layout)
  const byId = new Map(view.nodes.map(node => [node.id, node]))
  const at = (node: GraphNode) => proj.map({ x: node.x, y: node.y })
  const layerDelay = (layer: number) => FIRST_DELAY_MS + layer * LAYER_DELAY_MS
  const region = view.regions[0]

  const shapes: Shape[] = []
  const labels: Shape[] = []
  const glyphBoxes: Rect[] = []

  // The loop's box, with its header above it.
  const box = region.box
  const corner = proj.map({ x: box.x, y: box.y })
  const across = proj.map({ x: box.x + box.width, y: box.y + box.height })
  const boxRect = {
    x: Math.min(corner.x, across.x),
    y: Math.min(corner.y, across.y),
    width: Math.abs(across.x - corner.x),
    height: Math.abs(across.y - corner.y),
  }
  const boxDelay = layerDelay(box.firstLayer) - 60
  shapes.push({
    tag: "rect",
    attrs: { ...roundRect(boxRect), class: "pp-region pp-in" },
    delayMs: boxDelay,
  })
  const trips = structure.passes.find(pass => pass.variant === null)?.loops[0]?.trips
  if (proj.wide) {
    const right = boxRect.x + boxRect.width - 10
    labels.push(text(right, boxRect.y + 18, region.label, "pp-label t-reg is-strong pp-in", { anchor: "end", delayMs: boxDelay }))
    if (trips) labels.push(text(right, boxRect.y + 36, `${trips} trips`, "pp-label pp-in", { anchor: "end", delayMs: boxDelay }))
  } else {
    labels.push(text(boxRect.x + boxRect.width + 8, boxRect.y + 14, "for kt", "pp-label t-reg pp-in", { delayMs: boxDelay }))
    if (trips) labels.push(text(boxRect.x + boxRect.width + 8, boxRect.y + 30, `${trips} trips`, "pp-label pp-in", { delayMs: boxDelay }))
  }

  // Edges, then the nodes over them.
  const edgeRoutes: GraphLayout["edgeRoutes"] = []
  const streams: Stream[] = []
  const pad = (node: GraphNode) => (glyphRadius(node) + 3) / proj.alongScale
  for (const edge of view.edges) {
    const from = byId.get(edge.from)
    const to = byId.get(edge.to)
    if (!from || !to) continue
    const route =
      edge.kind === "back"
        ? backRoute(from, to, box.y - BACK_LANE_UNITS, box.x - 24, pad(from), pad(to))
        : forwardRoute(from, to, view.nodes, pad(from), pad(to))
    const mapped = mapRoute(route, proj.map)
    const delayMs = edge.kind === "back" ? layerDelay(region.box.lastLayer) + 420 : layerDelay(from.layer) + 110
    const tone = edge.kind === "back" ? "reg" : nodeTone(from)
    shapes.push({
      tag: "path",
      attrs: { d: routePath(mapped), pathLength: 1, class: `pp-wire pp-draw t-${tone}${edge.kind === "back" ? " is-back" : ""}` },
      delayMs,
    })
    edgeRoutes.push({ from: edge.from, to: edge.to, kind: edge.kind, route: mapped })
    streams.push({ points: sampleRoute(mapped), tone, startMs: delayMs + STREAM_LAG_MS })
  }

  if (proj.wide) {
    const back = edgeRoutes.find(edge => edge.kind === "back")
    if (back) {
      const top = proj.map({ x: box.x + box.width * 0.5, y: box.y - BACK_LANE_UNITS })
      labels.push(text(top.x, top.y - 14, "carried acc, back to the next trip", "pp-label t-reg pp-in", { anchor: "middle", delayMs: layerDelay(6) + 700 }))
    }
  }

  for (const node of view.nodes) {
    const point = at(node)
    const delayMs = layerDelay(node.layer)
    shapes.push(glyph(node, point, delayMs))
    const r = glyphRadius(node) + 2
    glyphBoxes.push({ x: point.x - r, y: point.y - r, width: r * 2, height: r * 2 })
  }

  // Labels: wide, centered under each node; narrow, right of the node where there is room.
  for (const node of view.nodes) {
    const content = nodeLabel(node, proj.wide)
    if (!content) continue
    const point = at(node)
    const tone = nodeTone(node)
    const cls = `pp-label pp-in${tone === "cost" ? " t-cost" : node.kind === "param" ? " is-strong" : ""}`
    const delayMs = layerDelay(node.layer) + 60
    if (proj.wide) {
      labels.push(text(point.x, point.y + 19, content, cls, { anchor: "middle", delayMs }))
      continue
    }
    const candidates = [
      text(point.x + 9, point.y + 4.5, content, cls, { delayMs }),
      text(point.x - 9, point.y + 4.5, content, cls, { anchor: "end", delayMs }),
      text(point.x, point.y + 19, content, cls, { anchor: "middle", delayMs }),
    ]
    const own = { x: point.x - 8, y: point.y - 8, width: 16, height: 16 }
    const fits = candidates.find(candidate => {
      const rect = textRect(candidate, LABEL_UNITS)
      if (rect.x < 4 || rect.x + rect.width > frame.width - 4) return false
      if (glyphBoxes.some(glyphBox => !sameBox(glyphBox, own) && overlaps(rect, glyphBox, 1))) return false
      if (overlaps(rect, boxRect) && !contains(boxRect, rect)) return false
      return !labels.some(other => overlaps(rect, textRect(other, LABEL_UNITS), 2))
    })
    if (fits) labels.push(fits)
  }

  if (proj.wide) shapes.push(...legend(frame.height))

  return {
    geometry: { frame, shapes: [...shapes, ...labels], targets: graphTargets(view.nodes.map(at), view.nodes, streams), streams },
    labels,
    glyphBoxes,
    edgeRoutes,
  }
}

function forwardRoute(from: GraphNode, to: GraphNode, nodes: GraphNode[], padFrom: number, padTo: number): Route {
  const a = { x: from.x + padFrom, y: from.y }
  const b = { x: to.x - padTo, y: to.y }
  // Bow around a node the straight run would pass through.
  let bow = 0
  for (const node of nodes) {
    if (node === from || node === to || node.x <= from.x + 1 || node.x >= to.x - 1) continue
    const t = (node.x - a.x) / (b.x - a.x)
    const lineY = a.y + (b.y - a.y) * t
    if (Math.abs(lineY - node.y) < 22) bow = node.y >= lineY ? -36 : 36
  }
  return sCurve(a, b, bow)
}

function backRoute(from: GraphNode, to: GraphNode, laneY: number, laneX: number, padFrom: number, padTo: number): Route {
  const a = { x: from.x + padFrom, y: from.y }
  const b = { x: to.x - padTo, y: to.y }
  return {
    start: a,
    segments: [
      [{ x: a.x + 44, y: a.y }, { x: a.x + 44, y: laneY }, { x: a.x, y: laneY }],
      [{ x: a.x - 120, y: laneY - 16 }, { x: laneX + 160, y: laneY - 16 }, { x: laneX + 40, y: laneY }],
      [{ x: laneX - 10, y: laneY }, { x: laneX - 10, y: b.y }, b],
    ],
  }
}

/** Two points rest on each node; the rest are spread over the streams by length. */
function graphTargets(points: Point[], nodes: GraphNode[], streams: Stream[]): Target[] {
  const targets: Target[] = []
  points.forEach((point, index) => {
    const tone = nodeTone(nodes[index])
    targets.push(target(point.x, point.y, tone, 2, 0, 0.9), target(point.x + 1.5, point.y - 1, tone, 1, 0, 0.6))
  })
  const lengths = streams.map(stream => polylineLength(stream.points))
  const total = lengths.reduce((sum, length) => sum + length, 0)
  const left = PARTICLES - targets.length
  const counts = lengths.map(length => Math.max(3, Math.floor((length / total) * left)))
  let spare = left - counts.reduce((sum, count) => sum + count, 0)
  for (let index = 0; spare > 0; index = (index + 1) % counts.length, spare -= 1) counts[index] += 1
  for (let index = 0; spare < 0; index = (index + 1) % counts.length) {
    if (counts[index] > 3) {
      counts[index] -= 1
      spare += 1
    }
  }
  streams.forEach((stream, streamIndex) => {
    for (let index = 0; index < counts[streamIndex]; index += 1) {
      const phase = (index + 0.5) / counts[streamIndex]
      const start = stream.points[0]
      targets.push({
        ...target(start.x, start.y, stream.tone, (index % 3) as 0 | 1 | 2),
        alpha: 0,
        stream: streamIndex,
        phase: round(phase * 1000) / 1000,
      })
    }
  })
  while (targets.length < PARTICLES) targets.push(hidden(600, 300))
  return targets.slice(0, PARTICLES)
}

function polylineLength(points: Point[]): number {
  let length = 0
  for (let index = 1; index < points.length; index += 1) {
    length += Math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y)
  }
  return length
}

function legend(height: number): Shape[] {
  const y = height - 22
  const items: [string, Tone, string][] = [
    ["tensor, global", "global", "circle"],
    ["accumulator, registers", "reg", "circle"],
    ["index", "dust", "circle"],
    ["divisibility trap", "cost", "triangle"],
  ]
  const shapes: Shape[] = []
  let x = 40
  for (const [name, tone, kind] of items) {
    shapes.push(
      kind === "triangle"
        ? { tag: "path", attrs: { d: `M${x} ${y - 9}L${x + 5} ${y - 0.5}L${x - 5} ${y - 0.5}Z`, class: `pp-glyph pp-in t-${tone}` }, delayMs: 300 }
        : { tag: "circle", attrs: { cx: x, cy: y - 4.5, r: 4.5, class: `pp-glyph pp-in t-${tone}` }, delayMs: 300 }
    )
    shapes.push(text(x + 11, y, name, "pp-label pp-in", { delayMs: 300 }))
    x += 11 + name.length * LABEL_UNITS * 0.6 + 34
  }
  return shapes
}

function roundRect(rect: Rect): Record<string, number> {
  return { x: round(rect.x), y: round(rect.y), width: round(rect.width), height: round(rect.height) }
}

function sameBox(a: Rect, b: Rect): boolean {
  return Math.abs(a.x + a.width / 2 - (b.x + b.width / 2)) < 2 && Math.abs(a.y + a.height / 2 - (b.y + b.height / 2)) < 2
}

function contains(outer: Rect, inner: Rect): boolean {
  return inner.x >= outer.x && inner.y >= outer.y && inner.x + inner.width <= outer.x + outer.width && inner.y + inner.height <= outer.y + outer.height
}
