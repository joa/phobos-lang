// Geometry for the cluster figure: a tensor split into super-tiles, one per node, each node a grid
// of CTAs, and one CTA opened into its warps and lanes. Pure, so tests can check it in Node.

import type { Layout } from "../../../lib/figure/layout"
import { Polyline, type Point } from "../../../lib/figure/polyline"

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface TextAt {
  text: string
  x: number
  y: number
  small?: boolean
}

// Four super-tiles on four nodes, as the scheduler deals them: output super-tile i to node i % 4.
export const NODES = 4
// CTAs drawn per node: a schematic count, not one a launch is known to have.
export const CTAS_PER_NODE = 8
// One CTA at the default @launch(256): 8 warps of 32 lanes.
export const WARPS = 8
export const LANES = 32
// The node whose CTA the figure opens.
export const OPENED_NODE = NODES - 1
export const OPENED_CTA = CTAS_PER_NODE - 1

export interface ClusterTilesGeometry {
  widthUnits: number
  heightUnits: number
  // The cluster tier, drawn dashed: experimental.
  cluster: Rect
  // The output tensor's super-tiles, top to bottom, inside the cluster box.
  bands: Rect[]
  bandLabels: TextAt[]
  // One node per super-tile, also dashed, each with its grid of CTAs.
  nodes: Rect[]
  nodeLabels: TextAt[]
  ctaCells: { x: number; y: number; cellUnits: number; gapUnits: number }[]
  // Super-tile i to node i.
  links: Polyline[]
  // The opened CTA: warps as rows, lanes as cells.
  cta: Rect
  ctaLabel: TextAt
  lanes: { x: number; y: number; cellUnits: number; gapUnits: number }
  zoom: Polyline
  headers: TextAt[]
  notes: TextAt[]
}

const BAND_TOP = 76
const BAND_HEIGHT = 50
const BAND_GAP = 4
const CTA_CELL = 16
const CTA_GAP = 4

function bandRect(x: number, width: number, index: number): Rect {
  return { x, y: BAND_TOP + index * (BAND_HEIGHT + BAND_GAP), width, height: BAND_HEIGHT }
}

const centerY = (r: Rect) => r.y + r.height / 2

function rowWidth(count: number, cellUnits: number, gapUnits: number): number {
  return count * (cellUnits + gapUnits) - gapUnits
}

export function clusterTilesGeometry(layout: Layout): ClusterTilesGeometry {
  const wide = layout === "wide"
  const widthUnits = wide ? 720 : 400
  const cluster = wide ? { x: 24, y: 62, width: 176, height: 240 } : { x: 16, y: 62, width: 84, height: 240 }
  const bandX = wide ? 40 : 30
  const bandWidth = wide ? 56 : 48
  const bands = Array.from({ length: NODES }, (_, i) => bandRect(bandX, bandWidth, i))
  const bandLabels = bands.map((band, i) =>
    wide
      ? { text: `super-tile ${i}`, x: band.x + band.width + 8, y: centerY(band) - 12, small: true }
      : { text: String(i), x: band.x + band.width / 2, y: centerY(band), small: true }
  )

  const nodeX = wide ? 232 : 128
  const nodeWidth = wide ? 240 : 256
  const nodes = bands.map(band => ({ x: nodeX, y: band.y, width: nodeWidth, height: band.height }))
  const nodeLabels = nodes.map((node, i) => ({ text: `node ${i}`, x: node.x + 10, y: centerY(node) }))
  const cellsX = nodeX + nodeWidth - 10 - rowWidth(CTAS_PER_NODE, CTA_CELL, CTA_GAP)
  const ctaCells = nodes.map(node => ({
    x: cellsX,
    y: centerY(node) - CTA_CELL / 2,
    cellUnits: CTA_CELL,
    gapUnits: CTA_GAP,
  }))
  const links = bands.map(
    (band, i) =>
      new Polyline([
        { x: band.x + band.width, y: centerY(band) },
        { x: nodes[i].x, y: centerY(band) },
      ])
  )

  const opened = ctaCells[OPENED_NODE]
  const openedCell: Rect = {
    x: opened.x + OPENED_CTA * (CTA_CELL + CTA_GAP),
    y: opened.y,
    width: CTA_CELL,
    height: CTA_CELL,
  }

  if (wide) {
    const cta = { x: 508, y: 214, width: 188, height: 88 }
    const laneCell = 4
    const laneGap = 1
    const lanes = {
      x: cta.x + (cta.width - rowWidth(LANES, laneCell, laneGap)) / 2,
      y: cta.y + 34,
      cellUnits: laneCell,
      gapUnits: laneGap,
    }
    const zoomY = centerY(nodes[OPENED_NODE])
    return {
      widthUnits,
      heightUnits: 320,
      cluster,
      bands,
      bandLabels,
      nodes,
      nodeLabels,
      ctaCells,
      links,
      cta,
      ctaLabel: { text: "CTA: 8 warps x 32 lanes", x: cta.x + 10, y: cta.y + 16, small: true },
      lanes,
      zoom: new Polyline([
        { x: openedCell.x + openedCell.width, y: zoomY },
        { x: cta.x, y: zoomY },
      ]),
      headers: [
        { text: "cluster (experimental)", x: cluster.x, y: 46 },
        { text: "node: a grid of CTAs", x: nodeX, y: 46 },
      ],
      notes: [
        { text: "@cluster sizes a super-tile", x: 508, y: 84, small: true },
        { text: "@autotune sizes a CTA tile", x: 508, y: 104, small: true },
        { text: "dealt round robin", x: 508, y: 132, small: true },
        { text: "over the nodes", x: 508, y: 148, small: true },
      ],
    }
  }

  const cta = { x: 16, y: 330, width: 368, height: 120 }
  const laneCell = 8
  const laneGap = 2
  const lanes = {
    x: cta.x + (cta.width - rowWidth(LANES, laneCell, laneGap)) / 2,
    y: cta.y + 32,
    cellUnits: laneCell,
    gapUnits: laneGap,
  }
  const zoomX = openedCell.x + openedCell.width / 2
  return {
    widthUnits,
    heightUnits: 516,
    cluster,
    bands,
    bandLabels,
    nodes,
    nodeLabels,
    ctaCells,
    links,
    cta,
    ctaLabel: { text: "CTA: 8 warps x 32 lanes", x: cta.x + 10, y: cta.y + 16, small: true },
    lanes,
    zoom: new Polyline([
      { x: zoomX, y: openedCell.y + openedCell.height },
      { x: zoomX, y: cta.y },
    ]),
    headers: [
      { text: "cluster (experimental)", x: cluster.x, y: 46 },
      { text: "node: a grid of CTAs", x: 232, y: 46 },
    ],
    notes: [
      { text: "@cluster sizes a super-tile", x: 16, y: 474, small: true },
      { text: "@autotune sizes a CTA tile", x: 16, y: 494, small: true },
    ],
  }
}

export function ctaCellRect(geometry: ClusterTilesGeometry, node: number, index: number): Rect {
  const cells = geometry.ctaCells[node]
  return {
    x: cells.x + index * (cells.cellUnits + cells.gapUnits),
    y: cells.y,
    width: cells.cellUnits,
    height: cells.cellUnits,
  }
}

export function lanesRect(geometry: ClusterTilesGeometry): Rect {
  const { x, y, cellUnits, gapUnits } = geometry.lanes
  return { x, y, width: rowWidth(LANES, cellUnits, gapUnits), height: rowWidth(WARPS, cellUnits, gapUnits) }
}

export function pointOf(r: Rect): Point {
  return { x: r.x + r.width / 2, y: r.y + r.height / 2 }
}
