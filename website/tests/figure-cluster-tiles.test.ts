import { describe, expect, it } from "vitest"
import ClusterTilesFigure, { READOUT_END } from "../src/components/figures/cluster-tiles/cluster-tiles.figure"
import {
  CTAS_PER_NODE,
  LANES,
  NODES,
  OPENED_CTA,
  OPENED_NODE,
  WARPS,
  clusterTilesGeometry,
  ctaCellRect,
  lanesRect,
  type Rect,
  type TextAt,
} from "../src/components/figures/cluster-tiles/geometry"
import { textWidthUnits } from "../src/components/figures/ctx-hierarchy/text"

const EPS = 1e-6
const MARGIN_UNITS = 12

function inside(inner: Rect, outer: Rect): boolean {
  return (
    inner.x >= outer.x - EPS &&
    inner.y >= outer.y - EPS &&
    inner.x + inner.width <= outer.x + outer.width + EPS &&
    inner.y + inner.height <= outer.y + outer.height + EPS
  )
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

function textRect(at: TextAt, anchor: "start" | "middle" = "start"): Rect {
  const fontUnits = at.small ? 11 : 12
  const width = textWidthUnits(at.text, fontUnits)
  const x = anchor === "middle" ? at.x - width / 2 : at.x
  return { x, y: at.y - fontUnits / 2, width, height: fontUnits }
}

for (const layout of ["wide", "narrow"] as const) {
  describe(`cluster-tiles geometry, ${layout}`, () => {
    const g = clusterTilesGeometry(layout)
    const plateRect = { x: MARGIN_UNITS, y: 0, width: g.widthUnits - 2 * MARGIN_UNITS, height: g.heightUnits }

    it("keeps to the plate's size budget", () => {
      expect(g.heightUnits).toBeLessThanOrEqual(layout === "wide" ? 340 : 540)
    })

    it("puts one super-tile per node, inside the cluster box", () => {
      expect(g.bands).toHaveLength(NODES)
      expect(g.nodes).toHaveLength(NODES)
      for (const band of g.bands) expect(inside(band, g.cluster)).toBe(true)
      expect(inside(g.cluster, plateRect)).toBe(true)
    })

    it("keeps nodes apart and their CTA grids inside them", () => {
      for (let i = 0; i < NODES; i += 1) {
        expect(inside(g.nodes[i], plateRect)).toBe(true)
        expect(overlaps(g.nodes[i], g.cluster)).toBe(false)
        for (let j = i + 1; j < NODES; j += 1) expect(overlaps(g.nodes[i], g.nodes[j])).toBe(false)
        for (let c = 0; c < CTAS_PER_NODE; c += 1) expect(inside(ctaCellRect(g, i, c), g.nodes[i])).toBe(true)
        expect(overlaps(textRect(g.nodeLabels[i]), ctaCellRect(g, i, 0))).toBe(false)
      }
    })

    it("links each super-tile to its node", () => {
      g.links.forEach((line, i) => {
        const start = line.points[0]
        const end = line.points[line.points.length - 1]
        expect(start.x).toBeCloseTo(g.bands[i].x + g.bands[i].width)
        expect(end.x).toBeCloseTo(g.nodes[i].x)
        expect(start.y).toBeCloseTo(end.y)
      })
    })

    it("opens one CTA into warps of lanes", () => {
      expect(WARPS * LANES).toBe(256)
      expect(inside(g.cta, plateRect)).toBe(true)
      expect(inside(lanesRect(g), g.cta)).toBe(true)
      for (const node of g.nodes) expect(overlaps(node, g.cta)).toBe(false)
      const cell = ctaCellRect(g, OPENED_NODE, OPENED_CTA)
      const start = g.zoom.points[0]
      const end = g.zoom.points[g.zoom.points.length - 1]
      expect(start.x).toBeGreaterThanOrEqual(cell.x - EPS)
      expect(start.x).toBeLessThanOrEqual(cell.x + cell.width + EPS)
      expect(end.x).toBeGreaterThanOrEqual(g.cta.x - EPS)
      expect(end.y).toBeGreaterThanOrEqual(g.cta.y - EPS)
      expect(end.y).toBeLessThanOrEqual(g.cta.y + g.cta.height + EPS)
    })

    it("keeps labels on the plate and off each other", () => {
      const texts = [
        ...g.headers.map(at => textRect(at)),
        ...g.bandLabels.map(at => textRect(at, layout === "narrow" ? "middle" : "start")),
        ...g.nodeLabels.map(at => textRect(at)),
        textRect(g.ctaLabel),
        ...g.notes.map(at => textRect(at)),
      ]
      for (const rect of texts) expect(inside(rect, plateRect)).toBe(true)
      for (let i = 0; i < texts.length; i += 1)
        for (let j = i + 1; j < texts.length; j += 1) expect(overlaps(texts[i], texts[j])).toBe(false)
      expect(inside(textRect(g.ctaLabel), g.cta)).toBe(true)
      expect(overlaps(textRect(g.ctaLabel), lanesRect(g))).toBe(false)
      for (const at of g.headers) expect(at.y).toBeLessThan(g.cluster.y)
      for (const note of g.notes) {
        for (const node of g.nodes) expect(overlaps(textRect(note), node)).toBe(false)
        expect(overlaps(textRect(note), g.cta)).toBe(false)
      }
    })
  })
}

describe("cluster-tiles static render", () => {
  it("draws the settled end state in both layouts", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = ClusterTilesFigure.renderStatic(layout, {})
      expect(render.readout).toBe(READOUT_END)
      expect(render.drawing).toContain("cluster (experimental)")
      expect(render.drawing.match(/stroke-dasharray="4 3"/g)?.length).toBe(1 + NODES)
      expect(render.drawing.match(/chart-cell is-on/g)?.length).toBe(NODES * CTAS_PER_NODE + WARPS * LANES)
      expect(render.drawing).toContain("chart-cell is-on is-hot")
      expect(render.drawing.match(/cluster-experimental is-on/g)?.length).toBe(NODES)
    }
  })
})
