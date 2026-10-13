// The compiler scene's geometry and journey: every station fits its frame in both layouts, the graph is
// the real DAG with no label on another, the cloud's points stand for what the data counts, and the
// clock dwells, moves, wraps and rests longer after a jump.

import { describe, expect, it } from "vitest"
import { PipelineClock } from "../src/components/home/pipeline/clock"
import { structure, stagedPasses } from "../src/components/home/pipeline/data"
import { graphStation, LABEL_UNITS } from "../src/components/home/pipeline/graph"
import { shapeMarkup } from "../src/components/home/pipeline/markup"
import { regionDialects } from "../src/components/home/pipeline/mlir"
import { passesStation } from "../src/components/home/pipeline/passes"
import { ringSpot, ringTurn } from "../src/components/home/pipeline/ring"
import { allGeometry, STATIONS } from "../src/components/home/pipeline/stations"
import { FRAMES, PARTICLES, overlaps, textRect, type Layout } from "../src/components/home/pipeline/types"

const LAYOUTS: Layout[] = ["wide", "narrow"]

describe("stations", () => {
  for (const layout of LAYOUTS) {
    it(`fit the ${layout} frame, with a full cloud and labels at the floor or above`, () => {
      const frame = FRAMES[layout]
      allGeometry(layout).forEach((geometry, index) => {
        const id = STATIONS[index].id
        expect(geometry.targets, id).toHaveLength(PARTICLES)
        for (const point of geometry.targets.filter(target => target.alpha > 0)) {
          expect(point.x, id).toBeGreaterThanOrEqual(0)
          expect(point.x, id).toBeLessThanOrEqual(frame.width)
          expect(point.y, id).toBeGreaterThanOrEqual(0)
          expect(point.y, id).toBeLessThanOrEqual(frame.height)
        }
        for (const shape of geometry.shapes.filter(shape => shape.tag === "text")) {
          const size = Number(shape.attrs["font-size"] ?? LABEL_UNITS)
          expect(size).toBeGreaterThanOrEqual(12.5)
          const rect = textRect({ ...shape, text: shape.text ?? shape.spans?.map(span => span.text).join("") }, size)
          expect(rect.x, `${id}: ${shape.text}`).toBeGreaterThanOrEqual(0)
          expect(rect.x + rect.width, `${id}: ${shape.text}`).toBeLessThanOrEqual(frame.width + 0.5)
        }
      })
    })
  }

  it("never write a code dump: no text longer than a source line", () => {
    for (const layout of LAYOUTS) {
      for (const geometry of allGeometry(layout)) {
        for (const shape of geometry.shapes) expect((shape.text ?? "").length).toBeLessThanOrEqual(90)
      }
    }
  })
})

describe("graph IR", () => {
  for (const layout of LAYOUTS) {
    it(`draws every node and edge of the DAG, labels apart (${layout})`, () => {
      const { edgeRoutes, labels, glyphBoxes } = graphStation(layout)
      expect(glyphBoxes).toHaveLength(structure.graph.nodes.length)
      expect(edgeRoutes).toHaveLength(structure.graph.edges.length)
      expect(edgeRoutes.filter(edge => edge.kind === "back")).toHaveLength(1)
      const rects = labels.map(label => textRect(label, LABEL_UNITS))
      for (let a = 0; a < rects.length; a += 1) {
        for (let b = a + 1; b < rects.length; b += 1) {
          expect(overlaps(rects[a], rects[b]), `${labels[a].text} / ${labels[b].text}`).toBe(false)
        }
      }
    })
  }

  it("sends the carried value back above the loop's box", () => {
    const { edgeRoutes, geometry } = graphStation("wide")
    const back = edgeRoutes.find(edge => edge.kind === "back")
    const box = geometry.shapes.find(shape => String(shape.attrs.class).includes("pp-region"))
    expect(back && box).toBeTruthy()
    const highest = Math.min(...back!.route.segments.flat().map(point => point.y))
    expect(highest).toBeLessThan(Number(box!.attrs.y))
  })
})

describe("points stand for the data", () => {
  it("one per op in the MLIR regions", () => {
    const ops = Object.values(regionDialects(structure.mlir.root)).reduce((sum, count) => sum + count, 0)
    expect(ops).toBe(structure.mlir.root.total)
    const mlir = allGeometry("wide")[STATIONS.findIndex(station => station.id === "mlir")]
    expect(mlir.targets.filter(target => target.alpha > 0)).toHaveLength(ops)
  })

  it("one per PTX instruction", () => {
    const ptx = allGeometry("wide")[STATIONS.findIndex(station => station.id === "ptx")]
    expect(ptx.targets.filter(target => target.alpha > 0)).toHaveLength(structure.ptx.instructions)
    expect(structure.ptx.instructions).toBeLessThanOrEqual(PARTICLES)
  })

  it("passes shows the build without @aligned, buffers inside the strip", () => {
    const view = stagedPasses()
    expect(view.variant?.drop).toBe("@aligned")
    for (const buffer of view.buffers) expect(buffer.offsetBytes + buffer.bytes).toBeLessThanOrEqual(view.peakBytes)
    const { timing, shapes } = passesStation("wide")
    expect(timing.buffers).toHaveLength(view.buffers.length)
    expect(shapes.some(shape => shape.text?.startsWith("without @aligned"))).toBe(true)
  })
})

describe("clock", () => {
  const clock = () => new PipelineClock({ count: 3, dwellMs: () => 1000, moveMs: 500, jumpMoveMs: 200, onAt: 0.5, jumpRestMs: 5000 })

  it("dwells, moves to the next station and wraps to the first", () => {
    const c = clock()
    c.advance(999)
    expect(c.moving).toBe(false)
    c.advance(2)
    expect(c.station).toBe(1)
    expect(c.moving).toBe(true)
    expect(c.stationMs).toBeLessThan(0)
    c.advance(500)
    expect(c.moving).toBe(false)
    c.advance(1000)
    c.advance(500)
    c.advance(1000)
    expect(c.station).toBe(0)
  })

  it("rests longer after a jump", () => {
    const c = clock()
    c.jump(2)
    expect(c.station).toBe(2)
    c.advance(200)
    c.advance(3000)
    expect(c.station).toBe(2)
    c.advance(2000)
    expect(c.station).toBe(0)
  })
})

describe("ring", () => {
  it("puts the turned-to station at the front and turns the short way", () => {
    expect(ringSpot(3, 7, 3).depth).toBe(1)
    expect(ringTurn(6, 0, 7, 1)).toBe(7)
    expect(ringTurn(0, 6, 7, 1)).toBe(-1)
  })
})

describe("markup", () => {
  it("escapes text and carries delays", () => {
    const markup = shapeMarkup({ tag: "text", attrs: { x: 1, y: 2 }, text: "tile<f32>", delayMs: 40 })
    expect(markup).toContain("tile&lt;f32&gt;")
    expect(markup).toContain("--d: 40ms")
  })
})
