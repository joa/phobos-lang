import { describe, expect, it } from "vitest"
import CtxHierarchyFigure from "../src/components/figures/ctx-hierarchy/ctx-hierarchy.figure"
import {
  affineFor,
  boundsOf,
  fitAspect,
  hierarchyGeometry,
  mapRect,
  panelLines,
  pathOf,
  windowAt,
  type Rect,
} from "../src/components/figures/ctx-hierarchy/geometry"
import { textWidthUnits } from "../src/components/figures/ctx-hierarchy/text"

function inside(inner: Rect, outer: Rect): boolean {
  return (
    inner.x >= outer.x - 1e-6 &&
    inner.y >= outer.y - 1e-6 &&
    inner.x + inner.width <= outer.x + outer.width + 1e-6 &&
    inner.y + inner.height <= outer.y + outer.height + 1e-6
  )
}

const DEFAULT_STATE = { level: "grid", block: "2,1", warp: "3", lane: "17" }

describe("ctx-hierarchy model", () => {
  it("lists the page's worked example", () => {
    const lines = panelLines(pathOf({ x: 2, y: 1 }, 3, 17))
    expect(lines.grid).toEqual(["ctx.blocks.x, .y = 4, 3", "ctx.block.x, .y = 2, 1"])
    expect(lines.cta).toEqual(["ctx.threads = 256", "ctx.warps = 8", "ctx.warp = 3"])
    expect(lines.warp).toEqual(["ctx.lanes = 32", "ctx.lane = 17"])
    expect(lines.lane).toEqual(["ctx.thread = 113", "= 3 * 32 + 17"])
  })
})

describe("ctx-hierarchy zoom", () => {
  const stage = { x: 24, y: 60, width: 440, height: 250 }
  const cell = { x: 300, y: 150, width: 90, height: 60 }
  const target = fitAspect(cell, stage)

  it("grows a cell to the stage's aspect around its center", () => {
    expect(inside(cell, target)).toBe(true)
    expect(target.width / target.height).toBeCloseTo(stage.width / stage.height)
  })

  it("starts at the stage and ends at the cell", () => {
    const start = windowAt(stage, target, 0)
    for (const key of ["x", "y", "width", "height"] as const) expect(start[key]).toBeCloseTo(stage[key])
    const end = windowAt(stage, target, 1)
    expect(end.x).toBeCloseTo(target.x)
    expect(end.width).toBeCloseTo(target.width)
  })

  it("shows the outer level's window and the inner level's mapped window at the same place", () => {
    for (const amount of [0, 0.3, 0.7, 1]) {
      const window = windowAt(stage, target, amount)
      const outer = affineFor(window, stage)
      const inner = affineFor(mapRect(window, target, stage), stage)
      // A point of the cell in outer coordinates lands where its image in inner coordinates lands.
      const point = { x: cell.x + 10, y: cell.y + 5 }
      const mapped = mapRect({ ...point, width: 0, height: 0 }, target, stage)
      expect(point.x * outer.scale + outer.dx).toBeCloseTo(mapped.x * inner.scale + inner.dx)
      expect(point.y * outer.scale + outer.dy).toBeCloseTo(mapped.y * inner.scale + inner.dy)
    }
    const settled = affineFor(mapRect(windowAt(stage, target, 1), target, stage), stage)
    expect(settled.scale).toBeCloseTo(1)
    expect(settled.dx).toBeCloseTo(0)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  describe(`ctx-hierarchy geometry, ${layout}`, () => {
    const geometry = hierarchyGeometry(layout)
    const { stage } = geometry
    const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }

    it("draws every level inside the stage", () => {
      expect(inside(stage, plateRect)).toBe(true)
      expect(inside(boundsOf(geometry.grid.boxes.flat()), stage)).toBe(true)
      expect(inside(boundsOf(geometry.cta.cells.flat()), stage)).toBe(true)
      expect(inside(boundsOf(geometry.warp.cells), stage)).toBe(true)
      expect(inside(geometry.lane.box, stage)).toBe(true)
    })

    it("keeps the panel off the stage and on the plate", () => {
      const lines = panelLines(pathOf({ x: 3, y: 2 }, 7, 31))
      for (const section of geometry.panel) {
        if (layout === "wide") expect(section.x).toBeGreaterThan(stage.x + stage.width)
        else expect(section.heading.y).toBeGreaterThan(stage.y + stage.height + 12)
        const last = section.lineYs[section.lineYs.length - 1]
        expect(last).toBeLessThan(geometry.heightUnits - 8)
        for (const line of lines[section.level])
          expect(section.x + 10 + textWidthUnits(line, 11)).toBeLessThan(geometry.widthUnits - 8)
      }
    })

    it("fits lane numbers and the warp heading", () => {
      const cell = geometry.warp.cells[0]
      expect(textWidthUnits("31", 11)).toBeLessThan(cell.width)
      expect(textWidthUnits("warp 7: threads 224 to 255", 11)).toBeLessThan(stage.width - 52)
    })
  })
}

describe("ctx-hierarchy static render", () => {
  it("settles on the grid and names the picked thread", () => {
    for (const layout of ["wide", "narrow"] as const) {
      const render = CtxHierarchyFigure.renderStatic(layout, DEFAULT_STATE)
      expect(render.readout).toMatch(/^block \(2, 1\), warp 3, lane 17, thread 113\. The grid/)
      expect(render.drawing.match(/class="ctx-level"/g)?.length).toBe(1)
      expect(render.drawing.match(/aria-pressed="true"/g)?.length).toBe(3)
    }
  })

  it("shows the lane level on its own", () => {
    const render = CtxHierarchyFigure.renderStatic("wide", { ...DEFAULT_STATE, level: "lane" })
    expect(render.readout).toContain("One lane")
    expect(render.drawing).toContain(">thread 113<")
    expect(render.drawing).not.toContain("transform=")
  })
})
