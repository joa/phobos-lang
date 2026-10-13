// Masking: a tile that crosses a tensor's edge. Picking a tile stages it into the shared tile a; the
// elements past the runtime extent are drawn as cost, read as 0 and their stores skipped. Toggling
// @aligned draws the promise instead: extents that are whole tiles, no masks, 16-byte vectors, and
// the check at entry. Schematic.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { box, label, node, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { easeInOut, lerp } from "../../../lib/figure/math"
import { sweepWire } from "../../../lib/figure/motion"
import type { Polyline } from "../../../lib/figure/polyline"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import { enablePicking, makePickable, setTabStop } from "../ctx-hierarchy/pick"
import type { Block } from "../ctx-hierarchy/rules"
import {
  elementMasked,
  extentsOf,
  GRID_COLS,
  GRID_ROWS,
  M,
  maskingGeometry,
  N,
  PADDED_COLS,
  PADDED_ROWS,
  parseBlock,
  TILE,
  tileStats,
  type MaskingGeometry,
  type Rect,
} from "./geometry"

const FILL_MS = 560
const MAX_WIRE_POINTS = 4

export default class MaskingFigure extends ChartFigure {
  #geometry: MaskingGeometry | null = null
  #extentBox: SVGRectElement | null = null
  #tensorCells: SVGRectElement[][] = []
  #hits: SVGRectElement[] = []
  #outline: SVGRectElement | null = null
  #lead: SVGPathElement | null = null
  #wire: SVGPathElement | null = null
  #wireNodes: SVGCircleElement[] = []
  #sweep: SVGPathElement | null = null
  #sharedCells: SVGRectElement[][] = []
  #zeros: SVGTextElement[][] = []
  #vectors: SVGRectElement[] = []
  #tensorLabel: SVGTextElement | null = null
  #promise: SVGTextElement | null = null
  #notes: SVGTextElement[] = []
  #labels: SVGTextElement[] = []
  #shownExtent: Rect | null = null
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ tile: "1,1" }, { tile: "2,1" }, { tile: "2,3" }, { aligned: "on" }, { aligned: "off" }]
  }

  get #block(): Block {
    return parseBlock(String(this.state.tile ?? ""))
  }

  get #aligned(): boolean {
    return this.state.aligned === "on"
  }

  override start(): void {
    super.start()
    enablePicking(this.svg)
  }

  override build(layout: Layout): void {
    const geometry = maskingGeometry(layout)
    this.#geometry = geometry
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    const root = this.drawing
    plate(root, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: geometry.plateLabel,
      schematic: true,
    })
    this.#labels = []
    const block = this.#block

    const tensorGroup = svgEl("g", { class: "sp-global" }, root)
    this.#tensorLabel = this.#addLabel(tensorGroup, geometry.tensorLabel.x, geometry.tensorLabel.y)
    this.#extentBox = box(tensorGroup, { ...geometry.extentBox(extentsOf(false)), className: "is-on is-outline" })
    const cellGroup = svgEl("g", { class: "chart-cells" }, tensorGroup)
    this.#tensorCells = geometry.tensorCells.map(row => row.map(cell => rect(cellGroup, cell, "chart-cell")))
    this.#hits = []
    for (let x = 0; x < GRID_ROWS; x += 1) {
      for (let y = 0; y < GRID_COLS; y += 1) {
        const hit = rect(tensorGroup, geometry.tileHit({ x, y }), "chart-hit")
        makePickable(hit, {
          group: "tile",
          value: `${x},${y}`,
          label: `block (${x}, ${y})`,
          row: x,
          col: y,
          pressed: x === block.x && y === block.y,
        })
        this.#hits.push(hit)
      }
    }
    this.#outline = box(root, { ...geometry.outline(block), space: "global", className: "is-hot is-outline" })
    this.#lead = svgEl("path", { class: "chart-wire is-on sp-global" }, root)
    this.#wire = svgEl("path", { class: "chart-wire is-on sp-global" }, root)
    this.#wireNodes = Array.from({ length: MAX_WIRE_POINTS }, () =>
      node(root, 0, 0, { space: "global", className: "is-on" })
    )
    this.#sweep = sweepWire(root, "M0 0", "global")

    const sharedGroup = svgEl("g", { class: "sp-shared" }, root)
    this.#addLabel(sharedGroup, geometry.sharedLabel.x, geometry.sharedLabel.y, geometry.sharedLabel.text)
    this.#vectors = geometry.vectors.flat().map(at => rect(sharedGroup, at, "chart-box is-on is-outline"))
    const sharedCellGroup = svgEl("g", { class: "chart-cells" }, sharedGroup)
    this.#sharedCells = geometry.sharedCells.map(row => row.map(cell => rect(sharedCellGroup, cell, "chart-cell")))
    this.#zeros = geometry.sharedCells.map(row =>
      row.map(cell =>
        label(sharedGroup, "0", {
          x: cell.x + cell.width / 2,
          y: cell.y + cell.height / 2 + 0.5,
          anchor: "middle",
          space: "cost",
          className: "is-small is-warn",
        })
      )
    )

    this.#addLabel(root, geometry.code.x, geometry.code.y, geometry.code.text)
    this.#promise = this.#addLabel(root, geometry.promise.x, geometry.promise.y, geometry.promise.text)
    this.#notes = geometry.notes.map(at => this.#addLabel(root, at.x, at.y))
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    this.motion.finishAll()
    const timeline = this.#timeline ?? this.#makeTimeline(false)
    timeline.seekEnd()
    this.#timeline = null
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) {
      this.motion.drawIn(this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire, .chart-box"))
      this.motion.enter(this.#labels, { delayMs: 240 })
    }
    super.onFirstVisible()
  }

  override onChange(group: string): void {
    if (group === "tile") {
      this.motion.glide(this.#outline ? [this.#outline] : [], () => this.#placeOutline())
      this.#run(false)
    }
    if (group === "aligned") this.#run(true)
  }

  // Once the reader has taken over, a loop stages their tile again.
  override replay(): void {
    if (!this.hasInteracted) return super.replay()
    this.#run(false)
  }

  #run(animateExtents: boolean): void {
    this.#timeline = this.#makeTimeline(animateExtents)
    this.play()
  }

  #makeTimeline(animateExtents: boolean): Timeline {
    const aligned = this.#aligned
    const to = this.#geometry?.extentBox(extentsOf(aligned)) ?? null
    const from = animateExtents && this.#shownExtent ? this.#shownExtent : to
    const scale = (durationMs: number) => (this.reducedMotion ? 0 : durationMs)
    const steps: TimelineStep[] = [
      { durationMs: 0, apply: () => this.#applyState() },
      {
        durationMs: from === to ? 0 : scale(this.motion.tokens.dur3Ms),
        apply: progress => {
          if (!from || !to) return
          const eased = easeInOut(progress)
          this.#shownExtent = {
            x: lerp(from.x, to.x, eased),
            y: lerp(from.y, to.y, eased),
            width: lerp(from.width, to.width, eased),
            height: lerp(from.height, to.height, eased),
          }
          setRect(this.#extentBox, this.#shownExtent)
        },
      },
      {
        durationMs: scale(FILL_MS),
        apply: progress => {
          const litRows = Math.ceil(progress * TILE)
          this.#sweep?.classList.toggle("is-on", progress < 1)
          this.#sharedCells.forEach((row, rowIndex) =>
            row.forEach((_, col) => this.#lightShared(rowIndex, col, rowIndex < litRows))
          )
        },
      },
    ]
    return new Timeline(steps)
  }

  #applyState(): void {
    const block = this.#block
    const aligned = this.#aligned
    const { m, n } = extentsOf(aligned)
    this.#placeOutline()
    setTabStop(this.#hits, this.#hits[block.x * GRID_COLS + block.y])
    for (let row = 0; row < PADDED_ROWS; row += 1) {
      for (let col = 0; col < PADDED_COLS; col += 1) {
        const inside = row < m && col < n
        const inTile = Math.floor(row / TILE) === block.x && Math.floor(col / TILE) === block.y
        const cell = this.#tensorCells[row][col]
        cell.classList.toggle("is-on", inside && inTile)
        cell.classList.toggle("is-warn", !inside && inTile)
        cell.classList.toggle("sp-cost", !inside && inTile)
        cell.classList.toggle("is-hidden", !inside && !inTile)
      }
    }
    for (let row = 0; row < TILE; row += 1) for (let col = 0; col < TILE; col += 1) this.#lightShared(row, col, false)
    for (const vector of this.#vectors) vector.classList.toggle("is-hidden", !aligned)

    const stats = tileStats(block, aligned)
    const name = `block (${block.x}, ${block.y})`
    if (this.#tensorLabel)
      this.#tensorLabel.textContent = aligned ? `global  A  ${m} x ${n} f32, whole tiles` : `global  A  ${M} x ${N} f32`
    this.#promise?.classList.toggle("is-on", aligned)
    this.#promise?.classList.toggle("is-idle", !aligned)
    const notes: [string, string][] = aligned
      ? [
          ["M, N whole tiles", "is-on"],
          [`grid ${GRID_ROWS} x ${GRID_COLS} CTAs`, ""],
          ["no masks", "is-on"],
          ["16-byte vectors", "is-on"],
          ["checked once at entry", ""],
        ]
      : [
          [`M = ${M}, N = ${N}`, ""],
          [`grid ${GRID_ROWS} x ${GRID_COLS} CTAs`, ""],
          [`in bounds: ${stats.rows} x ${stats.cols}`, ""],
          [`masked: ${stats.masked}`, stats.masked > 0 ? "is-warn" : ""],
          [stats.masked > 0 ? "read 0, stores skipped" : "checked, none masked", stats.masked > 0 ? "is-warn" : ""],
        ]
    notes.forEach(([text, state], index) => {
      const note = this.#notes[index]
      note.textContent = text
      note.classList.toggle("is-on", state === "is-on")
      note.classList.toggle("is-warn", state === "is-warn")
      note.classList.toggle("sp-cost", state === "is-warn")
    })

    this.setReadout(
      aligned
        ? `@aligned(M = 8, N = 8) promises whole tiles, drawn here as ${m} x ${n}: ${name} loads its 64 elements with no masks, as 16-byte vectors. The promise is checked once at entry, and a launch with M = ${M} traps.`
        : stats.masked > 0
          ? `${name}: ${stats.inBounds} of 64 elements are in bounds; ${stats.masked} read as 0 and their stores are skipped.`
          : `${name}: all 64 elements are in bounds. The mask is still checked against the runtime extent, which rules out the widest vector loads.`
    )
  }

  #lightShared(row: number, col: number, lit: boolean): void {
    const masked = elementMasked(this.#block, row, col, this.#aligned)
    const cell = this.#sharedCells[row]?.[col]
    if (!cell) return
    cell.classList.toggle("is-on", lit && !masked)
    cell.classList.toggle("is-warn", lit && masked)
    cell.classList.toggle("sp-cost", lit && masked)
    this.#zeros[row][col].classList.toggle("is-hidden", !(lit && masked))
  }

  #placeOutline(): void {
    const geometry = this.#geometry
    if (!geometry) return
    const block = this.#block
    setRect(this.#outline, geometry.outline(block))
    this.#lead?.setAttribute("d", geometry.lead(block).toPathData())
    const wire = geometry.wire(block)
    this.#wire?.setAttribute("d", wire.toPathData())
    this.#sweep?.setAttribute("d", wire.toPathData())
    placeNodes(this.#wireNodes, wire)
  }

  #addLabel(parent: Element, x: number, y: number, text = ""): SVGTextElement {
    const element = label(parent, text, { x, y })
    this.#labels.push(element)
    return element
  }
}

function rect(parent: Element, at: Rect, className: string): SVGRectElement {
  return svgEl("rect", { class: className, x: at.x, y: at.y, width: at.width, height: at.height }, parent)
}

function setRect(element: SVGRectElement | null, at: Rect): void {
  if (!element) return
  element.setAttribute("x", String(round(at.x)))
  element.setAttribute("y", String(round(at.y)))
  element.setAttribute("width", String(round(at.width)))
  element.setAttribute("height", String(round(at.height)))
}

function placeNodes(nodes: SVGCircleElement[], line: Polyline): void {
  nodes.forEach((dot, index) => {
    const point = line.points[index]
    dot.classList.toggle("is-hidden", !point)
    if (!point) return
    dot.setAttribute("cx", String(round(point.x)))
    dot.setAttribute("cy", String(round(point.y)))
  })
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
