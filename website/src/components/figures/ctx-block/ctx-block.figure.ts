// ctx.block and ctx.blocks: an output tensor C[M, N] tiled into CTAs. The M and N ranges regrow the
// grid over --dur-3 while ctx.blocks counts along; picking a CTA outlines the tile its ctx.block
// owns; the z toggle stacks the grid behind itself for a batch axis. Schematic.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { box, label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { clamp, easeInOut, lerp } from "../../../lib/figure/math"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline } from "../../../lib/figure/timeline"
import { enablePicking, makePickable, setTabStop } from "../ctx-hierarchy/pick"
import { EXTENT_MULTIPLE, gridExtent } from "../ctx-hierarchy/rules"
import {
  BATCH,
  blockGeometry,
  blockModel,
  DEFAULT_M,
  DEFAULT_N,
  DEFAULT_PICK,
  isWholeTile,
  MAX_BLOCKS,
  MAX_EXTENT,
  MIN_EXTENT,
  parsePick,
  pickValue,
  readoutFor,
  sliceText,
  TILE,
  type BlockGeometry,
  type BlockModel,
} from "./geometry"

interface Ghost {
  group: SVGGElement
  grid: SVGRectElement
  tensor: SVGRectElement
}

export default class CtxBlockFigure extends ChartFigure {
  #geometry: BlockGeometry | null = null
  #boxes: SVGRectElement[][] = []
  #hits: SVGRectElement[][] = []
  #tensor: SVGRectElement | null = null
  #overRight: SVGRectElement | null = null
  #overBottom: SVGRectElement | null = null
  #ghosts: Ghost[] = []
  #rowIndex: SVGTextElement[] = []
  #colIndex: SVGTextElement[] = []
  #panel: SVGTextElement[] = []
  #labels: SVGTextElement[] = []
  #shown: { m: number; n: number } | null = null
  #model: BlockModel | null = null
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [
      { cta: "4,6" },
      { m: 384 },
      { n: 512 },
      { z: "on" },
      { m: DEFAULT_M, n: DEFAULT_N, z: "off", cta: DEFAULT_PICK },
    ]
  }

  override start(): void {
    super.start()
    enablePicking(this.svg)
  }

  #target(): BlockModel {
    const extent = (value: StateValue | undefined, fallback: number) =>
      clamp(Number(value ?? fallback), MIN_EXTENT, MAX_EXTENT)
    return blockModel(
      extent(this.state.m, DEFAULT_M),
      extent(this.state.n, DEFAULT_N),
      parsePick(String(this.state.cta ?? DEFAULT_PICK)),
      this.state.z === "on"
    )
  }

  override build(layout: Layout): void {
    const geometry = blockGeometry(layout)
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
    const addLabel = (parent: Element, text: string, at: { x: number; y: number }, extra = {}) => {
      const element = label(parent, text, { x: at.x, y: at.y, ...extra })
      this.#labels.push(element)
      return element
    }

    const { axisRows, axisCols } = geometry
    addLabel(root, axisCols.text ?? "", axisCols)
    addLabel(root, axisRows.text ?? "", axisRows, {
      anchor: "middle",
      transform: `rotate(${axisRows.rotate} ${axisRows.x} ${axisRows.y})`,
    })

    // Batches behind the front grid, farthest first.
    this.#ghosts = Array.from({ length: BATCH - 1 }, () => {
      const group = svgEl("g", { class: "sp-global is-hidden" }, root)
      const grid = box(group, { x: geometry.originX, y: geometry.originY, width: 0, height: 0, className: "is-idle" })
      const tensor = box(group, { x: geometry.originX, y: geometry.originY, width: 0, height: 0, className: "is-idle" })
      return { group, grid, tensor }
    }).reverse()

    const front = svgEl("g", { class: "sp-global" }, root)
    this.#boxes = []
    for (let bx = 0; bx < MAX_BLOCKS; bx += 1) {
      this.#boxes.push(Array.from({ length: MAX_BLOCKS }, (_, by) => box(front, geometry.tileRect(bx, by))))
    }
    this.#overRight = box(root, { x: 0, y: 0, width: 0, height: 0, space: "cost", className: "is-warn" })
    this.#overBottom = box(root, { x: 0, y: 0, width: 0, height: 0, space: "cost", className: "is-warn" })
    this.#tensor = box(root, {
      x: geometry.originX,
      y: geometry.originY,
      width: 0,
      height: 0,
      space: "global",
      className: "is-on",
    })

    const pick = this.#target().pick
    this.#hits = []
    const hits = svgEl("g", {}, root)
    for (let bx = 0; bx < MAX_BLOCKS; bx += 1) {
      this.#hits.push(
        Array.from({ length: MAX_BLOCKS }, (_, by) => {
          const hit = svgEl("rect", { class: "chart-hit", ...geometry.tileRect(bx, by) }, hits)
          makePickable(hit, {
            group: "cta",
            value: pickValue({ x: bx, y: by }),
            label: `CTA (${bx}, ${by})`,
            row: bx,
            col: by,
            pressed: bx === pick.x && by === pick.y,
          })
          return hit
        })
      )
    }

    this.#rowIndex = Array.from({ length: MAX_BLOCKS }, (_, bx) =>
      addLabel(root, String(bx), geometry.rowIndex(bx), { anchor: "end", className: "is-small" })
    )
    this.#colIndex = Array.from({ length: MAX_BLOCKS }, (_, by) =>
      addLabel(root, String(by), geometry.colIndex(by), { anchor: "middle", className: "is-small" })
    )
    this.#panel = geometry.panel.map(at => addLabel(root, "", at))
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    this.motion.finishAll()
    const timeline = this.#timeline ?? this.#makeTimeline()
    timeline.seekEnd()
    this.#timeline = null
    this.#placeGhosts(this.state.z === "on")
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) {
      this.motion.drawIn(this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-box:not(.is-warn)"), {
        staggerMs: 6,
      })
      this.motion.enter(this.#labels, { delayMs: 240, staggerMs: 10 })
    }
    super.onFirstVisible()
  }

  override onInput(param: string, value: number): void {
    if (param !== "m" && param !== "n") return
    const output = this.root.querySelector(`output[data-for='${param}']`)
    if (output) output.textContent = String(value)
    this.#run()
  }

  override onChange(group: string, value: string): void {
    if (group === "cta") this.#run()
    if (group === "z") {
      this.#run()
      this.motion.glide(
        this.#ghosts.map(ghost => ghost.group),
        () => this.#placeGhosts(value === "on")
      )
    }
  }

  #run(): void {
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  #makeTimeline(): Timeline {
    const model = this.#target()
    if (pickValue(model.pick) !== String(this.state.cta ?? DEFAULT_PICK)) this.setState({ cta: pickValue(model.pick) })
    const from = this.#shown ?? { m: model.m, n: model.n }
    const moves = from.m !== model.m || from.n !== model.n
    return new Timeline([
      { durationMs: 0, apply: () => this.#applyModel(model) },
      {
        durationMs: moves ? this.motion.tokens.dur3Ms : 0,
        apply: progress => {
          const eased = easeInOut(progress)
          this.#placeExtent(lerp(from.m, model.m, eased), lerp(from.n, model.n, eased))
        },
      },
    ])
  }

  #applyModel(model: BlockModel): void {
    this.#model = model
    const { pick } = model
    this.#boxes.forEach((row, bx) =>
      row.forEach((cell, by) => cell.classList.toggle("is-hot", bx === pick.x && by === pick.y))
    )
    this.#rowIndex.forEach((text, bx) => text.classList.toggle("is-on", bx === pick.x))
    this.#colIndex.forEach((text, by) => text.classList.toggle("is-on", by === pick.y))
    setTabStop(this.#hits.flat(), this.#hits[pick.x]?.[pick.y])

    const lines = this.#panel
    const set = (index: number, text: string, className?: string) => {
      const line = lines[index]
      if (!line) return
      line.textContent = text
      for (const name of ["is-on", "is-hot", "is-warn", "is-small", "sp-cost", "sp-global"])
        line.classList.toggle(name, className?.split(" ").includes(name) ?? false)
    }
    set(3, model.batched ? `ctx.blocks.z = ${BATCH}, one grid per batch` : "ctx.blocks.z = 1")
    set(5, "picked CTA", "is-small")
    set(6, `ctx.block.x = ${pick.x}, ctx.block.y = ${pick.y}`, "is-on sp-global")
    set(7, model.batched ? "ctx.block.z = 0, the front batch" : "ctx.block.z = 0")
    set(8, `owns ${sliceText(model)}`)
    set(
      9,
      isWholeTile(model)
        ? "a whole tile inside C"
        : `${model.rowsInside} x ${model.colsInside} inside C, the rest masked`,
      isWholeTile(model) ? "is-small" : "is-small is-warn sp-cost"
    )
    this.setReadout(readoutFor(model))
  }

  // Draws the tensor, the grid and the overhang for extents that may be mid-glide. The counts
  // follow the drawn extent, so ctx.blocks counts along as the grid regrows.
  #placeExtent(m: number, n: number): void {
    const geometry = this.#geometry
    const model = this.#model
    if (!geometry || !model) return
    this.#shown = { m, n }
    const blocksX = gridExtent(m, TILE)
    const blocksY = gridExtent(n, TILE)
    const s = geometry.unitsPerElement
    const tensorWidth = n * s
    const tensorHeight = m * s
    const gridWidth = blocksY * geometry.tileUnits
    const gridHeight = blocksX * geometry.tileUnits
    const { originX: x, originY: y } = geometry

    this.#boxes.forEach((row, bx) =>
      row.forEach((cell, by) => {
        const hidden = bx >= blocksX || by >= blocksY
        cell.classList.toggle("is-hidden", hidden)
        this.#hits[bx][by].classList.toggle("is-hidden", hidden)
      })
    )
    // Focus never stays on a tile the grid has just dropped.
    const focused = typeof document === "undefined" ? null : document.activeElement
    if (focused?.classList.contains("is-hidden") && this.#hits.flat().includes(focused as SVGRectElement)) {
      this.#hits[model.pick.x]?.[model.pick.y]?.focus()
    }
    this.#rowIndex.forEach((text, bx) => text.classList.toggle("is-hidden", bx >= blocksX))
    this.#colIndex.forEach((text, by) => text.classList.toggle("is-hidden", by >= blocksY))
    setRect(this.#tensor, { x, y, width: tensorWidth, height: tensorHeight })
    setRect(this.#overRight, { x: x + tensorWidth, y, width: gridWidth - tensorWidth, height: gridHeight })
    setRect(this.#overBottom, { x, y: y + tensorHeight, width: tensorWidth, height: gridHeight - tensorHeight })
    for (const ghost of this.#ghosts) {
      setRect(ghost.grid, { x, y, width: gridWidth, height: gridHeight })
      setRect(ghost.tensor, { x, y, width: tensorWidth, height: tensorHeight })
    }

    const shownM = Math.round(m / EXTENT_MULTIPLE) * EXTENT_MULTIPLE
    const shownN = Math.round(n / EXTENT_MULTIPLE) * EXTENT_MULTIPLE
    const lines = this.#panel
    const write = (index: number, text: string) => {
      if (lines[index] && lines[index].textContent !== text) lines[index].textContent = text
    }
    write(0, `C[${shownM}, ${shownN}], ${TILE} x ${TILE} tiles`)
    write(1, `ctx.blocks.x = ceil(${shownM} / ${TILE}) = ${blocksX}`)
    write(2, `ctx.blocks.y = ceil(${shownN} / ${TILE}) = ${blocksY}`)
    const total = blocksX * blocksY * model.blocksZ
    write(
      4,
      model.batched
        ? `grid ${blocksX} x ${blocksY} x ${BATCH} = ${total} CTAs`
        : `grid ${blocksX} x ${blocksY} = ${total} CTAs`
    )
  }

  #placeGhosts(on: boolean): void {
    const step = this.#geometry?.ghostStepUnits ?? 0
    this.#ghosts.forEach((ghost, index) => {
      const depth = this.#ghosts.length - index
      ghost.group.classList.toggle("is-hidden", !on)
      ghost.group.setAttribute("transform", on ? `translate(${depth * step} ${-depth * step})` : "translate(0 0)")
    })
  }
}

function setRect(element: SVGRectElement | null, rect: { x: number; y: number; width: number; height: number }): void {
  if (!element) return
  const width = Math.max(0, rect.width)
  const height = Math.max(0, rect.height)
  element.setAttribute("x", String(round(rect.x)))
  element.setAttribute("y", String(round(rect.y)))
  element.setAttribute("width", String(round(width)))
  element.setAttribute("height", String(round(height)))
  element.classList.toggle("is-hidden", width < 0.5 || height < 0.5)
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
