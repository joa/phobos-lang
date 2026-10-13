// The hierarchy: one continuous zoom from the grid of CTAs to one CTA as its warps, one warp as its
// lanes, and one lane. Each level labels the fields that address it, and picking a cell names the
// thread in the readout, such as "block (2, 1), warp 3, lane 17, thread 113". A pick zooms one level
// in; the breadcrumb (and Escape) zooms back out. Each zoom step glides over --dur-3. Schematic.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { box, label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { clamp, easeInOut } from "../../../lib/figure/math"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import {
  affineFor,
  DEFAULT_PATH,
  fitAspect,
  GRID_X,
  GRID_Y,
  hierarchyGeometry,
  LEVELS,
  mapRect,
  panelLines,
  pathOf,
  transformAttribute,
  WARPS,
  windowAt,
  type HierarchyGeometry,
  type Level,
  type Rect,
  type TextAt,
} from "./geometry"
import { enablePicking, makePickable } from "./pick"
import { describeThread, LANES, type ThreadPath } from "./rules"
import { textWidthUnits } from "./text"

// The crossfade between two levels happens around the middle of a zoom step.
const FADE_OUT_AT = 0.55
const FADE_IN_AT = 0.35
const PATH_GAP_UNITS = 10

export default class CtxHierarchyFigure extends ChartFigure {
  #geometry: HierarchyGeometry | null = null
  #levels = new Map<Level, SVGGElement>()
  #pickables = new Map<Level, SVGElement[]>()
  #blockBoxes: SVGRectElement[][] = []
  #warpLines: SVGRectElement[][][] = []
  #warpCells: SVGRectElement[][] = []
  #warpLabels: SVGTextElement[] = []
  #laneCells: SVGRectElement[] = []
  #laneNumbers: SVGTextElement[] = []
  #warpHeading: SVGTextElement | null = null
  #laneLines: SVGTextElement[] = []
  #pathLabels: SVGTextElement[] = []
  #sectionHeadings = new Map<Level, SVGTextElement>()
  #sectionLines = new Map<Level, SVGTextElement[]>()
  #shownLevel: Level | null = null
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ level: "grid" }, { level: "cta" }, { level: "warp" }, { level: "lane" }, { level: "grid" }]
  }

  get #level(): Level {
    const level = String(this.state.level ?? "grid")
    return (LEVELS as readonly string[]).includes(level) ? (level as Level) : "grid"
  }

  get #path(): ThreadPath {
    const [x, y] = String(this.state.block ?? `${DEFAULT_PATH.block.x},${DEFAULT_PATH.block.y}`)
      .split(",")
      .map(Number)
    const block = { x: clamp(x || 0, 0, GRID_X - 1), y: clamp(y || 0, 0, GRID_Y - 1) }
    const warp = clamp(Number(this.state.warp ?? DEFAULT_PATH.warp), 0, WARPS - 1)
    const lane = clamp(Number(this.state.lane ?? DEFAULT_PATH.lane), 0, LANES - 1)
    return pathOf(block, warp, lane)
  }

  override start(): void {
    super.start()
    enablePicking(this.svg)
    this.svg.addEventListener("keydown", event => {
      if (event.key !== "Escape") return
      const parent = LEVELS[LEVELS.indexOf(this.#level) - 1]
      if (!parent) return
      event.preventDefault()
      this.root.querySelector<HTMLElement>(`[data-group="level"][data-value="${parent}"]`)?.click()
    })
  }

  override build(layout: Layout): void {
    const geometry = hierarchyGeometry(layout)
    this.#geometry = geometry
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    const root = this.drawing
    plate(root, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: "FIG / CTX / HIERARCHY",
      schematic: true,
    })
    const { stage } = geometry
    const clipId = `${this.id}-stage`
    const clip = svgEl("clipPath", { id: clipId }, svgEl("defs", {}, root))
    svgEl("rect", { x: stage.x - 4, y: stage.y - 4, width: stage.width + 8, height: stage.height + 8 }, clip)
    const stageGroup = svgEl("g", { "clip-path": `url(#${clipId})` }, root)
    const path = this.#path
    this.#levels.clear()
    this.#pickables.clear()
    for (const level of LEVELS) this.#levels.set(level, svgEl("g", { class: "ctx-level is-hidden" }, stageGroup))
    this.#buildGrid(geometry, path)
    this.#buildCta(geometry, path)
    this.#buildWarp(geometry, path)
    this.#buildLane(geometry)

    this.#pathLabels = LEVELS.map(() => label(root, "", { x: stage.x, y: geometry.pathY }))
    this.#sectionHeadings.clear()
    this.#sectionLines.clear()
    for (const section of geometry.panel) {
      this.#sectionHeadings.set(section.level, text(root, section.heading))
      this.#sectionLines.set(
        section.level,
        section.lineYs.map(y => label(root, "", { x: section.x + 10, y, className: "is-small" }))
      )
    }
    this.#shownLevel = null
  }

  #buildGrid(geometry: HierarchyGeometry, path: ThreadPath): void {
    const group = this.#levels.get("grid") as SVGGElement
    const { grid } = geometry
    for (const at of [...grid.columnLabels, ...grid.rowLabels, ...grid.axes]) text(group, at)
    const pickables: SVGElement[] = []
    this.#warpLines = grid.warpLines.map(row => row.map(() => []))
    this.#blockBoxes = grid.boxes.map((row, y) =>
      row.map((at, x) => {
        const cell = box(group, { ...at })
        const lines = svgEl("g", { class: "chart-cells" }, group)
        this.#warpLines[y][x] = grid.warpLines[y][x].map(line => svgEl("rect", { class: "chart-cell", ...line }, lines))
        text(group, { x: at.x + 6, y: at.y + 11, text: `(${x}, ${y})`, small: true })
        const hit = svgEl("rect", { class: "chart-hit", ...at }, group)
        makePickable(hit, {
          group: "block",
          value: `${x},${y}`,
          label: `Block (${x}, ${y})`,
          row: y,
          col: x,
          pressed: x === path.block.x && y === path.block.y,
        })
        pickables.push(hit)
        return cell
      })
    )
    this.#pickables.set("grid", pickables)
  }

  #buildCta(geometry: HierarchyGeometry, path: ThreadPath): void {
    const group = this.#levels.get("cta") as SVGGElement
    const { cta } = geometry
    for (const at of cta.axis) text(group, at)
    this.#warpLabels = cta.rowLabels.map(at => text(group, at))
    const cellGroup = svgEl("g", { class: "chart-cells" }, group)
    this.#warpCells = cta.cells.map(row => row.map(at => svgEl("rect", { class: "chart-cell", ...at }, cellGroup)))
    const pickables: SVGElement[] = []
    cta.rows.forEach((at, warp) => {
      const hit = svgEl(
        "rect",
        { class: "chart-hit", x: at.x - 2, y: at.y - 3, width: at.width + 4, height: at.height + 6 },
        group
      )
      makePickable(hit, {
        group: "warp",
        value: String(warp),
        label: `Warp ${warp}`,
        row: warp,
        col: 0,
        pressed: warp === path.warp,
      })
      pickables.push(hit)
    })
    this.#pickables.set("cta", pickables)
  }

  #buildWarp(geometry: HierarchyGeometry, path: ThreadPath): void {
    const group = this.#levels.get("warp") as SVGGElement
    const { warp } = geometry
    this.#warpHeading = text(group, warp.heading)
    for (const at of warp.rowLabels) text(group, at)
    const pickables: SVGElement[] = []
    this.#laneCells = []
    this.#laneNumbers = []
    warp.cells.forEach((at, lane) => {
      const cell = box(group, { ...at })
      this.#laneCells.push(cell)
      this.#laneNumbers.push(
        text(group, {
          x: at.x + at.width / 2,
          y: at.y + at.height / 2,
          text: String(lane),
          anchor: "middle",
          small: true,
        })
      )
      makePickable(cell, {
        group: "lane",
        value: String(lane),
        label: `Lane ${lane}`,
        row: Math.floor(lane / 16),
        col: lane % 16,
        pressed: lane === path.lane,
      })
      pickables.push(cell)
    })
    this.#pickables.set("warp", pickables)
  }

  #buildLane(geometry: HierarchyGeometry): void {
    const group = this.#levels.get("lane") as SVGGElement
    const { lane } = geometry
    box(group, { ...lane.box, space: "reg", className: "is-on" })
    this.#laneLines = lane.lines.map(at => text(group, at))
    const registers = svgEl("g", { class: "chart-cells sp-reg" }, group)
    for (const at of lane.registers) svgEl("rect", { class: "chart-cell is-on", ...at }, registers)
    text(group, lane.registersLabel)
    this.#pickables.set("lane", [])
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
      this.motion.drawIn(this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks"))
      this.motion.enter([...this.#sectionHeadings.values(), ...[...this.#sectionLines.values()].flat()], {
        delayMs: 160,
      })
    }
    super.onFirstVisible()
  }

  override onChange(group: string): void {
    const next: Record<string, Level> = { block: "cta", warp: "warp", lane: "lane" }
    if (group in next) this.setState({ level: next[group] })
    else if (group !== "level") return
    this.#timeline = this.#makeTimeline(true)
    this.play()
  }

  #makeTimeline(animated: boolean): Timeline {
    const target = this.#level
    const from = this.#shownLevel ?? target
    const path = this.#path
    const steps: TimelineStep[] = [{ durationMs: 0, apply: () => this.#applyPath(path, target) }]
    const durationMs = animated ? this.motion.tokens.dur3Ms : 0
    let index = LEVELS.indexOf(from)
    const end = LEVELS.indexOf(target)
    while (index !== end) {
      const inward = end > index
      const outer = LEVELS[inward ? index : index - 1]
      const inner = LEVELS[inward ? index + 1 : index]
      steps.push({
        durationMs,
        apply: progress => this.#zoom(outer, inner, easeInOut(inward ? progress : 1 - progress)),
      })
      index += inward ? 1 : -1
    }
    steps.push({ durationMs: 0, apply: () => this.#show(target) })
    return new Timeline(steps)
  }

  // amount 0 shows the outer level whole, 1 the inner level whole.
  #zoom(outer: Level, inner: Level, amount: number): void {
    const geometry = this.#geometry
    if (!geometry) return
    const { stage } = geometry
    const target = fitAspect(this.#pickedRect(outer, geometry), stage)
    const window = windowAt(stage, target, amount)
    const outerGroup = this.#levels.get(outer)
    const innerGroup = this.#levels.get(inner)
    for (const [level, group] of this.#levels) {
      if (level !== outer && level !== inner) group.classList.add("is-hidden")
    }
    outerGroup?.setAttribute("transform", transformAttribute(affineFor(window, stage)))
    innerGroup?.setAttribute("transform", transformAttribute(affineFor(mapRect(window, target, stage), stage)))
    outerGroup?.classList.toggle("is-hidden", amount >= FADE_OUT_AT)
    innerGroup?.classList.toggle("is-hidden", amount < FADE_IN_AT)
  }

  #show(level: Level): void {
    const active = typeof document === "undefined" ? null : document.activeElement
    const wasFocused = active !== null && this.#pickables.get(this.#shownLevel ?? level)?.includes(active as SVGElement)
    for (const [each, group] of this.#levels) {
      group.classList.toggle("is-hidden", each !== level)
      group.removeAttribute("transform")
    }
    this.#shownLevel = level
    if (wasFocused) {
      const stop = this.#pickables.get(level)?.find(cell => cell.getAttribute("tabindex") === "0")
      stop?.focus()
    }
  }

  #pickedRect(level: Level, geometry: HierarchyGeometry): Rect {
    const path = this.#path
    if (level === "grid") return geometry.grid.boxes[path.block.y][path.block.x]
    if (level === "cta") return geometry.cta.rows[path.warp]
    return geometry.warp.cells[path.lane]
  }

  #applyPath(path: ThreadPath, level: Level): void {
    this.#blockBoxes.forEach((row, y) =>
      row.forEach((cell, x) => {
        const isBlock = x === path.block.x && y === path.block.y
        cell.classList.toggle("is-hot", isBlock)
        this.#warpLines[y][x].forEach((line, warp) => line.classList.toggle("is-on", isBlock && warp === path.warp))
      })
    )
    this.#warpCells.forEach((row, warp) =>
      row.forEach((cell, lane) => {
        const isThread = warp === path.warp && lane === path.lane
        cell.classList.toggle("is-on", warp === path.warp && !isThread)
        cell.classList.toggle("is-hot", isThread)
        cell.classList.toggle("sp-reg", isThread)
      })
    )
    this.#warpLabels.forEach((warpLabel, warp) => warpLabel.classList.toggle("is-hot", warp === path.warp))
    this.#laneCells.forEach((cell, lane) => {
      cell.classList.toggle("is-hot", lane === path.lane)
      cell.classList.toggle("sp-reg", lane === path.lane)
    })
    this.#laneNumbers.forEach((number, lane) => number.classList.toggle("is-hot", lane === path.lane))
    const firstThread = path.warp * LANES
    if (this.#warpHeading) {
      this.#warpHeading.textContent = `warp ${path.warp}: threads ${firstThread} to ${firstThread + LANES - 1}`
    }
    const laneText = [
      `thread ${path.thread}`,
      `lane ${path.lane} of warp ${path.warp}`,
      `in block (${path.block.x}, ${path.block.y})`,
    ]
    this.#laneLines.forEach((line, index) => {
      line.textContent = laneText[index]
      line.classList.toggle("is-hot", index === 0)
    })
    for (const [each, cells] of this.#pickables) {
      for (const cell of cells) {
        const pressed = cell.getAttribute("aria-pressed") === "true"
        cell.setAttribute("aria-pressed", String(this.#isPicked(each, cell, path)))
        if (pressed !== this.#isPicked(each, cell, path) && this.#isPicked(each, cell, path)) {
          for (const other of cells) other.setAttribute("tabindex", other === cell ? "0" : "-1")
        }
      }
    }

    this.#placePath(path, level)
    const lines = panelLines(path)
    for (const each of LEVELS) {
      const isLevel = each === level
      this.#sectionHeadings.get(each)?.classList.toggle("is-hot", isLevel)
      this.#sectionLines.get(each)?.forEach((line, index) => {
        line.textContent = lines[each][index]
        line.classList.toggle("is-on", isLevel)
      })
    }
    this.setReadout(`${describeThread(path)}. ${LEVEL_NOTES[level]}`)
  }

  #isPicked(level: Level, cell: Element, path: ThreadPath): boolean {
    const value = cell.getAttribute("data-value")
    if (level === "grid") return value === `${path.block.x},${path.block.y}`
    if (level === "cta") return value === String(path.warp)
    return value === String(path.lane)
  }

  #placePath(path: ThreadPath, level: Level): void {
    const geometry = this.#geometry
    if (!geometry) return
    const segments = ["grid", `block (${path.block.x}, ${path.block.y})`, `warp ${path.warp}`, `lane ${path.lane}`]
    const depth = LEVELS.indexOf(level)
    let x = geometry.stage.x
    this.#pathLabels.forEach((pathLabel, index) => {
      const content = index === 0 ? segments[0] : `/ ${segments[index]}`
      pathLabel.textContent = content
      pathLabel.setAttribute("x", String(Math.round(x * 100) / 100))
      pathLabel.classList.toggle("is-hot", index === depth)
      pathLabel.classList.toggle("is-idle", index > depth)
      x += textWidthUnits(content) + PATH_GAP_UNITS
    })
  }
}

const LEVEL_NOTES: Record<Level, string> = {
  grid: `The grid: ${GRID_X} x ${GRID_Y} CTAs, ctx.blocks.x and .y; ctx.block.x and .y pick one.`,
  cta: `One CTA: ${WARPS} warps of ${LANES} threads; ctx.warp picks one.`,
  warp: `One warp: ${LANES} lanes that execute together; ctx.lane picks one.`,
  lane: "One lane: a thread with its own registers; ctx.thread = ctx.warp * ctx.lanes + ctx.lane.",
}

function text(parent: Element, at: TextAt): SVGTextElement {
  return label(parent, at.text, {
    x: at.x,
    y: at.y,
    anchor: at.anchor ?? "start",
    className: at.small ? "is-small" : undefined,
  })
}
