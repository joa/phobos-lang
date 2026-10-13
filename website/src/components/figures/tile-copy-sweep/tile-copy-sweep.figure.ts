// A tile's life: var a = A[...] copies a slice into shared memory, spread over the CTA's threads, and
// a three-line program steps a through allocated, live and released on a ruler of shared-memory
// bytes, where the next tile takes its bytes. With let, a is a view of A and takes none. Hovering or
// focusing a run of the tile names the thread that moves it. Schematic.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { box, label, node, plate, wire, type Wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { clamp } from "../../../lib/figure/math"
import { sweepWire } from "../../../lib/figure/motion"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline } from "../../../lib/figure/timeline"
import { enablePicking, makePickable, setTabStop } from "../ctx-hierarchy/pick"
import {
  liveAfter,
  PROGRAM,
  RULER_BYTES,
  RUN,
  runOf,
  S_BYTES,
  SLICE_COL,
  SLICE_ROW,
  STEPS,
  TENSOR_COLS,
  TENSOR_ROWS,
  THREADS,
  TILE_BYTES,
  TILE_COLS,
  TILE_ROWS,
  tileCopyGeometry,
  type Allocation,
  type Bind,
  type Rect,
  type Step,
  type TileCopyGeometry,
} from "./geometry"

const FILL_MS = 720
const RELEASE_MS = 240
const S_FILL_MS = 420

export default class TileCopySweepFigure extends ChartFigure {
  #geometry: TileCopyGeometry | null = null
  #tensorCells: SVGRectElement[][] = []
  #slice: SVGRectElement | null = null
  #sliceLabel: SVGTextElement | null = null
  #wire: Wire | null = null
  #sweep: SVGPathElement | null = null
  #tileGroup: SVGGElement | null = null
  #tileLabel: SVGTextElement | null = null
  #tileCells: SVGRectElement[][] = []
  #hits: SVGRectElement[] = []
  #sCells: SVGRectElement[] = []
  #sLabel: SVGTextElement | null = null
  #sGroup: SVGGElement | null = null
  #markers: SVGCircleElement[] = []
  #lines: SVGTextElement[] = []
  #segments: Record<Allocation["name"], { rect: SVGRectElement; text: SVGTextElement }> | null = null
  #rulerNote: SVGTextElement | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [
      { step: "copy" },
      { step: "read" },
      { step: "reuse" },
      { bind: "let", step: "copy" },
      { step: "read" },
      { bind: "var", step: "read" },
    ]
  }

  get #bind(): Bind {
    return this.state.bind === "let" ? "let" : "var"
  }

  get #step(): Step {
    const step = String(this.state.step)
    return (STEPS as readonly string[]).includes(step) ? (step as Step) : "read"
  }

  get #thread(): number {
    return clamp(Math.floor(Number(this.state.thread ?? 0)) || 0, 0, THREADS - 1)
  }

  // Which tile, if any, the shared tile holds after the step.
  get #tileHolds(): "a" | "b" | null {
    if (this.#step === "reuse") return "b"
    return this.#bind === "var" ? "a" : null
  }

  override start(): void {
    super.start()
    enablePicking(this.svg, { pickOnMove: true })
  }

  override build(layout: Layout): void {
    const geometry = tileCopyGeometry(layout)
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

    const globalGroup = svgEl("g", { class: "sp-global" }, root)
    this.#addLabel(globalGroup, geometry.tensorLabel, `global  A  ${TENSOR_ROWS} x ${TENSOR_COLS} f32`)
    const tensorCellGroup = svgEl("g", { class: "chart-cells" }, globalGroup)
    this.#tensorCells = geometry.tensorCells.map(row => row.map(cell => rect(tensorCellGroup, cell, "chart-cell")))
    this.#slice = box(globalGroup, { ...geometry.slice, className: "is-on is-outline" })
    this.#sliceLabel = this.#addLabel(globalGroup, geometry.sliceLabel, "", "is-small")

    this.#wire = wire(root, geometry.wire, { space: "global" })
    this.#sweep = sweepWire(root, geometry.wire.toPathData(), "global")

    this.#tileGroup = svgEl("g", { class: "sp-shared" }, root)
    this.#tileLabel = this.#addLabel(this.#tileGroup, geometry.tileLabel)
    const tileCellGroup = svgEl("g", { class: "chart-cells" }, this.#tileGroup)
    this.#tileCells = geometry.tileCells.map(row => row.map(cell => rect(tileCellGroup, cell, "chart-cell")))
    const thread = this.#thread
    this.#hits = geometry.runHits.map((at, index) => {
      const hit = rect(this.#tileGroup as SVGGElement, at, "chart-hit")
      const { row, col } = runOf(index)
      makePickable(hit, {
        group: "thread",
        value: String(index),
        label: `thread ${index}, a[${row}, ${col}:${col + RUN}]`,
        row,
        col: col / RUN,
        pressed: index === thread,
      })
      return hit
    })
    const sGroup = svgEl("g", { class: "sp-shared" }, root)
    this.#sGroup = sGroup
    this.#sLabel = this.#addLabel(sGroup, geometry.sLabel, "s")
    this.#sLabel.setAttribute("text-anchor", "middle")
    const sCellGroup = svgEl("g", { class: "chart-cells" }, sGroup)
    this.#sCells = geometry.sCells.map(cell => rect(sCellGroup, cell, "chart-cell"))

    this.#markers = geometry.program.map(at => node(root, at.marker.x, at.marker.y, { className: "is-end" }))
    this.#lines = geometry.program.map(at => this.#addLabel(root, at.text))

    const { ruler } = geometry
    this.#addLabel(root, ruler.label, "shared memory, bytes", "is-small")
    const rulerGroup = svgEl("g", { class: "sp-shared" }, root)
    const segment = () => {
      const rectElement = svgEl(
        "rect",
        { class: "chart-box", y: ruler.segmentY, height: ruler.segmentHeight },
        rulerGroup
      )
      const text = label(rulerGroup, "", { x: 0, y: ruler.segmentY + ruler.segmentHeight / 2, className: "is-small" })
      return { rect: rectElement, text }
    }
    this.#segments = { a: segment(), b: segment(), s: segment() }
    svgEl(
      "path",
      {
        class: "chart-wire",
        d: `M${ruler.x} ${ruler.axisY}H${ruler.x + ruler.width}${[0, 256, 512, 768, 1024]
          .map(bytes => `M${round(geometry.rulerX(bytes))} ${ruler.axisY}v4`)
          .join("")}`,
      },
      root
    )
    for (const bytes of [0, 512, 1024])
      this.#addLabel(
        root,
        { x: geometry.rulerX(bytes), y: ruler.tickY },
        bytes === RULER_BYTES ? `${bytes} B` : String(bytes),
        "is-small",
        bytes === 0 ? "start" : bytes === RULER_BYTES ? "end" : "middle"
      )
    this.#rulerNote = label(root, "", {
      x: geometry.rulerX(RULER_BYTES / 2),
      y: ruler.segmentY + ruler.segmentHeight / 2,
      anchor: "middle",
      className: "is-small is-idle",
    })
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
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion) {
      this.motion.drawIn(this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire, .chart-box"))
      this.motion.enter(this.#labels, { delayMs: 240 })
    }
    super.onFirstVisible()
  }

  override onChange(group: string): void {
    if (group === "thread") {
      this.#applyPick()
      this.setReadout(this.#pickReadout())
      return
    }
    if (group === "bind" || group === "step") {
      this.#timeline = this.#makeTimeline()
      this.play()
    }
  }

  override onAction(action: string): void {
    if (action === "replay") this.onChange("step")
  }

  #makeTimeline(): Timeline {
    const bind = this.#bind
    const step = this.#step
    const holds = this.#tileHolds
    const fills = (bind === "var" && step === "copy") || step === "reuse"
    return new Timeline([
      { durationMs: 0, apply: () => this.#applyScene() },
      {
        durationMs: step === "reuse" && bind === "var" ? RELEASE_MS : 0,
        apply: progress => {
          // a's last use has passed: its cells go dark before b lands in the same bytes.
          if (progress < 1) this.#fillTile(0)
        },
      },
      {
        durationMs: fills ? FILL_MS : 0,
        apply: progress => {
          this.#sweep?.classList.toggle("is-on", fills && holds !== "b" && progress < 1)
          this.#fillTile(holds === null ? 0 : fills ? progress * THREADS : THREADS)
        },
      },
      {
        durationMs: bind === "var" && step === "read" ? S_FILL_MS : 0,
        apply: progress => {
          const lit = bind === "var" && step !== "copy" ? Math.ceil(progress * TILE_ROWS) : 0
          this.#sCells.forEach((cell, row) => cell.classList.toggle("is-on", row < lit))
        },
      },
    ])
  }

  #applyScene(): void {
    const geometry = this.#geometry
    if (!geometry) return
    const bind = this.#bind
    const step = this.#step
    const stepIndex = STEPS.indexOf(step)
    const holds = this.#tileHolds
    const isView = bind === "let"
    const sliceActive = step !== "reuse"

    for (let row = 0; row < TENSOR_ROWS; row += 1) {
      for (let col = 0; col < TENSOR_COLS; col += 1) {
        const inSlice =
          row >= SLICE_ROW && row < SLICE_ROW + TILE_ROWS && col >= SLICE_COL && col < SLICE_COL + TILE_COLS
        this.#tensorCells[row][col].classList.toggle("is-on", inSlice && sliceActive)
      }
    }
    this.#slice?.classList.toggle("is-hot", isView && sliceActive)
    this.#slice?.classList.toggle("is-idle", !sliceActive)
    if (this.#sliceLabel) {
      this.#sliceLabel.textContent = isView
        ? "a: a view, no copy"
        : `slice A[${SLICE_ROW}:${SLICE_ROW + TILE_ROWS}, ${SLICE_COL}:${SLICE_COL + TILE_COLS}]`
      this.#sliceLabel.classList.toggle("is-on", isView && sliceActive)
    }
    this.#wire?.group.classList.toggle("is-on", holds === "a")
    this.#wire?.group.classList.toggle("is-idle", holds !== "a")

    this.#tileGroup?.classList.toggle("is-idle", holds === null)
    if (this.#tileLabel) {
      this.#tileLabel.textContent =
        holds === "a"
          ? `shared  a  ${TILE_ROWS} x ${TILE_COLS} f32, ${TILE_BYTES} B`
          : holds === "b"
            ? isView
              ? `shared  b  ${TILE_ROWS} x ${TILE_COLS} f32, from byte 0`
              : "shared  b  in a's bytes"
            : "shared  no tile: a is a view"
    }
    this.#tileLabel?.classList.toggle("is-on", holds !== null)
    this.#sGroup?.classList.toggle("is-hidden", isView || step === "copy")

    PROGRAM[bind].forEach((text, line) => {
      const lineLabel = this.#lines[line]
      lineLabel.textContent = text
      lineLabel.classList.toggle("is-on", line === stepIndex)
      lineLabel.classList.toggle("is-idle", line > stepIndex)
      this.#markers[line].classList.toggle("is-on", line === stepIndex)
      this.#markers[line].classList.toggle("is-hidden", line !== stepIndex)
    })

    const live = liveAfter(bind, step)
    const segmentText: Record<Allocation["name"], string> = {
      a: step === "copy" ? "a  allocated, live" : "a  live, last use",
      s: "s",
      b: isView ? "b" : "b  in a's bytes",
    }
    for (const name of ["a", "s", "b"] as const) {
      const segment = this.#segments?.[name]
      if (!segment) continue
      const alloc = live.find(entry => entry.name === name)
      const shown: Allocation = alloc ?? {
        name,
        offsetBytes: name === "s" ? TILE_BYTES : 0,
        sizeBytes: name === "s" ? S_BYTES : TILE_BYTES,
      }
      const x = geometry.rulerX(shown.offsetBytes)
      const width = geometry.rulerX(shown.offsetBytes + shown.sizeBytes) - x
      segment.rect.setAttribute("x", String(round(x)))
      segment.rect.setAttribute("width", String(round(width)))
      segment.rect.classList.toggle("is-on", alloc !== undefined)
      segment.rect.classList.toggle("is-hidden", alloc === undefined)
      segment.text.textContent = segmentText[name]
      // s is too narrow to hold its name; it sits just past the segment.
      segment.text.setAttribute("x", String(round(name === "s" ? x + width + 4 : x + 6)))
      segment.text.classList.toggle("is-on", alloc !== undefined)
      segment.text.classList.toggle("is-hidden", alloc === undefined)
    }
    if (this.#rulerNote) {
      this.#rulerNote.textContent = live.length === 0 ? "a view: 0 bytes" : ""
      this.#rulerNote.setAttribute("x", String(round(geometry.rulerX(RULER_BYTES / 4))))
    }

    this.#applyPick()
    this.setReadout(this.#sceneReadout())
  }

  // Lights the first `runs` runs of the tile, in thread order.
  #fillTile(runs: number): void {
    for (let thread = 0; thread < THREADS; thread += 1) {
      const { row, col } = runOf(thread)
      for (let index = 0; index < RUN; index += 1)
        this.#tileCells[row][col + index].classList.toggle("is-on", thread < runs)
    }
  }

  #applyPick(): void {
    const thread = this.#thread
    const holds = this.#tileHolds
    const { row, col } = runOf(thread)
    for (let r = 0; r < TILE_ROWS; r += 1)
      for (let c = 0; c < TILE_COLS; c += 1)
        this.#tileCells[r][c].classList.toggle("is-hot", holds !== null && r === row && c >= col && c < col + RUN)
    for (let r = 0; r < TENSOR_ROWS; r += 1) {
      for (let c = 0; c < TENSOR_COLS; c += 1) {
        const source = holds === "a" && r === SLICE_ROW + row && c >= SLICE_COL + col && c < SLICE_COL + col + RUN
        this.#tensorCells[r][c].classList.toggle("is-hot", source)
      }
    }
    setTabStop(this.#hits, this.#hits[thread])
  }

  #pickReadout(): string {
    const thread = this.#thread
    const holds = this.#tileHolds
    const { row, col } = runOf(thread)
    const run = `${col}:${col + RUN}`
    if (holds === null) return "a is a view: no thread copies it, and the tile holds nothing yet."
    const source =
      holds === "a"
        ? `A[${SLICE_ROW + row}, ${SLICE_COL + col}:${SLICE_COL + col + RUN}]`
        : `B[${SLICE_ROW + row}, ${run}]`
    return `Thread ${thread} moves ${source} into ${holds}[${row}, ${run}]: ${RUN} adjacent f32, one 16-byte load.`
  }

  #sceneReadout(): string {
    const bind = this.#bind
    const step = this.#step
    if (bind === "let") {
      if (step === "copy")
        return "let binds a view: a names the slice of A in place. Nothing is copied, and a takes no shared memory."
      if (step === "read") return "a * 2.0 reads the slice through the view, once. a has no bytes of its own."
      return `var b is the first tile in shared memory, so it starts at byte 0: ${TILE_BYTES} bytes in all.`
    }
    if (step === "copy")
      return `var a allocates ${TILE_BYTES} bytes of shared memory and copies the slice into them: ${THREADS} threads, each moving ${RUN} adjacent f32 with one 16-byte load.`
    if (step === "read")
      return `rowsum(a) is a's last use: a lives from its declaration to here. s takes the next ${S_BYTES} bytes.`
    return `a is released after its last use, so b takes its bytes. The footprint is the most bytes alive at once, ${TILE_BYTES + S_BYTES}, not the sum of all three tiles, ${2 * TILE_BYTES + S_BYTES}.`
  }

  #addLabel(
    parent: Element,
    at: { x: number; y: number },
    text = "",
    className?: string,
    anchor: "start" | "middle" | "end" = "start"
  ): SVGTextElement {
    const element = label(parent, text, { x: at.x, y: at.y, anchor, className })
    this.#labels.push(element)
    return element
  }
}

function rect(parent: Element, at: Rect, className: string): SVGRectElement {
  return svgEl("rect", { class: className, x: at.x, y: at.y, width: at.width, height: at.height }, parent)
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
