// ctx.threads, ctx.warps and ctx.lanes: @launch(threads) as a range in steps of 32, the CTA drawn as
// warps of ctx.lanes threads. Changing the count regroups the CTA over --dur-3: the block recenters,
// warps fade in or out one after another and the ctx.warps bracket follows. @launch(48) shows the
// missing half warp as cost and the compiler's error. Schematic.

import { ChartFigure, type StateValue } from "../../../lib/figure/chart-figure"
import { label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { clamp, easeInOut, lerp } from "../../../lib/figure/math"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline } from "../../../lib/figure/timeline"
import { DEFAULT_THREADS, LANES, MAX_THREADS } from "../ctx-hierarchy/rules"
import { wrapText } from "../ctx-hierarchy/text"
import {
  CELL_PITCH_UNITS,
  CELL_UNITS,
  countsGeometry,
  countsModel,
  labeledRows,
  MAX_ROWS,
  ROW_PITCH_UNITS,
  type CountsGeometry,
  type CountsModel,
} from "./geometry"

const BRACKET_TICK_UNITS = 4

export default class CtxCountsFigure extends ChartFigure {
  #geometry: CountsGeometry | null = null
  #block: SVGGElement | null = null
  #rows: SVGGElement[] = []
  #cells: SVGRectElement[][] = []
  #rowLabels: SVGTextElement[] = []
  #warpBracket: SVGPathElement | null = null
  #warpBracketLabel: SVGTextElement | null = null
  #launchLabel: SVGTextElement | null = null
  #fieldNames: SVGTextElement[] = []
  #fieldNotes: SVGTextElement[] = []
  #errorLabel: SVGTextElement | null = null
  #errorLines: SVGTextElement[] = []
  #entering: SVGElement[] = []
  #labeled = new Set<number>()
  // What is drawn now, so a change that interrupts another starts from where it got to.
  #shownRows = 0
  #timeline: Timeline | null = null

  override get tour(): Record<string, StateValue>[] {
    return [{ threads: 128 }, { threads: 512 }, { launch: "48" }, { threads: DEFAULT_THREADS }]
  }

  get #model(): CountsModel {
    const threads = clamp(Number(this.state.threads ?? DEFAULT_THREADS), LANES, MAX_THREADS)
    return countsModel(threads, this.state.launch === "48")
  }

  override build(layout: Layout): void {
    const geometry = countsGeometry(layout)
    this.#geometry = geometry
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    const root = this.drawing
    plate(root, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: geometry.plateLabel,
      schematic: true,
    })

    this.#block = svgEl("g", { class: "ctx-counts-block" }, root)
    const blockWidth = (LANES - 1) * CELL_PITCH_UNITS + CELL_UNITS
    const lanesY = -8
    svgEl(
      "path",
      {
        class: "chart-wire",
        d: `M${geometry.cellsX} ${lanesY + BRACKET_TICK_UNITS}V${lanesY}H${geometry.cellsX + blockWidth}V${lanesY + BRACKET_TICK_UNITS}`,
      },
      this.#block
    )
    label(this.#block, `ctx.lanes = ${LANES}`, {
      x: geometry.cellsX + blockWidth / 2,
      y: lanesY - 12,
      anchor: "middle",
    })
    this.#rows = []
    this.#cells = []
    this.#rowLabels = []
    for (let row = 0; row < MAX_ROWS; row += 1) {
      const y = row * ROW_PITCH_UNITS
      const rowGroup = svgEl("g", { class: "chart-cells" }, this.#block)
      this.#rows.push(rowGroup)
      this.#cells.push(
        Array.from({ length: LANES }, (_, lane) =>
          svgEl(
            "rect",
            {
              class: "chart-cell",
              x: geometry.cellsX + lane * CELL_PITCH_UNITS,
              y,
              width: CELL_UNITS,
              height: CELL_UNITS,
            },
            rowGroup
          )
        )
      )
      this.#rowLabels.push(
        label(this.#block, `warp ${row}`, {
          x: geometry.warpLabelX,
          y: y + CELL_UNITS / 2,
          anchor: "end",
          className: "is-small",
        })
      )
    }

    this.#warpBracket = svgEl("path", { class: "chart-wire" }, root)
    this.#warpBracketLabel = label(root, "ctx.warps", { x: geometry.bracketX + 10, y: 0 })

    this.#launchLabel = label(root, "", { ...geometry.launch, className: "is-on" })
    this.#fieldNames = geometry.fields.map(at => label(root, "", at.name))
    this.#fieldNotes = geometry.fields.map(at =>
      label(root, "compile-time constant", { x: at.note.x, y: at.note.y, className: "is-small" })
    )
    this.#errorLabel = label(root, "compiler error", { ...geometry.errorLabel, className: "is-small" })
    this.#errorLines = geometry.errorLines.map(at => label(root, "", { ...at, className: "is-small" }))
    this.#entering = [this.#launchLabel, ...this.#fieldNames, ...this.#fieldNotes]
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
      this.motion.drawIn(this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire"))
      this.motion.enter(this.#entering, { delayMs: 240 })
    }
    super.onFirstVisible()
  }

  override onInput(param: string, value: number): void {
    if (param !== "threads") return
    if (this.state.launch === "48") this.setState({ launch: "ok" })
    this.#showCount(value)
    this.#run()
  }

  override onChange(group: string): void {
    if (group === "launch") this.#run()
  }

  #run(): void {
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  #showCount(threads: number): void {
    const output = this.root.querySelector?.("output[data-for='threads']")
    if (output) output.textContent = String(threads)
  }

  #makeTimeline(): Timeline {
    const model = this.#model
    const fromRows = this.#shownRows === 0 ? model.rows : this.#shownRows
    const durationMs = this.motion.tokens.dur3Ms
    return new Timeline([
      { durationMs: 0, apply: () => this.#applyModel(model) },
      {
        durationMs: fromRows === model.rows ? 0 : durationMs,
        apply: progress => this.#placeRows(lerp(fromRows, model.rows, easeInOut(progress))),
      },
    ])
  }

  #applyModel(model: CountsModel): void {
    const invalid = model.error !== null
    for (let row = 0; row < MAX_ROWS; row += 1) {
      for (let lane = 0; lane < LANES; lane += 1) {
        const state = model.cellState(row, lane)
        const cell = this.#cells[row][lane]
        cell.classList.toggle("is-on", state === "on")
        cell.classList.toggle("is-warn", state === "warn")
        cell.classList.toggle("sp-cost", state === "warn")
      }
    }
    this.#labeled = new Set(labeledRows(model.rows))

    if (this.#launchLabel) {
      this.#launchLabel.textContent = `@launch(${model.threads})`
      this.#launchLabel.classList.toggle("is-on", !invalid)
      this.#launchLabel.classList.toggle("is-warn", invalid)
    }
    const values = [
      `ctx.threads = ${model.threads}`,
      `ctx.lanes = ${LANES}`,
      invalid ? `ctx.warps = ${model.threads} / ${LANES}` : `ctx.warps = ${model.threads} / ${LANES} = ${model.warps}`,
    ]
    this.#fieldNames.forEach((name, index) => {
      name.textContent = values[index]
    })
    const warpsNote = this.#fieldNotes[2]
    warpsNote.textContent = invalid ? "not a whole number of warps" : "compile-time constant"
    warpsNote.classList.toggle("is-warn", invalid)
    this.#warpBracket?.classList.toggle("is-warn", invalid)
    this.#warpBracketLabel?.classList.toggle("is-warn", invalid)
    this.#errorLabel?.classList.toggle("is-hidden", !invalid)
    this.#errorLabel?.classList.toggle("is-warn", invalid)
    const geometry = this.#geometry
    const lines = geometry && model.error ? wrapText(model.error, geometry.errorMaxChars) : []
    this.#errorLines.forEach((line, index) => {
      line.textContent = lines[index] ?? ""
      line.classList.toggle("is-hidden", !invalid)
      line.classList.toggle("is-warn", invalid)
    })

    this.setReadout(
      invalid
        ? `@launch(${model.threads}) is rejected before any code is generated: ${model.error}`
        : `@launch(${model.threads}): ${model.threads} threads in ${model.warps} ${model.warps === 1 ? "warp" : "warps"} of ${LANES} lanes. All three counts are compile-time constants.`
    )
  }

  #placeRows(rows: number): void {
    const geometry = this.#geometry
    if (!geometry) return
    this.#shownRows = rows
    const top = geometry.blockTop(rows)
    this.#block?.setAttribute("transform", `translate(0 ${round(top)})`)
    const visible = Math.round(rows)
    this.#rows.forEach((row, index) => row.classList.toggle("is-hidden", index >= visible))
    const bottom = top + (rows - 1) * ROW_PITCH_UNITS + CELL_UNITS
    const x = geometry.bracketX
    this.#warpBracket?.setAttribute(
      "d",
      `M${x - BRACKET_TICK_UNITS} ${round(top)}H${x}V${round(bottom)}H${x - BRACKET_TICK_UNITS}`
    )
    this.#warpBracketLabel?.setAttribute("y", String(round((top + bottom) / 2)))
    this.#rowLabels.forEach((rowLabel, index) =>
      rowLabel.classList.toggle("is-hidden", index >= visible || !this.#labeled.has(index))
    )
  }
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
