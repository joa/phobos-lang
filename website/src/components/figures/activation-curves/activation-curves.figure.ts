// The standard library's activation functions, plotted from their source. A family's curves draw in
// one at a time, each lit with its definition in the readout, then all of them rest together. The
// curves are in the registers' hue: an activation runs on each element where its thread holds it.

import { ChartFigure } from "../../../lib/figure/chart-figure"
import { label, plate } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { svgEl } from "../../../lib/figure/svg"
import { Timeline } from "../../../lib/figure/timeline"
import { activationGeometry, FAMILIES, type ActivationGeometry } from "./geometry"

const FOCUS_MS = 1100

interface Drawn {
  group: SVGGElement
  curves: SVGGElement[]
  legends: SVGTextElement[]
}

export default class ActivationCurvesFigure extends ChartFigure {
  #geometry: ActivationGeometry | null = null
  #families = new Map<string, Drawn>()
  #timeline: Timeline | null = null

  override get tour() {
    return FAMILIES.map(family => ({ family: family.key }))
  }

  get #family(): string {
    const key = String(this.state.family)
    return FAMILIES.some(family => family.key === key) ? key : FAMILIES[0].key
  }

  override build(layout: Layout): void {
    const geometry = activationGeometry(layout)
    this.#geometry = geometry
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    const root = this.drawing
    plate(root, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: geometry.plateLabel,
    })
    const { plot } = geometry
    svgEl("rect", { class: "chart-box", ...plot }, root)
    svgEl("path", { class: "chart-ticks", d: `M${geometry.zeroX} ${plot.y}V${plot.y + plot.height}` }, root)
    for (const tick of geometry.xTicks) {
      const bottom = plot.y + plot.height
      svgEl("path", { class: "chart-ticks", d: `M${tick.x} ${bottom}V${bottom + 4}` }, root)
      label(root, tick.label.text, { x: tick.label.x, y: tick.label.y, anchor: "middle", className: "is-small" })
    }

    this.#families.clear()
    for (const family of geometry.families) {
      const group = svgEl("g", { class: "sp-reg" }, root)
      svgEl("path", { class: "chart-ticks", d: `M${plot.x} ${family.zeroY}H${plot.x + plot.width}` }, group)
      for (const tick of family.yTicks) {
        svgEl("path", { class: "chart-ticks", d: `M${plot.x - 4} ${tick.y}H${plot.x}` }, group)
        label(group, tick.label.text, { x: tick.label.x, y: tick.label.y, anchor: "end", className: "is-small" })
      }
      const curves = family.curves.map(curve => {
        const g = svgEl("g", { class: "chart-wire-group" }, group)
        for (const segment of curve.segments) {
          const d = segment.map((p, i) => `${i === 0 ? "M" : "L"}${p.x} ${p.y}`).join("")
          svgEl("path", { class: "chart-wire", d }, g)
        }
        return g
      })
      const legends = family.curves.map(curve =>
        label(group, curve.legend.text, { x: curve.legend.x, y: curve.legend.y, className: "is-small" })
      )
      this.#families.set(family.key, { group, curves, legends })
    }
  }

  override update(dtMs: number): boolean {
    if (!this.#timeline) return false
    const running = this.#timeline.advance(dtMs)
    if (!running) this.#timeline = null
    return running
  }

  override settle(): void {
    const timeline = this.#timeline ?? this.#makeTimeline()
    timeline.seekEnd()
    this.#timeline = null
  }

  override onChange(group: string, _value: string): void {
    if (group !== "family") return
    this.#run()
  }

  override onAction(action: string): void {
    if (action === "replay") this.#run()
  }

  #run(): void {
    this.#timeline = this.#makeTimeline()
    this.play()
  }

  #makeTimeline(): Timeline {
    const geometry = this.#geometry
    const family = this.#family
    const drawn = this.#families.get(family)
    const curves = geometry?.families.find(f => f.key === family)?.curves ?? []
    if (!geometry || !drawn) return new Timeline([])
    const show = (focus: number | null, upTo: number) => {
      for (const [key, other] of this.#families) other.group.classList.toggle("is-hidden", key !== family)
      drawn.curves.forEach((curve, i) => {
        curve.classList.toggle("is-hidden", i > upTo)
        curve.classList.toggle("is-on", focus === null || i === focus)
        curve.classList.toggle("is-idle", focus !== null && i !== focus)
      })
      drawn.legends.forEach((legend, i) => {
        legend.classList.toggle("is-hot", i === focus)
        legend.classList.toggle("is-on", focus === null)
        legend.classList.toggle("is-idle", i > upTo)
      })
    }
    const title = FAMILIES.find(f => f.key === family)?.title ?? family
    return new Timeline([
      ...curves.map((curve, i) => ({
        durationMs: FOCUS_MS,
        apply: () => {
          show(i, i)
          this.setReadout(curve.formula)
        },
      })),
      {
        durationMs: 0,
        apply: () => {
          show(null, curves.length - 1)
          this.setReadout(`${title}: ${curves.map(curve => curve.call).join(", ")}, plotted from the library's source.`)
        },
      },
    ])
  }
}
