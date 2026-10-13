// The cluster level above the grid, experimental: an output tensor split into super-tiles, each
// dealt to one node, each node running a grid of CTAs over its super-tile, and one CTA opened into
// warps and lanes. One run: super-tiles travel to their nodes, the grids run, the opened CTA's
// warps run. The figure base loops it.

import { ChartFigure } from "../../../lib/figure/chart-figure"
import { box, cellGrid, label, plate, wire } from "../../../lib/figure/draw"
import type { Layout } from "../../../lib/figure/layout"
import { StarPacket } from "../../../lib/figure/packet"
import { setAttributes } from "../../../lib/figure/svg"
import { Timeline, type TimelineStep } from "../../../lib/figure/timeline"
import {
  CTAS_PER_NODE,
  LANES,
  NODES,
  OPENED_CTA,
  OPENED_NODE,
  WARPS,
  clusterTilesGeometry,
  type ClusterTilesGeometry,
  type TextAt,
} from "./geometry"

const PACKET_SPEED_UNITS_PER_MS = 0.22
const PACKET_STAGGER_MS = 320
const GRID_MS = 1100
const WARPS_MS = 1100
const HOLD_MS = 1000

export const READOUT_START = "The scheduler deals the output's super-tiles to the nodes, one each here."
export const READOUT_END =
  "Each node runs the leaf as an ordinary grid of CTAs over its super-tile; each CTA runs as warps of lanes."

export default class ClusterTilesFigure extends ChartFigure {
  #geometry: ClusterTilesGeometry | null = null
  #bands: SVGRectElement[] = []
  #nodes: SVGRectElement[] = []
  #ctaCells: SVGRectElement[][] = []
  #cta: SVGRectElement | null = null
  #lanes: SVGRectElement[][] = []
  #links: SVGGElement[] = []
  #zoom: SVGGElement | null = null
  #packets: StarPacket[] = []
  #zoomPacket: StarPacket | null = null
  #labels: SVGTextElement[] = []
  #timeline: Timeline | null = null

  override build(layout: Layout): void {
    const geometry = clusterTilesGeometry(layout)
    this.#geometry = geometry
    this.setViewBox(geometry.widthUnits, geometry.heightUnits)
    const root = this.drawing
    plate(root, this.id, {
      widthUnits: geometry.widthUnits,
      heightUnits: geometry.heightUnits,
      label: "FIG / CLUSTER / SUPER-TILES",
      schematic: true,
    })
    this.#labels = []

    const cluster = box(root, { ...geometry.cluster, className: "cluster-experimental" })
    setAttributes(cluster, { "stroke-dasharray": "4 3" })

    this.#links = geometry.links.map(line => wire(root, line, { space: "global" }).group)
    this.#zoom = wire(root, geometry.zoom, { space: "reg" }).group

    this.#bands = geometry.bands.map(band => box(root, { ...band, space: "global" }))
    this.#nodes = geometry.nodes.map(node => {
      const rect = box(root, { ...node, space: "global", className: "cluster-experimental" })
      setAttributes(rect, { "stroke-dasharray": "4 3" })
      return rect
    })
    this.#ctaCells = geometry.ctaCells.map(
      cells => cellGrid(root, { ...cells, rows: 1, cols: CTAS_PER_NODE, space: "global" })[0]
    )
    this.#cta = box(root, { ...geometry.cta, space: "reg" })
    this.#lanes = cellGrid(root, { ...geometry.lanes, rows: WARPS, cols: LANES, space: "reg" })

    for (const at of geometry.headers) this.#text(at)
    for (const at of geometry.bandLabels) this.#text(at, layout === "narrow" ? "middle" : "start")
    for (const at of geometry.nodeLabels) this.#text(at)
    this.#text(geometry.ctaLabel)
    for (const at of geometry.notes) this.#text(at)

    this.#packets = geometry.links.map(() => new StarPacket(root, { space: "global" }))
    this.#zoomPacket = new StarPacket(root, { space: "reg" })
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
    if (this.reducedMotion) return
    const lines = this.drawing.querySelectorAll<SVGGeometryElement>(".chart-ticks, .chart-wire")
    this.motion.drawIn(lines)
    this.motion.enter([...this.#labels, ...this.drawing.querySelectorAll(".cluster-experimental")], {
      delayMs: 240,
    })
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
    if (!geometry) return new Timeline([])
    const travelMs = (index: number) => geometry.links[index].lengthUnits / PACKET_SPEED_UNITS_PER_MS
    const dealMs = (NODES - 1) * PACKET_STAGGER_MS + Math.max(...geometry.links.map((_, i) => travelMs(i)))
    const zoomMs = geometry.zoom.lengthUnits / PACKET_SPEED_UNITS_PER_MS
    const steps: TimelineStep[] = [
      { durationMs: 0, apply: () => this.#reset() },
      {
        durationMs: dealMs,
        apply: progress => {
          const elapsedMs = progress * dealMs
          geometry.links.forEach((line, i) => {
            const local = (elapsedMs - i * PACKET_STAGGER_MS) / travelMs(i)
            this.#packets[i]?.placeAtFraction(line, progress >= 1 ? 2 : local)
            this.#bands[i]?.classList.toggle("is-on", local >= 0)
            this.#links[i]?.classList.toggle("is-on", local >= 0)
            this.#nodes[i]?.classList.toggle("is-on", progress >= 1 || local >= 1)
          })
        },
      },
      {
        durationMs: GRID_MS,
        apply: progress => {
          const lit = progress * CTAS_PER_NODE
          for (const cells of this.#ctaCells)
            cells.forEach((cell, i) => {
              cell.classList.toggle("is-on", i < lit)
              cell.classList.toggle("is-hot", progress < 1 && i < lit && i >= lit - 1)
            })
        },
      },
      {
        durationMs: zoomMs,
        apply: progress => {
          this.#ctaCells[OPENED_NODE]?.[OPENED_CTA]?.classList.add("is-hot")
          this.#zoom?.classList.add("is-on")
          this.#zoomPacket?.placeAtFraction(geometry.zoom, progress >= 1 ? 2 : progress)
          this.#cta?.classList.toggle("is-on", progress >= 1)
        },
      },
      {
        durationMs: WARPS_MS,
        apply: progress => {
          const lit = progress * WARPS
          this.#lanes.forEach((row, warp) => {
            for (const lane of row) lane.classList.toggle("is-on", warp < lit)
          })
          if (progress >= 1) this.setReadout(READOUT_END)
        },
      },
    ]
    return new Timeline(steps, { holdMs: HOLD_MS })
  }

  #reset(): void {
    for (const element of [...this.#bands, ...this.#nodes, ...this.#links]) element.classList.remove("is-on")
    for (const cells of this.#ctaCells) for (const cell of cells) cell.classList.remove("is-on", "is-hot")
    for (const row of this.#lanes) for (const lane of row) lane.classList.remove("is-on")
    this.#cta?.classList.remove("is-on")
    this.#zoom?.classList.remove("is-on")
    for (const packet of this.#packets) packet.hide()
    this.#zoomPacket?.hide()
    this.setReadout(READOUT_START)
  }

  #text(at: TextAt, anchor: "start" | "middle" = "start"): void {
    this.#labels.push(
      label(this.drawing, at.text, { x: at.x, y: at.y, anchor, className: at.small ? "is-small" : undefined })
    )
  }
}
