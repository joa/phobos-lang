// Layout of the activation figure, pure so the tests can check it: the library's activation
// functions in four families, each plotted over x in [-4, 4] from the shipped source through
// `evaluator`, on axes whose range is the family's own, with a legend naming each curve.

import type { Layout } from "../../../lib/figure/layout"
import { evaluator } from "../../../lib/stdlib/library"
import { LIBRARY, libraryFunction } from "../../../lib/stdlib/sources"
import type { Rect, TextAt } from "../../inference/figure-geometry"

export interface Point {
  x: number
  y: number
}

export interface Curve {
  // As called: `leaky_relu(x, 0.1)`.
  call: string
  // The library's definition, for the readout.
  formula: string
  // The curve split where it leaves the plot.
  segments: Point[][]
  legend: TextAt
}

export interface Family {
  key: string
  title: string
  curves: Curve[]
  yTicks: { y: number; label: TextAt }[]
  // Where y = 0 is, for the x axis.
  zeroY: number
}

export interface ActivationGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  plot: Rect
  xTicks: { x: number; label: TextAt }[]
  zeroX: number
  families: Family[]
}

interface Member {
  name: string
  // Arguments after `x`.
  extra?: number[]
}

// The families the figure shows, each a set of functions that read best on one pair of axes.
export const FAMILIES: { key: string; title: string; members: Member[] }[] = [
  {
    key: "rectifiers",
    title: "Rectifiers",
    members: [{ name: "relu" }, { name: "leaky_relu", extra: [0.1] }, { name: "elu", extra: [1] }, { name: "selu" }],
  },
  {
    key: "gelu",
    title: "GELU and its kin",
    members: [
      { name: "gelu" },
      { name: "gelu_tanh" },
      { name: "quick_gelu" },
      { name: "silu" },
      { name: "mish" },
      { name: "hardswish" },
    ],
  },
  {
    key: "squash",
    title: "Squashing",
    members: [{ name: "sigmoid" }, { name: "hardsigmoid" }, { name: "softsign" }, { name: "hardtanh" }],
  },
  {
    key: "smooth",
    title: "Smooth",
    members: [{ name: "softplus" }, { name: "log_sigmoid" }, { name: "squared_relu" }],
  },
]

export const X_RANGE: [number, number] = [-4, 4]
const SAMPLES = 161
// The y range a family may take at most, so a curve that grows fast leaves the plot instead of
// flattening the others.
const Y_CAP: [number, number] = [-4.5, 4.5]
const LEGEND_PITCH_UNITS = 22

interface Spec {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  plot: Rect
  legend: { x: number; y: number; columns: number; columnUnits: number }
}

const SPECS: Record<Layout, Spec> = {
  wide: {
    widthUnits: 720,
    heightUnits: 320,
    plateLabel: "FIG / STDLIB / ACTIVATIONS",
    plot: { x: 64, y: 52, width: 432, height: 236 },
    legend: { x: 532, y: 64, columns: 1, columnUnits: 170 },
  },
  narrow: {
    widthUnits: 400,
    heightUnits: 440,
    plateLabel: "FIG / ACTIVATIONS",
    plot: { x: 52, y: 52, width: 324, height: 236 },
    legend: { x: 24, y: 338, columns: 2, columnUnits: 186 },
  },
}

// A step of 0.5, 1 or 2 that gives at most six ticks over a span.
function tickStep(span: number): number {
  for (const step of [0.5, 1, 2]) if (span / step <= 6) return step
  return 4
}

export function formulaOf(name: string): string {
  const fn = libraryFunction(name)
  const body = fn.source.slice(fn.source.indexOf("=") + 1).trim()
  return `${name}(${fn.params.join(", ")}) = ${body}`
}

export function activationGeometry(layout: Layout): ActivationGeometry {
  const spec = SPECS[layout]
  const { plot } = spec
  const f = evaluator(LIBRARY)
  const xOf = (x: number) => plot.x + ((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])) * plot.width
  const xs = Array.from({ length: SAMPLES }, (_, i) => X_RANGE[0] + ((X_RANGE[1] - X_RANGE[0]) * i) / (SAMPLES - 1))

  const families = FAMILIES.map(({ key, title, members }) => {
    const values = members.map(member => xs.map(x => f(member.name, x, ...(member.extra ?? []))))
    const all = values.flat()
    const yMin = Math.max(Y_CAP[0], Math.floor(Math.min(...all) * 2) / 2)
    const yMax = Math.min(Y_CAP[1], Math.ceil(Math.max(...all) * 2) / 2)
    const yOf = (y: number) => plot.y + plot.height - ((y - yMin) / (yMax - yMin)) * plot.height
    const curves = members.map((member, index): Curve => {
      const segments: Point[][] = []
      let current: Point[] = []
      xs.forEach((x, i) => {
        const y = values[index][i]
        if (y < yMin || y > yMax) {
          if (current.length > 1) segments.push(current)
          current = []
        } else current.push({ x: round(xOf(x)), y: round(yOf(y)) })
      })
      if (current.length > 1) segments.push(current)
      const column = index % spec.legend.columns
      const row = Math.floor(index / spec.legend.columns)
      const call = `${member.name}(${["x", ...(member.extra ?? []).map(String)].join(", ")})`
      return {
        call,
        formula: formulaOf(member.name),
        segments,
        legend: {
          x: spec.legend.x + column * spec.legend.columnUnits,
          y: spec.legend.y + row * LEGEND_PITCH_UNITS,
          text: call,
          sizeUnits: 12.5,
        },
      }
    })
    const step = tickStep(yMax - yMin)
    const yTicks = []
    for (let y = Math.ceil(yMin / step) * step; y <= yMax + 1e-9; y += step) {
      const at = round(yOf(y))
      yTicks.push({ y: at, label: { x: plot.x - 10, y: at, text: fmt(y), anchor: "end" as const, sizeUnits: 12.5 } })
    }
    return { key, title, curves, yTicks, zeroY: round(yOf(0)) }
  })

  const xTicks = [-4, -2, 0, 2, 4].map(x => {
    const at = round(xOf(x))
    return {
      x: at,
      label: { x: at, y: plot.y + plot.height + 16, text: fmt(x), anchor: "middle" as const, sizeUnits: 12.5 },
    }
  })

  return {
    widthUnits: spec.widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: spec.plateLabel,
    plot,
    xTicks,
    zeroX: round(xOf(0)),
    families,
  }
}

function fmt(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}
