// What every station of the compiler scene is made of: SVG shapes, drawn at build time and animated by
// CSS once the station is on, and the targets of the point cloud the canvas draws behind them. Pure.

import type { Point } from "../../../lib/figure/polyline"

export type Layout = "wide" | "narrow"

/** Points in the cloud: one per PTX instruction, the most any station needs. */
export const PARTICLES = 448

export interface Frame {
  width: number
  height: number
}

export const FRAMES: Record<Layout, Frame> = {
  wide: { width: 1200, height: 600 },
  narrow: { width: 400, height: 640 },
}

/** The canvas's colors, one per token; `line` is --sky-line, `hi` --brand-hi. */
export type Tone = "global" | "shared" | "reg" | "cost" | "dust" | "line" | "hi"

export const TONES: Tone[] = ["global", "shared", "reg", "cost", "dust", "line", "hi"]

/** Mono glyph advance per unit of font size, for Noto Sans Mono. */
export const CHAR_EM = 0.6

export function textWidth(text: string, fontUnits: number): number {
  return text.length * fontUnits * CHAR_EM
}

export interface Shape {
  tag: "path" | "rect" | "circle" | "text" | "line"
  attrs: Record<string, string | number>
  text?: string
  /** For text: runs with their own class, in place of `text`. */
  spans?: { text: string; cls: string }[]
  /** Start of the shape's entrance after its station comes on, in ms. */
  delayMs?: number
  /** A second time for shapes with a later change (a buffer retiring), in ms. */
  laterMs?: number
}

export interface Target {
  x: number
  y: number
  /** Depth, units toward the back; small at rest so the cloud registers on the drawing. */
  z: number
  tone: Tone
  /** Depth layer of the particle, 0 far and small to 2 near and large. */
  size: 0 | 1 | 2
  /** 0 hides the point at this station. */
  alpha: number
  /** For a point that flows: its stream and where along it (0 to 1) it is when the station comes on. */
  stream?: number
  phase?: number
  /** A station's own grouping, such as the warp a point lands in. */
  group?: number
}

/** A path the cloud can flow along: operand to user in the graph. */
export interface Stream {
  points: Point[]
  tone: Tone
  /** When the path has drawn in and the flow may start, in ms after the station comes on. */
  startMs: number
}

export interface StationGeometry {
  frame: Frame
  shapes: Shape[]
  targets: Target[]
  /** Flow paths for the graph station; empty elsewhere. */
  streams: Stream[]
}

export function round(value: number): number {
  return Math.round(value * 10) / 10
}

export function text(
  x: number,
  y: number,
  content: string,
  className: string,
  options: { anchor?: "start" | "middle" | "end"; delayMs?: number; size?: number } = {}
): Shape {
  const attrs: Shape["attrs"] = { x: round(x), y: round(y), class: className }
  if (options.anchor && options.anchor !== "start") attrs["text-anchor"] = options.anchor
  if (options.size) attrs["font-size"] = options.size
  return { tag: "text", attrs, text: content, delayMs: options.delayMs }
}

export function target(x: number, y: number, tone: Tone, size: 0 | 1 | 2 = 1, z = 0, alpha = 1): Target {
  return { x: round(x), y: round(y), z: round(z), tone, size, alpha }
}

export function hidden(x: number, y: number): Target {
  return { x: round(x), y: round(y), z: 0, tone: "dust", size: 0, alpha: 0 }
}

/** A box's corners for overlap checks. */
export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export function overlaps(a: Rect, b: Rect, marginUnits = 0): boolean {
  return (
    a.x < b.x + b.width + marginUnits &&
    b.x < a.x + a.width + marginUnits &&
    a.y < b.y + b.height + marginUnits &&
    b.y < a.y + a.height + marginUnits
  )
}

/** The box a text shape covers, from its anchor, font size and length. */
export function textRect(shape: Shape, fontUnits: number): Rect {
  const width = textWidth(shape.text ?? "", fontUnits)
  const anchor = shape.attrs["text-anchor"] ?? "start"
  const x = Number(shape.attrs.x) - (anchor === "middle" ? width / 2 : anchor === "end" ? width : 0)
  return { x, y: Number(shape.attrs.y) - fontUnits * 0.75, width, height: fontUnits }
}
