// Geometry of the abi-descriptor figure, pure so the tests can check it. One parameter,
// `A: tensor<f32>[M, K]`, and the seven kernel arguments it lowers to: the allocated and aligned
// pointers, the offset, the two sizes and the two strides. Box widths are proportional to bytes:
// a pointer is 8 bytes, an index field 4 at the default 32-bit index and 8 at 64. The byte counts
// are what phobos-compile's PTX declares (.u64 pointers, .u32 or .u64 index fields).

import { VIEW_WIDTH_UNITS, type Layout } from "../../../lib/figure/layout"
import { Polyline } from "../../../lib/figure/polyline"

export type IndexBits = 32 | 64

export const POINTER_BYTES = 8

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export interface TextAt {
  x: number
  y: number
  text: string
  anchor?: "start" | "middle" | "end"
}

export interface Slot {
  // What the slot holds: a pointer or an index field.
  kind: "pointer" | "index"
  // The value the host passes, as the docs write it.
  value: string
  // Its role, as the narrow layout labels each row.
  role: string
  bytes: number
}

export const SLOTS_OF_RANK_2: readonly Omit<Slot, "bytes">[] = [
  { kind: "pointer", value: "ptr", role: "allocated" },
  { kind: "pointer", value: "ptr", role: "aligned" },
  { kind: "index", value: "0", role: "offset" },
  { kind: "index", value: "M", role: "size 0" },
  { kind: "index", value: "K", role: "size 1" },
  { kind: "index", value: "K", role: "stride 0" },
  { kind: "index", value: "1", role: "stride 1" },
]

// The groups the wide layout brackets under the row, as slot index ranges (inclusive).
export const GROUPS: readonly { text: string; first: number; last: number }[] = [
  { text: "pointers", first: 0, last: 1 },
  { text: "offset", first: 2, last: 2 },
  { text: "sizes", first: 3, last: 4 },
  { text: "strides", first: 5, last: 6 },
]

export function slotsFor(bits: IndexBits): Slot[] {
  const indexBytes = bits / 8
  return SLOTS_OF_RANK_2.map(slot => ({ ...slot, bytes: slot.kind === "pointer" ? POINTER_BYTES : indexBytes }))
}

export function totalBytes(bits: IndexBits): number {
  return slotsFor(bits).reduce((sum, slot) => sum + slot.bytes, 0)
}

export interface SlotAt extends Slot {
  rect: Rect
  valueLabel: TextAt
  bytesLabel: TextAt
  // Narrow only: the role to the left of the bar.
  roleLabel: TextAt | null
  // From the parameter to the slot.
  wire: Polyline
}

export interface AbiGeometry {
  widthUnits: number
  heightUnits: number
  plateLabel: string
  param: { rect: Rect; label: TextAt }
  widthNote: TextAt
  slots: SlotAt[]
  // Wide only: a bracket under each group of slots, with its name.
  groups: { path: string; label: TextAt; extent: Rect }[]
  total: TextAt[]
}

export const PARAM_TEXT = "A: tensor<f32>[M, K]"

// A rough mono advance at the label size, for the tests' fit checks.
export const CHAR_UNITS = 7.3

interface WideSpec {
  heightUnits: number
  x: number
  param: Rect
  busY: number
  slotY: number
  slotHeight: number
  unitsPerByte: number
  gapUnits: number
}

const WIDE: WideSpec = {
  heightUnits: 316,
  x: 36,
  param: { x: 36, y: 48, width: 230, height: 34 },
  busY: 122,
  slotY: 168,
  slotHeight: 40,
  unitsPerByte: 11,
  gapUnits: 3,
}

interface NarrowSpec {
  heightUnits: number
  param: Rect
  busY: number
  railX: number
  barX: number
  firstRowY: number
  rowPitch: number
  barHeight: number
  unitsPerByte: number
}

const NARROW: NarrowSpec = {
  heightUnits: 470,
  param: { x: 90, y: 44, width: 220, height: 34 },
  busY: 116,
  railX: 100,
  barX: 112,
  firstRowY: 134,
  rowPitch: 38,
  barHeight: 26,
  unitsPerByte: 22,
}

export function abiGeometry(layout: Layout, bits: IndexBits): AbiGeometry {
  return layout === "wide" ? wide(bits) : narrow(bits)
}

function paramOf(rect: Rect): { rect: Rect; label: TextAt } {
  return { rect, label: { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2, text: PARAM_TEXT, anchor: "middle" } }
}

function totalText(bits: IndexBits): string {
  return `${SLOTS_OF_RANK_2.length} arguments, ${totalBytes(bits)} bytes`
}

function fieldText(bits: IndexBits): string {
  return `index fields i${bits}, ${bits / 8} B each`
}

function wide(bits: IndexBits): AbiGeometry {
  const spec = WIDE
  const widthUnits = VIEW_WIDTH_UNITS.wide
  const paramBottom = { x: spec.param.x + spec.param.width / 2, y: spec.param.y + spec.param.height }
  let x = spec.x
  const slots = slotsFor(bits).map(slot => {
    const rect = { x, y: spec.slotY, width: slot.bytes * spec.unitsPerByte, height: spec.slotHeight }
    x += rect.width + spec.gapUnits
    const centerX = rect.x + rect.width / 2
    return {
      ...slot,
      rect,
      valueLabel: { x: centerX, y: rect.y + rect.height / 2, text: slot.value, anchor: "middle" as const },
      bytesLabel: { x: centerX, y: rect.y + rect.height + 16, text: `${slot.bytes} B`, anchor: "middle" as const },
      roleLabel: null,
      wire: new Polyline([
        paramBottom,
        { x: paramBottom.x, y: spec.busY },
        { x: centerX, y: spec.busY },
        { x: centerX, y: rect.y },
      ]).simplified(),
    }
  })
  const bracketY = spec.slotY + spec.slotHeight + 34
  const groups = GROUPS.map(group => {
    const left = slots[group.first].rect.x + 2
    const right = slots[group.last].rect.x + slots[group.last].rect.width - 2
    const labelY = bracketY + 16
    return {
      path: `M${left} ${bracketY - 5}V${bracketY}H${right}V${bracketY - 5}`,
      label: { x: (left + right) / 2, y: labelY, text: group.text, anchor: "middle" as const },
      extent: { x: left, y: bracketY - 5, width: right - left, height: 5 },
    }
  })
  const totalY = bracketY + 48
  return {
    widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: "FIG / ABI / TENSOR DESCRIPTOR",
    param: paramOf(spec.param),
    widthNote: {
      x: widthUnits - spec.x,
      y: spec.param.y + spec.param.height / 2,
      text: `index width ${bits} bits`,
      anchor: "end",
    },
    slots,
    groups,
    total: [{ x: spec.x, y: totalY, text: `${totalText(bits)}; ${fieldText(bits)}` }],
  }
}

function narrow(bits: IndexBits): AbiGeometry {
  const spec = NARROW
  const widthUnits = VIEW_WIDTH_UNITS.narrow
  const paramBottom = { x: spec.param.x + spec.param.width / 2, y: spec.param.y + spec.param.height }
  const slots = slotsFor(bits).map((slot, index) => {
    const rowY = spec.firstRowY + index * spec.rowPitch
    const rect = { x: spec.barX, y: rowY, width: slot.bytes * spec.unitsPerByte, height: spec.barHeight }
    const midY = rowY + spec.barHeight / 2
    return {
      ...slot,
      rect,
      valueLabel: { x: rect.x + rect.width / 2, y: midY, text: slot.value, anchor: "middle" as const },
      bytesLabel: { x: rect.x + rect.width + 10, y: midY, text: `${slot.bytes} B` },
      roleLabel: { x: 20, y: midY, text: slot.role },
      wire: new Polyline([
        paramBottom,
        { x: paramBottom.x, y: spec.busY },
        { x: spec.railX, y: spec.busY },
        { x: spec.railX, y: midY },
        { x: rect.x, y: midY },
      ]).simplified(),
    }
  })
  const lastBottom = spec.firstRowY + (slots.length - 1) * spec.rowPitch + spec.barHeight
  return {
    widthUnits,
    heightUnits: spec.heightUnits,
    plateLabel: "FIG / ABI / DESCRIPTOR",
    param: paramOf(spec.param),
    widthNote: { x: 20, y: spec.param.y + spec.param.height + 18, text: `index width ${bits} bits` },
    slots,
    groups: [],
    total: [
      { x: 20, y: lastBottom + 30, text: totalText(bits) },
      { x: 20, y: lastBottom + 50, text: fieldText(bits) },
    ],
  }
}
