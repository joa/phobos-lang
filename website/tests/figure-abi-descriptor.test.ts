import { describe, expect, it } from "vitest"
import AbiDescriptorFigure, { readoutFor } from "../src/components/figures/abi-descriptor/abi-descriptor.figure"
import {
  abiGeometry,
  CHAR_UNITS,
  slotsFor,
  totalBytes,
  type Rect,
  type TextAt,
} from "../src/components/figures/abi-descriptor/geometry"

function inside(inner: Rect, outer: Rect): boolean {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  )
}

function overlaps(a: Rect, b: Rect): boolean {
  return a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

// A label's box from its anchor, at the mono advance and a 14-unit line.
function textRect(at: TextAt): Rect {
  const width = at.text.length * CHAR_UNITS
  const x = at.anchor === "middle" ? at.x - width / 2 : at.anchor === "end" ? at.x - width : at.x
  return { x, y: at.y - 7, width, height: 14 }
}

describe("abi-descriptor slots", () => {
  it("lowers a rank-2 tensor to 3 + 2r arguments", () => {
    expect(slotsFor(32)).toHaveLength(7)
    expect(slotsFor(32).map(slot => slot.value)).toEqual(["ptr", "ptr", "0", "M", "K", "K", "1"])
  })

  it("counts bytes as the PTX declares them", () => {
    expect(totalBytes(32)).toBe(2 * 8 + 5 * 4)
    expect(totalBytes(64)).toBe(2 * 8 + 5 * 8)
  })
})

for (const layout of ["wide", "narrow"] as const) {
  for (const bits of [32, 64] as const) {
    describe(`abi-descriptor geometry, ${layout}, ${bits}-bit`, () => {
      const geometry = abiGeometry(layout, bits)
      const plateRect = { x: 0, y: 0, width: geometry.widthUnits, height: geometry.heightUnits }
      const labels = [
        geometry.param.label,
        geometry.widthNote,
        ...geometry.slots.flatMap(slot => [slot.bytesLabel, ...(slot.roleLabel ? [slot.roleLabel] : [])]),
        ...geometry.groups.map(group => group.label),
        ...geometry.total,
      ]

      it("keeps every box on the plate and apart", () => {
        const rects = [geometry.param.rect, ...geometry.slots.map(slot => slot.rect)]
        for (const rect of rects) expect(inside(rect, plateRect)).toBe(true)
        for (let i = 0; i < rects.length; i += 1)
          for (let j = i + 1; j < rects.length; j += 1) expect(overlaps(rects[i], rects[j])).toBe(false)
      })

      it("draws box widths in proportion to bytes", () => {
        const [pointer, , offset] = geometry.slots
        expect(offset.rect.width / pointer.rect.width).toBe(bits / 64)
      })

      it("fits each value inside its slot", () => {
        for (const slot of geometry.slots) expect(inside(textRect(slot.valueLabel), slot.rect)).toBe(true)
      })

      it("keeps labels on the plate, clear of the plate label and of each other and the boxes", () => {
        const rects = labels.map(textRect)
        const boxes = [...geometry.slots.map(slot => slot.rect), geometry.param.rect]
        for (const rect of rects) {
          expect(inside(rect, plateRect)).toBe(true)
          expect(rect.y).toBeGreaterThan(28)
        }
        for (let i = 0; i < rects.length; i += 1) {
          for (let j = i + 1; j < rects.length; j += 1) expect(overlaps(rects[i], rects[j])).toBe(false)
          // The parameter's own label sits inside its box.
          if (labels[i] === geometry.param.label) continue
          for (const box of boxes) expect(overlaps(rects[i], box)).toBe(false)
        }
      })

      it("runs every wire from the parameter to its slot", () => {
        const paramBottom = geometry.param.rect.y + geometry.param.rect.height
        for (const slot of geometry.slots) {
          expect(slot.wire.start.y).toBe(paramBottom)
          const end = slot.wire.end
          const touches =
            (end.y === slot.rect.y && end.x > slot.rect.x && end.x < slot.rect.x + slot.rect.width) ||
            (end.x === slot.rect.x && end.y > slot.rect.y && end.y < slot.rect.y + slot.rect.height)
          expect(touches).toBe(true)
          for (const point of slot.wire.points) expect(inside({ ...point, width: 0, height: 0 }, plateRect)).toBe(true)
        }
      })

      it("keeps wires out of the labels", () => {
        const rects = labels.filter(at => at !== geometry.param.label).map(textRect)
        for (const slot of geometry.slots) {
          const points = slot.wire.points
          for (let i = 1; i < points.length; i += 1) {
            const a = points[i - 1]
            const b = points[i]
            const segment = {
              x: Math.min(a.x, b.x),
              y: Math.min(a.y, b.y),
              width: Math.abs(a.x - b.x) || 0.01,
              height: Math.abs(a.y - b.y) || 0.01,
            }
            for (const rect of rects) expect(overlaps(segment, rect)).toBe(false)
          }
        }
      })
    })
  }
}

describe("abi-descriptor static render", () => {
  it("draws the settled end state for both widths in both layouts", () => {
    for (const layout of ["wide", "narrow"] as const) {
      for (const bits of [32, 64] as const) {
        const render = AbiDescriptorFigure.renderStatic(layout, { bits: String(bits) })
        expect(render.readout).toBe(readoutFor(bits))
        // The parameter and all seven slots are lit; no packet is left in flight.
        expect(render.drawing.match(/chart-box[^"]*is-on/g)?.length).toBe(8)
        const packets = render.drawing.match(/class="chart-packet[^"]*"/g) ?? []
        expect(packets).toHaveLength(7)
        for (const packet of packets) expect(packet).toContain("is-hidden")
        expect(render.drawing).toContain(`${totalBytes(bits)} bytes`)
      }
    }
  })
})
