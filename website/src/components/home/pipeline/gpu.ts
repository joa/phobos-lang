// The GPU station: the launch the trace was made with, a grid of CTAs, each 256 threads in 8 warps of
// 32 lanes. The cloud pours into the lanes and the warps light up in turn. The grid is the host's
// choice for the trace's dims, not part of the kernel; the caption says so.

import { structure } from "./data"
import { fill } from "./source"
import { FRAMES, round, target, text, type Layout, type Shape, type StationGeometry, type Target } from "./types"

export const WARP_LIGHT_MS = 900
export const WARP_STEP_MS = 55

export interface CtaBox {
  x: number
  y: number
  width: number
  height: number
  bx: number
  by: number
}

export function ctaBoxes(layout: Layout): { boxes: CtaBox[]; laneUnits: number; warpUnits: number; headerUnits: number } {
  const wide = layout === "wide"
  const [gx, gy] = structure.gpu.grid
  const lanes = structure.gpu.lanes
  const warps = structure.gpu.warps
  const laneUnits = wide ? 8.5 : 5.2
  const warpUnits = wide ? 15 : 11
  const headerUnits = wide ? 26 : 22
  const pad = wide ? 12 : 6
  const width = lanes * laneUnits + pad * 2
  const height = headerUnits + warps * warpUnits + pad
  const gap = wide ? 34 : 14
  const totalWidth = gx * width + (gx - 1) * gap
  const left = wide ? 150 : (FRAMES.narrow.width - totalWidth) / 2
  const top = wide ? 140 : 84
  const boxes: CtaBox[] = []
  for (let by = 0; by < gy; by += 1) {
    for (let bx = 0; bx < gx; bx += 1) {
      boxes.push({ x: left + bx * (width + gap), y: top + by * (height + gap), width, height, bx, by })
    }
  }
  return { boxes, laneUnits, warpUnits, headerUnits }
}

/** When warp `warp` of CTA `cta` lights, after the cloud has landed. */
export function warpLightMs(cta: number, warp: number): number {
  return WARP_LIGHT_MS + (cta * structure.gpu.warps + warp) * WARP_STEP_MS
}

/** "M = N = K = 64" when the dims agree, else each one. */
export function dimsText(dims: Record<string, number>): string {
  const order = ["M", "N", "K"].filter(name => name in dims).concat(Object.keys(dims).filter(name => !["M", "N", "K"].includes(name)))
  const values = order.map(name => dims[name])
  if (values.every(value => value === values[0])) return `${order.join(" = ")} = ${values[0]}`
  return order.map(name => `${name} = ${dims[name]}`).join(", ")
}

export function gpuStation(layout: Layout): StationGeometry {
  const frame = FRAMES[layout]
  const wide = layout === "wide"
  const view = structure.gpu
  const { boxes, laneUnits, warpUnits, headerUnits } = ctaBoxes(layout)
  const shapes: Shape[] = []
  const targets: Target[] = []
  const pad = wide ? 12 : 6
  const perWarp = Math.floor(448 / (boxes.length * view.warps))

  boxes.forEach((box, cta) => {
    const delayMs = 100 + cta * 90
    shapes.push({ tag: "rect", attrs: { x: round(box.x), y: round(box.y), width: round(box.width), height: round(box.height), class: "pp-cta pp-in" }, delayMs })
    shapes.push(text(box.x + pad, box.y + (wide ? 17 : 15), wide ? `CTA (${box.bx}, ${box.by})` : `(${box.bx}, ${box.by})`, "pp-label pp-in", { delayMs }))
    for (let warp = 0; warp < view.warps; warp += 1) {
      const y = round(box.y + headerUnits + (warp + 0.5) * warpUnits)
      const x0 = round(box.x + pad)
      const x1 = round(box.x + pad + view.lanes * laneUnits)
      shapes.push({
        tag: "path",
        attrs: {
          d: `M${x0} ${y}H${x1}`,
          class: "pp-warp",
          "stroke-width": round(warpUnits - 4),
          "stroke-dasharray": `${round(laneUnits - 1.5)} 1.5`,
        },
        delayMs: delayMs + 120,
        laterMs: warpLightMs(cta, warp),
      })
      for (let k = 0; k < perWarp; k += 1) {
        const lane = Math.floor(((k + 0.5) / perWarp) * view.lanes)
        targets.push({
          ...target(x0 + (lane + 0.5) * laneUnits - 0.75, y, k % 2 === 0 ? "global" : "hi", k % 4 === 0 ? 2 : 1, 0, 0.9),
          group: cta * view.warps + warp,
        })
      }
    }
  })

  const dims = dimsText(view.dims)
  const facts = [
    `grid ${view.grid.join(" x ")}`,
    `${view.threads} threads a CTA`,
    `${view.warps} warps of ${view.lanes} lanes`,
    `${view.sharedBytesPerCta} B shared a CTA`,
    dims,
    view.chip,
  ]
  if (wide) {
    const x = boxes[boxes.length - 1].x + boxes[boxes.length - 1].width + 48
    facts.forEach((fact, index) => shapes.push(text(x, 150 + index * 24, fact, `pp-label pp-in${index === 0 ? " is-strong" : ""}`, { delayMs: 300 + index * 80 })))
    shapes.push(text(x, 150 + facts.length * 24 + 12, "the host's choice for these dims", "pp-label is-faint pp-in", { delayMs: 800 }))
  } else {
    const y0 = boxes[boxes.length - 1].y + boxes[boxes.length - 1].height + 34
    facts.forEach((fact, index) => shapes.push(text(14, y0 + index * 20, fact, `pp-label pp-in${index === 0 ? " is-strong" : ""}`, { delayMs: 300 + index * 80 })))
  }

  // The pour lands top rows first.
  targets.sort((a, b) => a.y - b.y || a.x - b.x)
  return { frame, shapes, targets: fill(targets, frame.width, frame.height), streams: [] }
}
