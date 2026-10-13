// The PTX station: the instruction mix as bars of points, one point per instruction in the text, by
// family, colored by where the data goes. Static counts, not executed ones.

import type { PtxFamily } from "../pipeline-data"
import { structure } from "./data"
import { fill } from "./source"
import { FRAMES, round, target, text, type Layout, type Shape, type StationGeometry, type Target, type Tone } from "./types"

export function familyTone(family: PtxFamily): Tone {
  if (family.name.endsWith(".global")) return "global"
  if (family.name.endsWith(".shared")) return "shared"
  if (family.group === "float") return "reg"
  if (family.name === "bar.sync" || family.name === "trap") return "cost"
  if (family.group === "memory") return "hi"
  return "dust"
}

export function ptxFamilies(): PtxFamily[] {
  return structure.ptx.families.filter(family => family.count > 0).sort((a, b) => b.count - a.count)
}

export function ptxStation(layout: Layout): StationGeometry {
  const frame = FRAMES[layout]
  const wide = layout === "wide"
  const view = structure.ptx
  const families = ptxFamilies()
  const shapes: Shape[] = []
  const targets: Target[] = []

  const headY = wide ? 92 : 74
  const head = [view.target, view.maxntid, `${view.instructions} instructions`]
  if (wide) {
    let x = 150
    for (const part of head) {
      shapes.push(text(x, headY, part, "pp-label is-strong pp-in", { delayMs: 60 }))
      x += part.length * 7.5 + 28
    }
    shapes.push(text(1080, headY, "one point per instruction", "pp-label is-faint pp-in", { anchor: "end", delayMs: 60 }))
  } else {
    shapes.push(text(14, headY, `${view.target}  ${view.maxntid}`, "pp-label is-strong pp-in", { delayMs: 60 }))
    shapes.push(text(14, headY + 18, `${view.instructions} instructions, a point each`, "pp-label pp-in", { delayMs: 60 }))
  }

  const top = wide ? 134 : 130
  const pitch = wide ? 26 : 31
  const labelX = wide ? 200 : 104
  const dotX = wide ? 216 : 112
  const dotPitch = wide ? 6.2 : 4.2
  const lanes = wide ? 1 : 2
  families.forEach((family, row) => {
    const y = top + row * pitch
    const delayMs = 160 + row * 70
    const tone = familyTone(family)
    const columns = Math.ceil(family.count / lanes)
    const barWidth = columns * dotPitch
    const label = wide ? family.name : `${family.name} ${family.count}`
    shapes.push(text(labelX, y + 4.5, label, `pp-label pp-in${tone === "dust" ? "" : ` t-${tone}`}`, { anchor: "end", delayMs }))
    shapes.push({
      tag: "rect",
      attrs: { x: dotX - 3, y: round(y - (lanes === 1 ? 5 : 7)), width: round(barWidth + 1), height: lanes === 1 ? 10 : 14, class: `pp-bar pp-grow t-${tone}` },
      delayMs,
    })
    if (wide) shapes.push(text(dotX + barWidth + 8, y + 4.5, String(family.count), "pp-label pp-in", { delayMs: delayMs + 200 }))
    for (let index = 0; index < family.count; index += 1) {
      const column = Math.floor(index / lanes)
      const lane = index % lanes
      targets.push(target(dotX + column * dotPitch, y + (lanes === 1 ? 0 : lane * 5 - 2.5), tone, index % 5 === 0 ? 2 : 1, 0, 0.95))
    }
  })

  const note = wide ? "static counts over the PTX text, not executed instructions" : "static counts, not executed"
  shapes.push(text(wide ? 150 : 14, wide ? frame.height - 22 : frame.height - 22, note, "pp-label is-faint pp-in", { delayMs: 600 }))

  return { frame, shapes, targets: fill(targets, frame.width, frame.height), streams: [] }
}
