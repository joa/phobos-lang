// The AST station: the parsed kernel as a tree, the `for` kept whole and every other statement folded
// to one node with a count of what it hides. Wide lays it out sideways (depth across, leaves down);
// narrow as an indented outline. Points gather on the nodes and along the edges.

import type { AstNode } from "../pipeline-data"
import { structure } from "./data"
import { fill } from "./source"
import { FRAMES, round, target, text, type Layout, type Shape, type StationGeometry, type Target, type Tone } from "./types"

const SHOWN_ROLES = new Set(["unroll", "start", "end", "step", "len"])

export interface PlacedNode {
  node: AstNode
  depth: number
  x: number
  y: number
  focus: boolean
  parent: PlacedNode | null
  /** Visit order, for the draw-in. */
  order: number
}

/** The name a node shows: what it binds or does, never a whole source line. */
export function astLabel(node: AstNode): string {
  switch (node.kind) {
    case "Kernel":
      return `kernel ${node.label}`
    case "Attributes":
      return "attributes"
    case "Params":
      return "params"
    case "For":
      return `for ${node.label}`
    case "Let":
      return `let ${node.label}`
    case "Var":
      return node.children || node.folded ? `var ${node.label}` : node.label
    case "Assign":
      return node.label === "+=" ? "+=" : "store C"
    case "Index":
      return "index"
    case "Call":
      return `${node.label}( )`
    default:
      return node.label
  }
}

function place(root: AstNode, wide: boolean): PlacedNode[] {
  const placed: PlacedNode[] = []
  let leaf = 0
  let order = 0
  const walk = (node: AstNode, depth: number, parent: PlacedNode | null, focus: boolean): PlacedNode => {
    const self: PlacedNode = { node, depth, x: 0, y: 0, focus, parent, order: order++ }
    placed.push(self)
    const children = node.children ?? []
    if (wide) {
      if (children.length === 0) {
        self.y = leaf++
      } else {
        const ys = children.map(child => walk(child, depth + 1, self, focus || child.kind === structure.ast.focus).y)
        self.y = (ys[0] + ys[ys.length - 1]) / 2
      }
    } else {
      self.y = leaf++
      for (const child of children) walk(child, depth + 1, self, focus || child.kind === structure.ast.focus)
    }
    return self
  }
  walk(root, 0, null, false)
  return placed
}

export function astStation(layout: Layout): StationGeometry & { nodes: PlacedNode[] } {
  const frame = FRAMES[layout]
  const wide = layout === "wide"
  const nodes = place(structure.ast.root, wide)
  const rowsTotal = Math.max(...nodes.map(node => node.y)) + 1
  if (wide) {
    const pitch = Math.min(19.5, 470 / (rowsTotal - 1))
    const top = 300 - (pitch * (rowsTotal - 1)) / 2
    for (const node of nodes) {
      node.x = 70 + node.depth * 160
      node.y = round(top + node.y * pitch)
    }
  } else {
    const pitch = 14.6
    for (const node of nodes) {
      node.x = 18 + node.depth * 22
      node.y = round(66 + node.y * pitch)
    }
  }

  const shapes: Shape[] = []
  const targets: Target[] = []
  const tone = (node: PlacedNode): Tone => (node.focus ? "reg" : "dust")
  const delay = (node: PlacedNode) => 120 + node.order * 34

  // Wide, a parent's edges leave after its label, so no edge crosses one.
  const labelEnd = (node: PlacedNode) => {
    const role = node.node.role && SHOWN_ROLES.has(node.node.role) && node.focus ? node.node.role.length + 1 : 0
    return node.x + 10 + (astLabel(node.node).length + role) * 12.5 * 0.6 + 6
  }
  for (const node of nodes) {
    if (!node.parent) continue
    const p = node.parent
    const startX = wide ? round(labelEnd(p)) : p.x
    const d = wide
      ? `M${startX} ${p.y}C${round((startX + node.x) / 2)} ${p.y} ${round((startX + node.x) / 2)} ${node.y} ${node.x - 5} ${node.y}`
      : `M${p.x} ${p.y + 5}V${node.y}H${node.x - 5}`
    shapes.push({
      tag: "path",
      attrs: { d, pathLength: 1, class: `pp-wire pp-draw t-${tone(node)}${node.focus ? "" : " is-faint"}` },
      delayMs: delay(node) - 20,
    })
    const steps = wide ? 7 : 4
    for (let step = 1; step <= steps; step += 1) {
      const t = step / (steps + 1)
      const x = wide ? startX + (node.x - startX) * t : t < 0.5 ? p.x : p.x + (node.x - p.x) * (t - 0.5) * 2
      const y = wide ? p.y + (node.y - p.y) * (3 * t * t - 2 * t * t * t) : t < 0.5 ? p.y + (node.y - p.y) * t * 2 : node.y
      targets.push(target(x, y, tone(node), 0, 0, node.focus ? 0.7 : 0.35))
    }
  }

  for (const node of nodes) {
    const isFor = node.node.kind === structure.ast.focus
    const leafy = !node.node.children?.length
    const cls = `pp-glyph pp-in t-${isFor ? "global" : tone(node)}${leafy && !node.node.folded ? " is-ring" : ""}`
    const r = isFor ? 5.5 : node.depth === 0 ? 5.5 : 3.8
    shapes.push(
      isFor || node.node.folded
        ? { tag: "rect", attrs: { x: node.x - r, y: node.y - r, width: r * 2, height: r * 2, class: cls }, delayMs: delay(node) }
        : { tag: "circle", attrs: { cx: node.x, cy: node.y, r, class: cls }, delayMs: delay(node) }
    )
    const label = astLabel(node.node)
    const labelCls = `pp-label pp-in${isFor ? " t-global is-strong" : node.focus ? " is-strong" : ""}`
    shapes.push(text(node.x + 10, node.y + 4.5, label, labelCls, { delayMs: delay(node) + 40 }))
    let after = node.x + 10 + label.length * 12.5 * 0.6 + 8
    if (node.node.role && SHOWN_ROLES.has(node.node.role) && node.focus) {
      shapes.push(text(after, node.y + 4.5, node.node.role, "pp-label is-faint pp-in", { delayMs: delay(node) + 40 }))
      after += node.node.role.length * 7.5 + 8
    }
    if (node.node.folded) {
      shapes.push(text(after, node.y + 4.5, `+${node.node.folded}`, "pp-label is-faint pp-in", { delayMs: delay(node) + 40 }))
    }
    for (let k = 0; k < 4; k += 1) {
      const angle = (k / 4) * Math.PI * 2 + node.order
      targets.push(target(node.x + Math.cos(angle) * 2.5, node.y + Math.sin(angle) * 2.5, isFor ? "global" : tone(node), k === 0 ? 2 : 1, 0, 0.8))
    }
  }

  if (wide) {
    shapes.push(
      text(40, frame.height - 22, `${structure.ast.nodeCount} nodes. Statements outside the for are folded; +n is what each one hides`, "pp-label pp-in", {
        delayMs: 400,
      })
    )
  }

  targets.sort((a, b) => a.x - b.x)
  return { frame, shapes, targets: fill(targets, frame.width, frame.height), streams: [], nodes }
}
