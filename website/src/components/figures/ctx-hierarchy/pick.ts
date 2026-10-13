// Cells a reader can pick. A pickable cell carries data-group and data-value, so the base's click
// delegation turns a click into a state change, cancels a running tour and syncs aria-pressed.
// enablePicking adds the keyboard: one tab stop per group, arrow keys move by data-row and data-col,
// Home and End jump to the ends, Enter and Space pick. With pickOnMove, moving focus or the mouse
// onto a cell picks it, for figures where hovering is the interaction.

import { setAttributes } from "../../../lib/figure/svg"

export const PICK_CLASS = "chart-pick"

export interface PickableOptions {
  group: string
  value: string
  label: string
  row: number
  col: number
  pressed: boolean
}

export function makePickable(element: Element, options: PickableOptions): void {
  element.classList.add(PICK_CLASS)
  setAttributes(element, {
    "data-group": options.group,
    "data-value": options.value,
    "data-row": options.row,
    "data-col": options.col,
    role: "button",
    "aria-label": options.label,
    "aria-pressed": String(options.pressed),
    tabindex: options.pressed ? 0 : -1,
  })
}

// Makes target the group's one tab stop.
export function setTabStop(cells: Iterable<Element>, target: Element | undefined): void {
  if (!target) return
  for (const cell of cells) cell.setAttribute("tabindex", cell === target ? "0" : "-1")
}

const ARROWS: Record<string, [number, number]> = {
  ArrowLeft: [0, -1],
  ArrowRight: [0, 1],
  ArrowUp: [-1, 0],
  ArrowDown: [1, 0],
}

export function enablePicking(svg: SVGSVGElement, { pickOnMove = false }: { pickOnMove?: boolean } = {}): void {
  const cellOf = (target: EventTarget | null): SVGElement | null =>
    target instanceof Element ? target.closest<SVGElement>(`.${PICK_CLASS}`) : null
  const pick = (cell: Element) => {
    if (cell.getAttribute("aria-pressed") !== "true") cell.dispatchEvent(new MouseEvent("click", { bubbles: true }))
  }
  const groupOf = (cell: Element) =>
    [...svg.querySelectorAll<SVGElement>(`.${PICK_CLASS}[data-group="${cell.getAttribute("data-group")}"]`)].filter(
      candidate => !candidate.closest(".is-hidden")
    )

  svg.addEventListener("keydown", event => {
    const cell = cellOf(event.target)
    if (!cell) return
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault()
      cell.dispatchEvent(new MouseEvent("click", { bubbles: true }))
      return
    }
    const cells = groupOf(cell)
    let next: SVGElement | undefined
    if (event.key === "Home") next = cells[0]
    else if (event.key === "End") next = cells[cells.length - 1]
    else if (event.key in ARROWS) next = neighbor(cell, cells, ARROWS[event.key])
    else return
    event.preventDefault()
    if (!next) return
    setTabStop(cells, next)
    next.focus()
    if (pickOnMove) pick(next)
  })

  svg.addEventListener("focusin", event => {
    const cell = cellOf(event.target)
    if (cell) setTabStop(groupOf(cell), cell)
  })

  if (pickOnMove) {
    svg.addEventListener("pointerover", event => {
      const cell = cellOf(event.target)
      if (cell && event.pointerType === "mouse") pick(cell)
    })
  }
}

// The nearest cell in a direction: least steps along it, then least drift across it.
function neighbor(from: Element, cells: Element[], [dRow, dCol]: [number, number]): SVGElement | undefined {
  const at = (cell: Element) => [Number(cell.getAttribute("data-row")), Number(cell.getAttribute("data-col"))]
  const [row, col] = at(from)
  let best: { cell: Element; score: number } | null = null
  for (const cell of cells) {
    const [r, c] = at(cell)
    const along = dRow !== 0 ? (r - row) * dRow : (c - col) * dCol
    if (along <= 0) continue
    const across = dRow !== 0 ? Math.abs(c - col) : Math.abs(r - row)
    const score = along * 1000 + across
    if (!best || score < best.score) best = { cell, score }
  }
  return best?.cell as SVGElement | undefined
}
