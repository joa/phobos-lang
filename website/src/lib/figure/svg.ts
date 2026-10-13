// SVG element helpers. They create elements through the current document, which is the browser's
// unless static rendering (static.ts) swaps in its own, so the same build() code draws in both.

export const SVG_NS = "http://www.w3.org/2000/svg"

type AttributeValue = string | number | null | undefined
export type Attributes = Record<string, AttributeValue>

export interface ElementFactory {
  createElementNS(namespace: string, tag: string): Element
}

let factoryOverride: ElementFactory | null = null

// Runs fn with elements created by factory instead of the browser's document.
export function withElementFactory<T>(factory: ElementFactory, fn: () => T): T {
  const previous = factoryOverride
  factoryOverride = factory
  try {
    return fn()
  } finally {
    factoryOverride = previous
  }
}

function factory(): ElementFactory {
  return factoryOverride ?? document
}

export function setAttributes(element: Element, attrs: Attributes): void {
  for (const [name, value] of Object.entries(attrs)) {
    if (value !== undefined && value !== null) element.setAttribute(name, String(value))
  }
}

export function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attributes = {},
  parent: Element | null = null
): SVGElementTagNameMap[K] {
  const element = factory().createElementNS(SVG_NS, tag) as SVGElementTagNameMap[K]
  setAttributes(element, attrs)
  parent?.append(element)
  return element
}

// A string becomes the text's content; an array of strings becomes one <tspan> per line.
export function svgText(
  content: string | string[],
  attrs: Attributes = {},
  parent: Element | null = null
): SVGTextElement {
  const text = svgEl("text", attrs, parent)
  if (!Array.isArray(content)) {
    text.textContent = content
    return text
  }
  content.forEach((line, index) => {
    const tspan = svgEl("tspan", { x: attrs.x ?? 0, dy: index === 0 ? 0 : "1.25em" }, text)
    tspan.textContent = line
  })
  return text
}
