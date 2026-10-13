// Shapes to SVG markup, for the build-time render. Pure.

import type { Shape } from "./types"

function escape(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
}

export function shapeMarkup(shape: Shape): string {
  const attrs = Object.entries(shape.attrs)
    .filter(([name]) => name !== "style")
    .map(([name, value]) => `${name}="${escape(String(value))}"`)
  const style: string[] = shape.attrs.style ? [String(shape.attrs.style)] : []
  if (shape.delayMs !== undefined) style.push(`--d: ${Math.round(shape.delayMs)}ms`)
  if (shape.laterMs !== undefined) style.push(`--l: ${Math.round(shape.laterMs)}ms`)
  if (style.length > 0) attrs.push(`style="${style.join("; ")}"`)
  const open = `<${shape.tag} ${attrs.join(" ")}`
  if (shape.spans) {
    const runs = shape.spans.map(span => `<tspan class="${escape(span.cls)}">${escape(span.text)}</tspan>`).join("")
    return `${open} xml:space="preserve">${runs}</${shape.tag}>`
  }
  return shape.text === undefined ? `${open}></${shape.tag}>` : `${open}>${escape(shape.text)}</${shape.tag}>`
}

export function shapesMarkup(shapes: Shape[]): string {
  return shapes.map(shapeMarkup).join("")
}
