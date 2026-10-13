// Static rendering: runs a figure's build() and settle() at build time, in Node, and serializes the
// drawing, so a page shows every figure in its end state without JavaScript.
//
// The element below implements only what build() and settle() may use: attributes, classList,
// style.setProperty, textContent, append and remove. Measuring (getBBox, getTotalLength) and lookups
// (querySelector inside the drawing) belong in the motion paths, which never run here.

import type { ElementFactory } from "./svg"

class StaticClassList {
  #owner: StaticElement

  constructor(owner: StaticElement) {
    this.#owner = owner
  }

  #names(): string[] {
    return (this.#owner.getAttribute("class") ?? "").split(/\s+/).filter(Boolean)
  }

  #write(names: string[]): void {
    if (names.length === 0) this.#owner.removeAttribute("class")
    else this.#owner.setAttribute("class", names.join(" "))
  }

  contains(name: string): boolean {
    return this.#names().includes(name)
  }

  add(...names: string[]): void {
    const current = this.#names()
    for (const name of names) if (!current.includes(name)) current.push(name)
    this.#write(current)
  }

  remove(...names: string[]): void {
    this.#write(this.#names().filter(name => !names.includes(name)))
  }

  toggle(name: string, force?: boolean): boolean {
    const on = force ?? !this.contains(name)
    if (on) this.add(name)
    else this.remove(name)
    return on
  }
}

class StaticStyle {
  #properties = new Map<string, string>()

  setProperty(name: string, value: string | null): void {
    if (value === null || value === "") this.#properties.delete(name)
    else this.#properties.set(name, value)
  }

  removeProperty(name: string): string {
    const previous = this.#properties.get(name) ?? ""
    this.#properties.delete(name)
    return previous
  }

  getPropertyValue(name: string): string {
    return this.#properties.get(name) ?? ""
  }

  get cssText(): string {
    return [...this.#properties].map(([name, value]) => `${name}: ${value}`).join("; ")
  }
}

export class StaticElement {
  readonly tagName: string
  readonly classList = new StaticClassList(this)
  readonly style = new StaticStyle()
  readonly dataset: Record<string, string> = {}
  parent: StaticElement | null = null
  #attributes = new Map<string, string>()
  #children: (StaticElement | string)[] = []

  constructor(tagName: string) {
    this.tagName = tagName
  }

  get id(): string {
    return this.getAttribute("id") ?? ""
  }

  get children(): StaticElement[] {
    return this.#children.filter((child): child is StaticElement => typeof child !== "string")
  }

  setAttribute(name: string, value: string): void {
    this.#attributes.set(name, String(value))
  }

  getAttribute(name: string): string | null {
    return this.#attributes.get(name) ?? null
  }

  hasAttribute(name: string): boolean {
    return this.#attributes.has(name)
  }

  removeAttribute(name: string): void {
    this.#attributes.delete(name)
  }

  append(...nodes: (StaticElement | string)[]): void {
    for (const node of nodes) {
      if (typeof node !== "string") {
        node.remove()
        node.parent = this
      }
      this.#children.push(node)
    }
  }

  remove(): void {
    if (!this.parent) return
    this.parent.#children = this.parent.#children.filter(child => child !== this)
    this.parent = null
  }

  get textContent(): string {
    return this.#children.map(child => (typeof child === "string" ? child : child.textContent)).join("")
  }

  set textContent(value: string) {
    for (const child of this.children) child.parent = null
    this.#children = value === "" ? [] : [value]
  }

  // Finds the first descendant carrying a class, for the figure's root lookups (".chart-svg").
  querySelector(selector: string): StaticElement | null {
    if (!/^\.[\w-]+$/.test(selector))
      throw new Error(`static rendering supports only ".class" lookups, not "${selector}"`)
    const name = selector.slice(1)
    for (const child of this.children) {
      if (child.classList.contains(name)) return child
      const found = child.querySelector(selector)
      if (found) return found
    }
    return null
  }

  querySelectorAll(): StaticElement[] {
    return []
  }

  get outerHTML(): string {
    const attributes = [...this.#attributes].map(([name, value]) => ` ${name}="${escapeAttribute(value)}"`)
    const style = this.style.cssText
    if (style) attributes.push(` style="${escapeAttribute(style)}"`)
    const open = `<${this.tagName}${attributes.join("")}`
    if (this.#children.length === 0) return `${open}/>`
    return `${open}>${this.innerHTML}</${this.tagName}>`
  }

  get innerHTML(): string {
    return this.#children.map(child => (typeof child === "string" ? escapeText(child) : child.outerHTML)).join("")
  }
}

export const staticFactory: ElementFactory = {
  createElementNS: (_namespace, tag) => new StaticElement(tag) as unknown as Element,
}

function escapeText(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
}

function escapeAttribute(value: string): string {
  return escapeText(value).replace(/"/g, "&quot;")
}
