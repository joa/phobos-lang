import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"

const lexer = readFileSync(new URL("../../phobos-lang/src/lexer.rs", import.meta.url), "utf8")
const grammar = readFileSync(new URL("../src/grammars/phobos.tmLanguage.json", import.meta.url), "utf8")

describe("phobos grammar", () => {
  it("knows every keyword the lexer knows", () => {
    const keywords = [...lexer.matchAll(/"([a-z_]+)" => Tok::[A-Z][A-Za-z]*,/g)].map(match => match[1])
    expect(keywords.length).toBeGreaterThan(10)
    const missing = keywords.filter(keyword => !new RegExp(`[(|]${keyword}[|)]`).test(grammar))
    expect(missing).toEqual([])
  })
})
