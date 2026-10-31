import { readFileSync } from "node:fs"
import { type BundledLanguage, createHighlighter } from "shiki"
import { describe, expect, it } from "vitest"

const tokens = readFileSync(new URL("../../phobos-lang/src/token.rs", import.meta.url), "utf8")
const grammar = readFileSync(new URL("../src/grammars/phobos.tmLanguage.json", import.meta.url), "utf8")

/** Each token's text and its innermost scope, tokenized by shiki as the site's pages are. */
async function scopes(code: string): Promise<[string, string][]> {
  const highlighter = await createHighlighter({ langs: [JSON.parse(grammar)], themes: ["github-dark-default"] })
  const { tokens } = highlighter.codeToTokens(code, {
    lang: "phobos" as BundledLanguage,
    theme: "github-dark-default",
    includeExplanation: true,
  })
  return tokens
    .flat()
    .flatMap(token => token.explanation ?? [])
    .map(part => [part.content, part.scopes.at(-1)?.scopeName ?? ""])
}

describe("phobos grammar", () => {
  it("knows every keyword the lexer knows", () => {
    const keywords = [...tokens.matchAll(/\("([a-z_]+)", Tok::[A-Z][A-Za-z]*\)/g)].map(match => match[1])
    expect(keywords.length).toBeGreaterThan(10)
    const missing = keywords.filter(keyword => !new RegExp(`[(|]${keyword}[|)]`).test(grammar))
    expect(missing).toEqual([])
  })

  it("highlights the unroll setting of a for", async () => {
    for (const n of ["4", "U"]) {
      const found = new Map(await scopes(`for<unroll ${n}> i in range(0, N) {}`))
      expect(found.get("for")).toBe("keyword.control.phobos")
      expect(found.get("<")).toBe("punctuation.definition.unroll.begin.phobos")
      expect(found.get("unroll")).toBe("keyword.control.unroll.phobos")
      expect(found.get(">")).toBe("punctuation.definition.unroll.end.phobos")
      expect(found.get(n)).toBe(n === "U" ? "variable.other.constant.phobos" : "constant.numeric.integer.phobos")
    }
  })
})
