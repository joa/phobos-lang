// A tiny highlighter for the few Phobos lines the tiles-and-lanes scene shows: keywords, builtins,
// ctx fields and numbers. Pure, so the build and the tests share it.

export type TokenKind = "kw" | "fn" | "ctx" | "num" | "plain"

export interface Token {
  text: string
  kind: TokenKind
}

const KEYWORDS = new Set(["for", "in", "var", "let", "if", "kernel"])
const BUILTINS = new Set(["range", "dot", "sum"])
const PATTERN = /(ctx\.[a-z_]+)|([A-Za-z_][A-Za-z0-9_]*)|(\d+(?:\.\d+)?)|(\s+)|(.)/g

export function tokenize(line: string): Token[] {
  const tokens: Token[] = []
  const push = (text: string, kind: TokenKind) => {
    const last = tokens[tokens.length - 1]
    if (last && last.kind === kind) last.text += text
    else tokens.push({ text, kind })
  }
  for (const match of line.matchAll(PATTERN)) {
    const [text, ctx, word, number] = match
    if (ctx) push(text, "ctx")
    else if (word) push(text, KEYWORDS.has(word) ? "kw" : BUILTINS.has(word) ? "fn" : "plain")
    else if (number) push(text, "num")
    else push(text, "plain")
  }
  return tokens
}
