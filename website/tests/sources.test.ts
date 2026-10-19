import { readFileSync } from "node:fs"
import { describe, expect, it } from "vitest"
import {
  BLOB_URL,
  REPO_URL,
  SOURCES,
  TREE_URL,
  frontMatter,
  isPhobosCode,
  liftHeadings,
  render,
  rewriteLinks,
  tagFences,
} from "../integrations/sources.mjs"

const spec = SOURCES.find((s: { file: string }) => s.file === "SPEC.md")!
const windows = SOURCES.find((s: { file: string }) => s.file === "WINDOWS.md")!

describe("front matter", () => {
  it("takes the title from the H1 and removes it", () => {
    const page = frontMatter("# Phobos Language Spec\n\nBody.\n", { ...spec, source: "SPEC.md" })
    expect(page).toMatch(/^---\ntitle: "Phobos Language Spec"\nslug: "docs\/reference\/spec"\n/)
    expect(page).toContain('  label: "Language spec"\n  order: 1\n')
    expect(page).not.toContain("# Phobos Language Spec")
    expect(page.endsWith("\n\nBody.\n")).toBe(true)
  })

  it("falls back to the given title and keeps the body whole without an H1", () => {
    const page = frontMatter("### Building LLVM+MLIR on Windows\n\nText.\n", { ...windows, source: "WINDOWS.md" })
    expect(page).toContain('title: "Building on Windows"')
    expect(page).toContain("### Building LLVM+MLIR on Windows\n\nText.")
  })

  it("normalizes CRLF", () => {
    const page = frontMatter("# Benchmarks\r\n\r\nLine.\r\n", {
      route: "/docs/reference/benchmarks/",
      order: 3,
      source: "B.md",
    })
    expect(page).not.toContain("\r")
    expect(page).toContain('title: "Benchmarks"')
  })

  it("refuses a file with no H1 and no fallback", () => {
    expect(() => frontMatter("Text.\n", { route: "/x/", order: 1, source: "X.md" })).toThrow()
  })
})

describe("fence tagging", () => {
  it("tags plain fences that hold a kernel as phobos", () => {
    const text = "SGEMM\n```plain\n@autotune(T in [4, 8])\n@launch(256)\nkernel k(A: tensor<f32>[N]) {\n}\n```\n"
    expect(tagFences(text)).toContain("```phobos\n@autotune")
  })

  it("turns the grammar's ebnf fence into text, not phobos, though it mentions kernel", () => {
    const body = 'kernel      = { attribute } "kernel" ident "(" [ params ] ")" block ;\n'
    expect(tagFences("```ebnf\n" + body + "```\n")).toBe("```text\n" + body + "```\n")
  })

  it("keeps a plain fence that is not Phobos code", () => {
    const text = "```plain\ncargo run -p phobos-cli\n```\n"
    expect(tagFences(text)).toBe(text)
  })

  it("recognizes a leading comment and a function", () => {
    expect(isPhobosCode("// note\nkernel k() {\n}\n")).toBe(true)
    expect(isPhobosCode("fn silu(x) = x / (1.0 + exp(-x))\n")).toBe(true)
    expect(isPhobosCode('program     = { kernel | function } ;\nkernel      = { attribute } "kernel"\n')).toBe(false)
  })

  it("tags every kernel fence in SPEC.md and nothing else", () => {
    const text = readFileSync(new URL("../../SPEC.md", import.meta.url), "utf8").replace(/\r\n?/g, "\n")
    const tagged = tagFences(text)
    expect(tagged).toContain("```text\nprogram")
    expect(tagged).not.toContain("```plain\n")
    expect(tagged.match(/```phobos\n/g)?.length).toBe(text.match(/```plain\n/g)?.length)
  })
})

describe("link rewriting", () => {
  const isDir = (path: string) => path === "examples"

  it("routes the rendered root files to their pages", () => {
    const { text } = rewriteLinks("See [SPEC](./SPEC.md) and [env](ENV.md#logging-and-tracing).")
    expect(text).toBe("See [SPEC](/docs/reference/spec/) and [env](/docs/reference/environment/#logging-and-tracing).")
  })

  it("sends other repository paths to GitHub, directories as trees", () => {
    const { text } = rewriteLinks("[kda](./examples/kda_fp32.ph) and [`examples/`](./examples)", { isDir })
    expect(text).toBe(`[kda](${BLOB_URL}examples/kda_fp32.ph) and [\`examples/\`](${TREE_URL}examples)`)
  })

  it("copies images beside the page and points at the copy", () => {
    const { text, copies } = rewriteLinks("![Phobos vs cuBLAS](results/bench.svg)")
    expect(text).toBe("![Phobos vs cuBLAS](./results/bench.svg)")
    expect(copies).toEqual(["results/bench.svg"])
  })

  it("leaves absolute URLs, anchors and site routes alone", () => {
    const input = "[a](https://pi.dev) [b](#details) [c](/docs/) [d](mailto:x@example.com)"
    expect(rewriteLinks(input).text).toBe(input)
  })

  it("leaves code fences and inline code alone", () => {
    const input = "```bash\necho [x](./SPEC.md)\n```\n`[y](./ENV.md)` [z](./ENV.md)\n"
    expect(rewriteLinks(input).text).toBe(
      "```bash\necho [x](./SPEC.md)\n```\n`[y](./ENV.md)` [z](/docs/reference/environment/)\n"
    )
  })

  it("maps the repository root to the repository", () => {
    expect(rewriteLinks("[repo](./)").text).toBe(`[repo](${REPO_URL})`)
  })

  it("resolves a link against the directory of the file it is in", () => {
    const { text } = rewriteLinks("[spec](../SPEC.md) and [jit](phobos/jit.py)", { base: "python" })
    expect(text).toBe(`[spec](/docs/reference/spec/) and [jit](${BLOB_URL}python/phobos/jit.py)`)
  })
})

describe("render", () => {
  it("renders BENCHMARKS.md with its images copied and no relative links left", () => {
    const source = SOURCES.find((s: { file: string }) => s.file === "BENCHMARKS.md")
    const text = readFileSync(new URL("../../BENCHMARKS.md", import.meta.url), "utf8")
    const { page, copies } = render(source, text)
    expect(copies).toContain("results/inference.svg")
    expect(page).toContain('title: "Benchmarks"')
    const links = [...page.matchAll(/\]\(([^)]+)\)/g)].map(m => m[1])
    expect(links.filter(l => !/^(https?:|#|\/|\.\/results\/)/.test(l))).toEqual([])
  })
})

describe("liftHeadings", () => {
  it("lifts a file that starts at H3 so its shallowest heading is an H2", () => {
    const text = "### Building\n\nText.\n\n#### Step\n\n```sh\n### a comment, not a heading\n```\n"
    expect(liftHeadings(text)).toBe("## Building\n\nText.\n\n### Step\n\n```sh\n### a comment, not a heading\n```\n")
  })

  it("leaves a file whose shallowest heading is already an H2", () => {
    const text = "# Title\n\n## Section\n\n### Sub\n"
    expect(liftHeadings(text)).toBe(text)
  })
})
