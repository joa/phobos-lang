// @ts-check
import { readFileSync } from "node:fs"
import { defineConfig } from "astro/config"
import starlight from "@astrojs/starlight"
import sources from "./integrations/sources.mjs"

const phobosGrammar = JSON.parse(readFileSync(new URL("./src/grammars/phobos.tmLanguage.json", import.meta.url), "utf8"))

const REPO_URL = "https://github.com/joa/phobos-lang"

export default defineConfig({
  site: "https://phobos-lang.dev",
  trailingSlash: "ignore",
  integrations: [
    sources(),
    starlight({
      title: "Phobos",
      description: "Tile-based GPU kernel language & LLM inference engine",
      logo: { dark: "./src/assets/mark-night.svg", light: "./src/assets/mark-dawn.svg", alt: "Phobos" },
      favicon: "/favicon.svg",
      social: [{ icon: "github", label: "GitHub", href: REPO_URL }],
      customCss: ["./src/styles/fonts.css", "./src/styles/tokens.css", "./src/styles/starlight.css"],
      components: {
        ThemeProvider: "./src/components/overrides/ThemeProvider.astro",
        ThemeSelect: "./src/components/overrides/ThemeSelect.astro",
      },
      expressiveCode: {
        themes: ["github-dark-default", "github-light-default"],
        useStarlightDarkModeSwitch: true,
        shiki: { langs: [phobosGrammar] },
        styleOverrides: {
          borderRadius: "0px",
          borderColor: "var(--sky-line)",
          codeBackground: "var(--sky-1)",
          codeFontFamily: "var(--font-mono)",
          uiFontFamily: "var(--font-sans)",
          frames: { shadowColor: "transparent" },
        },
      },
      sidebar: [
        { label: "Getting started", items: [{ autogenerate: { directory: "docs/getting-started" } }] },
        { label: "Language guide", items: [{ autogenerate: { directory: "docs/language" } }] },
        { label: "Execution context: ctx", items: [{ autogenerate: { directory: "docs/ctx" } }] },
        { label: "Attributes", items: [{ autogenerate: { directory: "docs/attributes" } }] },
        { label: "Examples", items: [{ autogenerate: { directory: "docs/examples" } }] },
        { label: "Inference", items: [{ autogenerate: { directory: "docs/inference" } }] },
        {
          label: "Reference",
          // Listed by slug: the pages rendered from the root files live in a generated folder,
          // which autogenerate would show as a group of its own.
          items: [
            "docs/reference",
            "docs/reference/spec",
            "docs/reference/environment",
            "docs/reference/benchmarks",
            "docs/reference/cli",
            "docs/reference/windows",
          ],
        },
      ],
    }),
  ],
})
