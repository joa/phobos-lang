// @ts-check
import { readFileSync } from "node:fs"
import { defineConfig } from "astro/config"
import starlight from "@astrojs/starlight"

const phobosGrammar = JSON.parse(readFileSync(new URL("./src/grammars/phobos.tmLanguage.json", import.meta.url), "utf8"))

const REPO_URL = "https://github.com/joa/phobos-lang"

export default defineConfig({
  site: "https://phobos-lang.dev",
  trailingSlash: "ignore",
  integrations: [
    starlight({
      title: "Phobos",
      description: "Tile-based GPU kernel language & LLM inference engine",
      logo: { src: "./src/assets/mark.svg", alt: "Phobos" },
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
        { label: "Reference", items: [{ autogenerate: { directory: "docs/reference" } }] },
      ],
    }),
  ],
})
