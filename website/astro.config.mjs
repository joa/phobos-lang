// @ts-check
import { readFileSync } from "node:fs"
import { defineConfig } from "astro/config"
import sitemap from "@astrojs/sitemap"
import starlight from "@astrojs/starlight"
import starlightLinksValidator from "starlight-links-validator"
import sources from "./integrations/sources.mjs"
import traceFences from "./integrations/remark-trace-fences.mjs"

const phobosGrammar = JSON.parse(
  readFileSync(new URL("./src/grammars/phobos.tmLanguage.json", import.meta.url), "utf8")
)

const REPO_URL = "https://github.com/joa/phobos-lang"

export default defineConfig({
  site: "https://phobos-lang.dev",
  trailingSlash: "ignore",
  // The standard library's reference reads its sources from phobos-lang, outside the site.
  vite: { server: { fs: { allow: [".."] } } },
  integrations: [
    // Starlight adds the sitemap itself unless one is configured; this one leaves out the
    // unlisted lab page, which is marked noindex.
    sitemap({ filter: page => !page.includes("/lab/") }),
    sources(),
    traceFences(),
    starlight({
      title: "Phobos",
      plugins: [
        starlightLinksValidator({
          // The validator sees only Markdown: not the custom pages (the front page, /viz/), and not
          // the inference page's headings, which InferencePage.astro draws. tests/links.test.ts
          // checks those links in the built site.
          exclude: ({ slug, link }) =>
            link === "/" || link.startsWith("/viz/") || (slug === "inference" && link === "#numbers"),
        }),
      ],
      description: "Tile-based GPU kernel language & LLM inference engine",
      // No alt: the site title beside the mark already names the link.
      logo: { dark: "./src/assets/mark-night.svg", light: "./src/assets/mark-dawn.svg", alt: "" },
      favicon: "/favicon.svg",
      social: [{ icon: "github", label: "GitHub", href: REPO_URL }],
      customCss: ["./src/styles/fonts.css", "./src/styles/tokens.css", "./src/styles/starlight.css"],
      components: {
        ThemeProvider: "./src/components/overrides/ThemeProvider.astro",
        ThemeSelect: "./src/components/overrides/ThemeSelect.astro",
        Head: "./src/components/overrides/Head.astro",
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
        { label: "Standard library", items: [{ autogenerate: { directory: "docs/stdlib" } }] },
        { label: "Execution context: ctx", items: [{ autogenerate: { directory: "docs/ctx" } }] },
        { label: "Attributes", items: [{ autogenerate: { directory: "docs/attributes" } }] },
        {
          label: "Python",
          // Listed by slug, so the tutorials are a labelled group of their own and the reference,
          // python/README.md rendered into the generated folder, comes last.
          items: [
            "docs/python",
            "docs/python/install",
            "docs/python/kernels",
            "docs/python/launching",
            "docs/python/autotuning",
            "docs/python/errors",
            "docs/python/ph2py",
            { label: "Tutorials", items: [{ autogenerate: { directory: "docs/python/tutorials" } }] },
            "docs/python/reference",
          ],
        },
        { label: "Examples", items: [{ autogenerate: { directory: "docs/examples" } }] },
        { label: "Inference", items: [{ autogenerate: { directory: "docs/inference" } }] },
        {
          label: "Reference",
          // Listed by slug: the pages rendered from the root files live in a generated folder,
          // which autogenerate would show as a group of its own.
          items: [
            "docs/reference",
            "docs/reference/spec",
            "docs/reference/abi",
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
