# phobos-lang.dev

The Phobos website: Astro + Starlight, deployed to GitHub Pages by
`.github/workflows/website.yml`.

```
npm ci
npm run dev      # local server
npm run check    # tests and type checks, as CI runs them
npm run build    # static site in dist/
```

Site data that needs the compiler (kernel traces, CLI help, the attribute list)
is generated with the LLVM toolchain and committed; see `scripts/`.
