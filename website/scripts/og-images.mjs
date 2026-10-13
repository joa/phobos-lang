// Renders the Open Graph images, one per section, into public/og/: the mark on the night palette's
// star chart plate with the headline. Text goes through sharp's Pango renderer with the bundled Noto
// files, so the result does not depend on the fonts installed on the machine. Rerun after a change
// to the mark, the palette or the headline, and commit the PNGs.
//
//   node scripts/og-images.mjs

import { mkdirSync, readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import sharp from "sharp"

const WIDTH_PX = 1200
const HEIGHT_PX = 630
const MARGIN_PX = 80
const DOT_PITCH_PX = 24
const TICK_PX = 18
const MARK_SIZE_PX = 380
const TEXT_WIDTH_PX = 640

// Night tokens from src/styles/tokens.css.
const SKY_0 = "#05070f"
const SKY_LINE = "#1c2a48"
const STAR = "#e6ecf8"
const DUST = "#8d9ab3"
const BRAND = "#6ab8ff"

const HEADLINE = "Tile-based GPU kernel language &amp; LLM inference engine"

const SECTIONS = [
  { name: "home", plate: "PHOBOS", title: "Phobos" },
  { name: "docs", plate: "PHOBOS / DOCS", title: "Phobos docs" },
  { name: "inference", plate: "PHOBOS / INFERENCE", title: "Phobos inference" },
]

const fontFile = (family, weight) =>
  fileURLToPath(
    new URL(`../node_modules/@fontsource/${family}/files/${family}-latin-${weight}-normal.woff`, import.meta.url)
  )

const FONTS = {
  sans600: { font: "Noto Sans SemiBold", fontfile: fontFile("noto-sans", 600) },
  sans700: { font: "Noto Sans Bold", fontfile: fontFile("noto-sans", 700) },
  mono400: { font: "Noto Sans Mono", fontfile: fontFile("noto-sans-mono", 400) },
}

function plateSvg() {
  const dots = []
  for (let y = DOT_PITCH_PX; y < HEIGHT_PX; y += DOT_PITCH_PX) {
    for (let x = DOT_PITCH_PX; x < WIDTH_PX; x += DOT_PITCH_PX) dots.push(`<circle cx="${x}" cy="${y}" r="1"/>`)
  }
  const inset = 32
  const corners = [
    [inset, inset, 1, 1],
    [WIDTH_PX - inset, inset, -1, 1],
    [inset, HEIGHT_PX - inset, 1, -1],
    [WIDTH_PX - inset, HEIGHT_PX - inset, -1, -1],
  ]
  const ticks = corners
    .map(([x, y, dx, dy]) => `<path d="M${x} ${y + dy * TICK_PX}V${y}H${x + dx * TICK_PX}"/>`)
    .join("")
  const mark = readFileSync(new URL("../src/assets/mark-night.svg", import.meta.url), "utf8")
    .replace(/<svg[^>]*>/, "")
    .replace("</svg>", "")
    // At this scale the favicon's strokes would be heavy; keep them close to the figures' hairlines.
    .replace(/stroke-width="[\d.]+"/g, 'stroke-width="0.42"')
  const markScale = MARK_SIZE_PX / 32
  const markX = WIDTH_PX - MARGIN_PX - MARK_SIZE_PX + 20
  const markY = (HEIGHT_PX - MARK_SIZE_PX) / 2
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH_PX}" height="${HEIGHT_PX}">
  <defs>
    <radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="${BRAND}" stop-opacity="0.16"/>
      <stop offset="1" stop-color="${BRAND}" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="100%" height="100%" fill="${SKY_0}"/>
  <g fill="${SKY_LINE}">${dots.join("")}</g>
  <g fill="none" stroke="${DUST}" stroke-width="1.5">${ticks}</g>
  <circle cx="${markX + MARK_SIZE_PX / 2}" cy="${markY + MARK_SIZE_PX / 2}" r="${MARK_SIZE_PX * 0.62}" fill="url(#glow)"/>
  <g transform="translate(${markX} ${markY}) scale(${markScale})" fill="none" stroke-linejoin="round">${mark}</g>
</svg>`
}

async function text(markup, font, sizePx, color) {
  const { data, info } = await sharp({
    text: {
      text: `<span foreground="${color}" size="${sizePx}pt">${markup}</span>`,
      ...font,
      width: TEXT_WIDTH_PX,
      dpi: 72,
      rgba: true,
    },
  })
    .png()
    .toBuffer({ resolveWithObject: true })
  return { input: data, heightPx: info.height }
}

async function render(section) {
  const plate = await text(section.plate, FONTS.mono400, 22, DUST)
  const title = await text(section.title, FONTS.sans700, 76, STAR)
  const headline = await text(HEADLINE, FONTS.sans600, 40, STAR)
  const url = await text("phobos-lang.dev", FONTS.mono400, 24, BRAND)
  const blockHeightPx = title.heightPx + 20 + headline.heightPx
  const titleTopPx = Math.round((HEIGHT_PX - blockHeightPx) / 2)
  const layers = [
    { input: plate.input, left: MARGIN_PX, top: MARGIN_PX - 8 },
    { input: title.input, left: MARGIN_PX - 4, top: titleTopPx },
    { input: headline.input, left: MARGIN_PX, top: titleTopPx + title.heightPx + 20 },
    { input: url.input, left: MARGIN_PX, top: HEIGHT_PX - MARGIN_PX - url.heightPx + 8 },
  ]
  const out = fileURLToPath(new URL(`../public/og/${section.name}.png`, import.meta.url))
  await sharp(Buffer.from(plateSvg())).composite(layers).png({ compressionLevel: 9 }).toFile(out)
  console.log(`public/og/${section.name}.png`)
}

mkdirSync(new URL("../public/og/", import.meta.url), { recursive: true })
for (const section of SECTIONS) await render(section)
