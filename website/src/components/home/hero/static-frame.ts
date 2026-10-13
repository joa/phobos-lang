// The opening frame as a few SVG paths, for the build: what the page shows before the canvas starts,
// with no script, and the pose the canvas opens on. Points of one ink and one of three sizes share a
// path of zero-length segments with round caps.

import { composeFrame, inkOf, makeFrame, makeScene, type Frame, type Ink } from "./frame"

export interface StaticPath {
  d: string
  widthUnits: number
  ink: Ink
}

const SIZE_STEPS_UNITS = [1.5, 2.2, 3.1]

export interface StaticFrame {
  paths: StaticPath[]
  silhouette: Frame["silhouette"]
}

export function staticFrame(bodyCount = 2600): StaticFrame {
  const scene = makeScene(bodyCount)
  const frame = makeFrame(scene)
  composeFrame(scene, { timeS: 0, pitch: 0, yaw: 0 }, "emit", frame)
  const groups = new Map<string, { level: number; step: number; parts: string[] }>()
  for (let index = 0; index < frame.count; index += 1) {
    const size = frame.size[index]
    const step = size < 1.9 ? 0 : size < 2.6 ? 1 : 2
    const level = frame.level[index]
    const key = `${level}.${step}`
    let group = groups.get(key)
    if (!group) {
      group = { level, step, parts: [] }
      groups.set(key, group)
    }
    group.parts.push(`M${frame.x[index].toFixed(0)} ${frame.y[index].toFixed(0)}h0`)
  }
  const paths = [...groups.values()]
    .sort((a, b) => a.level - b.level || a.step - b.step)
    .map(group => ({ d: group.parts.join(""), widthUnits: SIZE_STEPS_UNITS[group.step], ink: inkOf(group.level) }))
  return { paths, silhouette: { ...frame.silhouette } }
}

// The ink as CSS, from the theme's variables.
export function inkCss(ink: Ink): string {
  const to = Math.round(ink.mix * 100)
  return `color-mix(in srgb, var(--${ink.to}) ${to}%, var(--${ink.from}))`
}
