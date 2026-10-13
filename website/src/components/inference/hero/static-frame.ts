// The opening frame as a few SVG paths, for the build: what the page shows before the canvas starts,
// with no script and under reduced motion, and the pose the canvas opens on. Points of one ink and one
// of three sizes share a path of zero-length segments with round caps. The inks are CSS variables, so
// the one frame reads in both themes.

import { composeFrame, inkOf, makeFrame, makeScene, countsFor, type Frame, type Ink, type Label } from "./model"
import { STEP_S } from "./path"

// A moment with a token midway through the stack, its ripples running out, and words on the arc.
export const STATIC_TIME_S = STEP_S * 3.42

// The words' size and offset from their token, in view units, on the canvas and in the SVG alike.
export const LABEL_UNITS = 18
export const LABEL_OFFSET_UNITS = { x: 14, y: -22 }

export interface StaticPath {
  d: string
  widthUnits: number
  ink: Ink
}

export interface StaticFrame {
  paths: StaticPath[]
  heads: Frame["heads"]
  labels: Label[]
}

const SIZE_STEPS_UNITS = [1.4, 2.1, 3]

export function staticFrame(widthPx = 1000): StaticFrame {
  const scene = makeScene(countsFor(widthPx))
  const frame = makeFrame(scene)
  composeFrame(scene, { timeS: STATIC_TIME_S, pitch: 0, yaw: 0 }, frame)
  const groups = new Map<string, { level: number; step: number; parts: string[] }>()
  for (let index = 0; index < frame.count; index += 1) {
    const size = frame.size[index]
    const step = size < 1.8 ? 0 : size < 2.6 ? 1 : 2
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
  return { paths, heads: frame.heads.map(head => ({ ...head })), labels: frame.labels.map(label => ({ ...label })) }
}
