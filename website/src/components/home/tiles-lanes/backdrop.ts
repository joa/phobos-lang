// The scene's backdrop: three low-poly bodies in the sky beside the stage, one per beat. Phobos, the
// third, stands in the middle; the first, for tiles, and the second, for lanes, orbit it on a tilted
// orbit, so they swing out toward the viewer and pass in front of it, then fall back behind. Each
// turns slowly about an axis of its own. While beat 1 and beat 2 play the bodies are plain spheres
// and the beat's own one is lit. In beat 3 the metaball blend comes on and the bodies turn to
// polished silver, light sweeping over them now and then; each orbit flattens into a line through
// Phobos, so the two swing in turn right through it and out the far side, fusing on the way in and
// drawing out a neck that snaps on the way out, while the whole floats and rocks. A blue line, a
// little way out, rings whatever is lit: the beat's own body, then all three. Pure, so the build
// renders the frame a page without script shows from the same functions the browser plays.
//
// A body is a geodesic mesh that turns with it. Each vertex starts on the body's own sphere and is
// drawn onto the surface of the bodies' shared field by Newton steps along its gradient, so a lone
// body stays a sphere and where fused bodies meet the vertices slide out onto the neck between them.
// A vertex that starts on ground another body owns first draws back along its ray to the edge of its
// own, so it comes out by the seam; a face whose every corner starts deep in another body's ground,
// as a body passing through Phobos's middle is, is that body's to draw and not drawn twice. The two
// bodies' faces meet at a seam edge to edge, which on a thin neck can leave a slit, so each seam face
// also lays a slightly larger, darker copy of itself under everything, which shows only where
// nothing covers it, as the shadow in the crease. Faces are shaded flat and drawn back to front.
//
// Units: a BACKDROP_WIDTH_UNITS x BACKDROP_HEIGHT_UNITS box, x right, y down, z toward the viewer.

import { clamp, easeInOutSine, lerp } from "../../../lib/figure/math"
import { makeMesh, multiply, normalize, rotateX, rotateY, rotateZ, type Mat3, type Mesh } from "./backdrop-mesh"
import type { BeatName } from "./scene"

export const BACKDROP_WIDTH_UNITS = 1000
export const BACKDROP_HEIGHT_UNITS = 700
const CENTER = { x: 500, y: 350 }
// How far the eye is from the orbit's center: the perspective, so a body near the viewer is drawn
// larger than one behind.
const EYE_UNITS = 2000

// Tiles, lanes and Phobos, in the beats' order.
const RADII = [82, 70, 94]
const RELIEF = [0.035, 0.035, 0.065]
const MESH_SEEDS = [0x7b0b5, 0x91e4, 0x5a1d]
// Each face of the icosahedron cut into FREQUENCY^2: about three facets along a radius.
const FREQUENCY = 3
const JITTER = 0.3

// The orbit, mostly in depth: tilted INCLINATION from edge-on, then rolled in the screen. A circle
// while the bodies are apart, once round in ORBIT_PERIOD_MS.
const HOME_ORBIT_UNITS = 300
// The lanes body rides a little inside the tiles body, so the two are not mirror images.
const ORBIT_SCALE = [1, 0.88]
const INCLINATION = 0.4
const ROLL = -0.16
const ORBIT_PERIOD_MS = 40000
// In beat 3 each orbit flattens into a line through Phobos, the two lines square to each other and a
// quarter swing apart, so one body is passing through while the other turns. A body swings out to
// REACH_UNITS, past where its neck snaps, there and back in SWING_PERIOD_MS; the lines turn round
// once in PRECESSION_PERIOD_MS, so a swing runs across the screen, then in depth, then across again.
const REACH_UNITS = 290
const SWING_PERIOD_MS = 12000
const PRECESSION_PERIOD_MS = 48000
// Blended, the whole floats up and down, and the orbit rocks in the screen, so the bodies rise and
// fall in turn.
const FLOAT = { units: 18, periodMs: 6100 }
const SWAY = { rad: 0.22, periodMs: 9700 }
// Each body's turn: its period, and how its axis leans from upright, sideways and toward the viewer.
const SPINS = [
  { periodMs: 29000, tilt: 0.42, lean: 0.2, phase: 0 },
  { periodMs: 34000, tilt: -0.35, lean: -0.15, phase: 2.1 },
  { periodMs: 47000, tilt: 0.18, lean: 0.35, phase: 4.4 },
]

// A body's field is (r / d)^FALLOFF, 1 on its surface. The steeper it falls, the tighter the necks
// between fused bodies, and the less a fused group swells past its bodies.
const FALLOFF = 2.5
// How the bodies' fields combine: as a power mean, so at the hard power the field is the largest of
// them and every body a sphere of its own, and at 1 it is their sum, the metaball.
const HARD_POWER = 8
// Colors follow the bodies more sharply than the field, so a neck shows where each one ends.
const COLOR_POWER = 1.5
const DIM = 0.3
// A body behind the orbit's center is dimmed by up to this much.
const DEPTH_FADE = 0.18
// Newton steps onto the surface, each at most this many radii long, until a vertex is this close.
const NEWTON = { steps: 12, maxStepRadii: 0.5, closeRadii: 1e-3 }
// Halvings that find where a vertex's ray leaves its own body's ground, and how near the body's
// middle that has to be, in radii, for the vertex to count as deep in another body's ground.
const SEAM_HALVINGS = 7
const DEEP_RADII = 0.55
// How far a seam face's copy underneath reaches past each of its corners, on screen, and how much of
// the face's light it keeps.
const UNDERLAY_UNITS = 5
const UNDERLAY_SHADE = 0.4

// A face on a seam, or folded, is shaded as the surface there faces rather than as its own plane, so
// the two bodies' faces along a seam agree and no crease shows dark. Folded: its normal this far or
// further from the surface's.
const FOLDED_COSINE = 0.6
// The highlight's line: how far out it runs from the bodies it rings, on screen, how much further
// out (in radii) as they blend and swell into each other, how round it fills the waist between
// fused ones and the grid its contour is traced on.
const OUTLINE = { padUnits: 16, swellRadii: 0.08, neckUnits: 70, cellUnits: 10 }
// Light from the upper left, toward the viewer.
const LIGHT = normalize([-0.5, -0.62, 0.6])
const AMBIENT = 0.3
// Silver: a facet's color goes this far toward the metal's, keeping a hint of its body's hue. Its
// brightness is what it reflects of a studio, bright overhead and dark below with a bright band at
// the horizon, rather than how it faces the light, and its shadows run deeper.
const METAL = { hueMix: 0.72, shadowDrop: 0.55 }
const STUDIO = { floor: 0.2, sky: 0.7, skyFrom: -0.3, skyTo: 0.75, band: 0.5, bandAt: 0.02, bandWidth: 0.11 }
// The polish: a sharp glint on the facets that face halfway between the light and the eye, and a
// faint rim where the surface turns away from the eye.
const GLINT = { strength: 1, power: 48 }
const RIM = { strength: 0.25, power: 3 }
// Now and then a band of light sweeps across the silver, corner to corner: once a period, taking
// crossingMs, the band widthUnits wide on screen and at most this bright.
const SWEEP = { periodMs: 5200, crossingMs: 1500, widthUnits: 75, strength: 0.65, angle: -0.6 }

export interface Pose {
  // The long half-axis of the orbit the first two bodies ride around the third.
  orbitUnits: number
  // How flat the orbit is: 0 a circle, 1 a line through the third body's middle.
  squash: number
  // 0 for separate spheres, 1 for the full metaball.
  blend: number
  // How lit each body is, 0 to 1: its beat's body is lit, the others dim.
  lights: number[]
  // How far the bodies have turned to polished silver, 0 to 1: only the fused body does.
  shine: number
  // What the blue line rings, and how strongly: one ring as a beat plays, two crossing over between
  // beats.
  highlights: Highlight[]
}

export interface Highlight {
  bodies: number[]
  opacity: number
}

// A highlight's line on screen: segments x0 y0 x1 y1 each, how strongly drawn, how deep it sits (the
// mean depth of what it rings) and which bodies those are.
export interface Outline {
  segments: number[]
  opacity: number
  depth: number
  bodies: number[]
}

// Where the bodies are in their motion, which never resets: how far round its orbit each body is,
// which way the flattened orbits point and the time that turns each body.
export interface Motion {
  orbitRad: number
  axisRad: number
  clockMs: number
}

export interface Placed {
  x: number
  y: number
  z: number
  r: number
  spin: Mat3
}

// One flat facet, as drawn: its outline, how much light it catches (0 to 1), what share of its color
// is each body's (summing to 1), how lit it is, how far it has turned to silver, how far its color
// goes on toward the shine within the body (gloss) and past how faint the bodies are drawn (sheen),
// all 0 to 1, and how deep it sits.
export interface Facet {
  points: number[]
  shade: number
  weights: number[]
  light: number
  metal: number
  gloss: number
  sheen: number
  depth: number
}

export const START_MOTION: Motion = { orbitRad: -0.6, axisRad: 0, clockMs: 0 }
const SETTLED_MOTION: Motion = { orbitRad: 0.65, axisRad: -0.2, clockMs: 0 }

// Beat 3: the blend comes on at once, while the bodies are still apart, so they reach for each other
// on the way in, and the orbits flatten until the bodies swing through Phobos.
const WAKE = { startMs: 0, durationMs: 600 }
const BLEND = { startMs: 0, durationMs: 500 }
const MERGE = { startMs: 0, durationMs: 1500 }
// How far into a transition out of beat 3 the blend is gone.
const PART_BY = 0.45

function phase(tMs: number, startMs: number, durationMs: number): number {
  return clamp((tMs - startMs) / durationMs, 0, 1)
}

export function beatPose(beat: BeatName, tMs: number): Pose {
  const apart = { orbitUnits: HOME_ORBIT_UNITS, squash: 0, blend: 0, shine: 0 }
  if (beat === "tiles") return { ...apart, lights: [1, DIM, DIM], highlights: [{ bodies: [0], opacity: 1 }] }
  if (beat === "lanes") return { ...apart, lights: [DIM, 1, DIM], highlights: [{ bodies: [1], opacity: 1 }] }
  const wake = easeInOutSine(phase(tMs, WAKE.startMs, WAKE.durationMs))
  const merge = easeInOutSine(phase(tMs, MERGE.startMs, MERGE.durationMs))
  return {
    orbitUnits: lerp(HOME_ORBIT_UNITS, REACH_UNITS, merge),
    squash: merge,
    blend: easeInOutSine(phase(tMs, BLEND.startMs, BLEND.durationMs)),
    lights: [lerp(0.65, 0.9, wake), lerp(0.65, 0.9, wake), lerp(0.65, 1, wake)],
    shine: merge,
    highlights: [{ bodies: [0, 1, 2], opacity: 1 }],
  }
}

// The frame a page without script shows: beat 3, one body sunk into Phobos as it swings through,
// the other drawing a neck out of it toward the far end of its swing.
export function settled(): { pose: Pose; motion: Motion } {
  return { pose: beatPose("mixed", MERGE.startMs + MERGE.durationMs), motion: SETTLED_MOTION }
}

// Parting, the blend and the shine are gone by halfway, where the player hands the pills and the
// code to the next beat, and the orbits are round again, so the bodies are out of Phobos and the
// necks have snapped while beat 3 still shows.
export function lerpPose(from: Pose, to: Pose, t: number): Pose {
  const parting = easeInOutSine(clamp(t / PART_BY, 0, 1))
  return {
    orbitUnits: lerp(from.orbitUnits, to.orbitUnits, t),
    squash: lerp(from.squash, to.squash, from.squash > to.squash ? parting : t),
    blend: lerp(from.blend, to.blend, from.blend > to.blend ? parting : t),
    lights: from.lights.map((light, index) => lerp(light, to.lights[index], t)),
    shine: lerp(from.shine, to.shine, from.shine > to.shine ? parting : t),
    highlights: crossfade(from.highlights, to.highlights, t),
  }
}

// The rings of both poses, the outgoing fading as the incoming comes up; one that both share stays.
function crossfade(from: Highlight[], to: Highlight[], t: number): Highlight[] {
  const out = new Map<string, Highlight>()
  for (const [list, weight] of [
    [from, 1 - t],
    [to, t],
  ] as const) {
    for (const { bodies, opacity } of list) {
      const key = bodies.join(",")
      const seen = out.get(key)
      out.set(key, { bodies, opacity: (seen?.opacity ?? 0) + opacity * weight })
    }
  }
  return [...out.values()].filter(highlight => highlight.opacity > 1e-3)
}

// How fast the bodies go round their orbits in this pose, radians per ms: a slow circle while they
// are apart, a quicker swing through Phobos once the orbits are flat.
export function orbitSpeed(pose: Pose): number {
  return 2 * Math.PI * lerp(1 / ORBIT_PERIOD_MS, 1 / SWING_PERIOD_MS, pose.squash)
}

// How fast the flattened orbits turn, radians per ms; a round orbit has no way it points.
export function axisSpeed(pose: Pose): number {
  return ((2 * Math.PI) / PRECESSION_PERIOD_MS) * pose.squash
}

export function makeMeshes(): Mesh[] {
  return MESH_SEEDS.map(seed => makeMesh(seed, FREQUENCY, JITTER))
}

export function place(pose: Pose, motion: Motion): Placed[] {
  const wave = (period: { periodMs: number }) => pose.blend * Math.sin((2 * Math.PI * motion.clockMs) / period.periodMs)
  const float = FLOAT.units * wave(FLOAT)
  const roll = ROLL + SWAY.rad * wave(SWAY)
  return RADII.map((r, body) => {
    const { periodMs, tilt, lean, phase: spinPhase } = SPINS[body]
    const spin = multiply(
      multiply(rotateZ(tilt), rotateX(lean)),
      rotateY((2 * Math.PI * motion.clockMs) / periodMs + spinPhase)
    )
    if (body === 2) return { x: CENTER.x, y: CENTER.y + float, z: 0, r, spin }
    // An ellipse in the orbit's plane, the second body's square to the first's and a quarter turn on;
    // round, the two ride opposite each other.
    const long = pose.orbitUnits * ORBIT_SCALE[body]
    const short = long * (1 - pose.squash)
    const turn = motion.orbitRad + (body * Math.PI) / 2
    const axis = motion.axisRad + (body * Math.PI) / 2
    const [along, aside] = [long * Math.cos(turn), short * Math.sin(turn)]
    const across = along * Math.cos(axis) - aside * Math.sin(axis)
    const deep = along * Math.sin(axis) + aside * Math.cos(axis)
    const down = deep * Math.sin(INCLINATION)
    return {
      x: CENTER.x + across * Math.cos(roll) - down * Math.sin(roll),
      y: CENTER.y + float + across * Math.sin(roll) + down * Math.cos(roll),
      z: deep * Math.cos(INCLINATION),
      r,
      spin,
    }
  })
}

// The bodies' field at a point as a reach, what d / r would be for a lone body with that field (1
// on the surface, below 1 inside), with its gradient after it in out and each body's own field in
// contributions. A reach is the distance to a lone body's surface in radii, so a Newton step on it
// lands a lone sphere's vertex exactly.
function reach(
  placed: Placed[],
  power: number,
  x: number,
  y: number,
  z: number,
  out: Float64Array,
  contributions: Float64Array
): number {
  let total = 0
  let gx = 0
  let gy = 0
  let gz = 0
  for (let body = 0; body < placed.length; body += 1) {
    const { r } = placed[body]
    const dx = x - placed[body].x
    const dy = y - placed[body].y
    const dz = z - placed[body].z
    // Held off the center, so a power of the field stays finite.
    const distance2 = Math.max((r * 0.1) ** 2, dx * dx + dy * dy + dz * dz)
    const own = ((r * r) / distance2) ** (FALLOFF / 2)
    contributions[body] = own
    const powered = own ** (power - 1)
    total += powered * own
    // own^(power - 1) times the gradient of own.
    const slope = (powered * -FALLOFF * own) / distance2
    gx += slope * dx
    gy += slope * dy
    gz += slope * dz
  }
  const field = total ** (1 / power)
  const value = field ** (-1 / FALLOFF)
  // d reach = -reach / (FALLOFF field) d field, and d field = field / total times the sum above.
  const scale = -value / (FALLOFF * total)
  out[0] = value
  out[1] = gx * scale
  out[2] = gy * scale
  out[3] = gz * scale
  return value
}

function owner(contributions: Float64Array): number {
  let best = 0
  for (let body = 1; body < contributions.length; body += 1) if (contributions[body] > contributions[best]) best = body
  return best
}

function powerOf(blend: number): number {
  return lerp(HARD_POWER, 1, clamp(blend, 0, 1))
}

// How far inside the bodies a point is: above 0 inside, 0 on the surface.
export function inside(placed: Placed[], blend: number, x: number, y: number, z: number): number {
  return 1 - reach(placed, powerOf(blend), x, y, z, new Float64Array(4), new Float64Array(placed.length))
}

export function project(x: number, y: number, z: number): [number, number] {
  const scale = EYE_UNITS / (EYE_UNITS - z)
  return [CENTER.x + (x - CENTER.x) * scale, CENTER.y + (y - CENTER.y) * scale]
}

// The facets the viewer sees the front of, back to front.
export function facets(pose: Pose, motion: Motion, meshes: Mesh[]): Facet[] {
  const placed = place(pose, motion)
  const power = powerOf(pose.blend)
  const bodies = placed.length
  const contributions = new Float64Array(bodies)
  const gradient = new Float64Array(4)
  const out: Facet[] = []
  placed.forEach((center, body) => {
    const { directions, relief, faces } = meshes[body]
    const count = relief.length
    const positions = new Float64Array(count * 3)
    const weights = new Float64Array(count * bodies)
    const foreign = new Uint8Array(count)
    const deep = new Uint8Array(count)
    const m = center.spin
    const longest = NEWTON.maxStepRadii * center.r
    for (let vertex = 0; vertex < count; vertex += 1) {
      const dx = directions[vertex * 3]
      const dy = directions[vertex * 3 + 1]
      const dz = directions[vertex * 3 + 2]
      const ux = m[0] * dx + m[1] * dy + m[2] * dz
      const uy = m[3] * dx + m[4] * dy + m[5] * dz
      const uz = m[6] * dx + m[7] * dy + m[8] * dz
      let rho = center.r
      reach(placed, power, center.x + rho * ux, center.y + rho * uy, center.z + rho * uz, gradient, contributions)
      if (owner(contributions) !== body) {
        let low = 0
        for (let halving = 0; halving < SEAM_HALVINGS; halving += 1) {
          const middle = (low + rho) / 2
          reach(
            placed,
            power,
            center.x + middle * ux,
            center.y + middle * uy,
            center.z + middle * uz,
            gradient,
            contributions
          )
          if (owner(contributions) === body) low = middle
          else rho = middle
        }
        rho = low
        deep[vertex] = low < DEEP_RADII * center.r ? 1 : 0
      }
      let x = center.x + rho * ux
      let y = center.y + rho * uy
      let z = center.z + rho * uz
      // Whether contributions still hold the field where the vertex is.
      let current = false
      for (let step = 0; step < NEWTON.steps; step += 1) {
        const value = reach(placed, power, x, y, z, gradient, contributions)
        current = true
        if (Math.abs(value - 1) < NEWTON.closeRadii) break
        const slope2 = gradient[1] ** 2 + gradient[2] ** 2 + gradient[3] ** 2
        if (slope2 < 1e-12) break
        const along = (value - 1) / slope2
        const length = Math.abs(along) * Math.sqrt(slope2)
        const shrink = length > longest ? longest / length : 1
        x -= along * gradient[1] * shrink
        y -= along * gradient[2] * shrink
        z -= along * gradient[3] * shrink
        current = false
      }
      if (!current) reach(placed, power, x, y, z, gradient, contributions)
      foreign[vertex] = owner(contributions) === body ? 0 : 1
      // Relief only on the body's own ground, so a seam is not roughened twice.
      const lift = foreign[vertex] ? 0 : RELIEF[body] * relief[vertex] * center.r
      positions[vertex * 3] = x + lift * ux
      positions[vertex * 3 + 1] = y + lift * uy
      positions[vertex * 3 + 2] = z + lift * uz
      let total = 0
      for (let other = 0; other < bodies; other += 1) total += contributions[other] ** COLOR_POWER
      for (let other = 0; other < bodies; other += 1)
        weights[vertex * bodies + other] = contributions[other] ** COLOR_POWER / total
    }
    for (let at = 0; at < faces.length; at += 3) {
      if (deep[faces[at]] && deep[faces[at + 1]] && deep[faces[at + 2]]) continue
      const [ia, ib, ic] = [faces[at] * 3, faces[at + 1] * 3, faces[at + 2] * 3]
      const ax = positions[ia]
      const ay = positions[ia + 1]
      const az = positions[ia + 2]
      const e1 = [positions[ib] - ax, positions[ib + 1] - ay, positions[ib + 2] - az]
      const e2 = [positions[ic] - ax, positions[ic + 1] - ay, positions[ic + 2] - az]
      let normal = cross(e1, e2)
      const length = Math.hypot(normal[0], normal[1], normal[2])
      if (length < 1e-9) continue
      normal = normal.map(v => v / length)
      const mid = [ax + (e1[0] + e2[0]) / 3, ay + (e1[1] + e2[1]) / 3, az + (e2[2] + e1[2]) / 3]
      // Out of the surface is where the reach grows. A lone body's surface faces away from its center;
      // a fused one's need not, at a neck's waist, so there the field's gradient says.
      const seam = foreign[ia / 3] || foreign[ib / 3] || foreign[ic / 3]
      const blended = pose.blend > 0 || seam
      let surface = [mid[0] - center.x, mid[1] - center.y, mid[2] - center.z]
      if (blended) {
        reach(placed, power, mid[0], mid[1], mid[2], gradient, contributions)
        surface = [gradient[1], gradient[2], gradient[3]]
      }
      if (dot(normal, surface) < 0) normal = normal.map(v => -v)
      if (dot(normal, [CENTER.x - mid[0], CENTER.y - mid[1], EYE_UNITS - mid[2]]) <= 0) continue
      let facing = normal
      const slope = Math.hypot(surface[0], surface[1], surface[2])
      if (blended && slope > 1e-12 && (seam || dot(normal, surface) / slope < FOLDED_COSINE))
        facing = surface.map(v => v / slope)
      const share = new Array<number>(bodies).fill(0)
      for (const index of [ia / 3, ib / 3, ic / 3])
        for (let other = 0; other < bodies; other += 1) share[other] += weights[index * bodies + other] / 3
      const fade = 1 - DEPTH_FADE * clamp(-mid[2] / HOME_ORBIT_UNITS, 0, 1)
      const light = share.reduce((sum, weight, other) => sum + weight * pose.lights[other], 0) * fade
      const points: number[] = []
      for (const index of [ia, ib, ic]) {
        const [px, py] = project(positions[index], positions[index + 1], positions[index + 2])
        points.push(round(px), round(py))
      }
      const metal = pose.shine
      const lit = clamp(light, 0, 1)
      const finish = polish(facing, mid, points, motion.clockMs)
      const facet = {
        points,
        shade: round(lerp(clamp(dot(facing, LIGHT), 0, 1), finish.studio, metal), 3),
        weights: share.map(weight => round(weight, 3)),
        light: round(light, 3),
        metal: round(metal, 3),
        gloss: round(metal * lit * finish.rim, 3),
        // A seam face, mostly overlapped by its neighbours, would glint as a bright edge peeking out.
        sheen: round(clamp(metal * lit * ((seam ? 0 : finish.glint) + finish.sweep), 0, 1), 3),
        depth: mid[2],
      }
      out.push(facet)
      // Shaded as the crease it fills, so a slit between two bodies pressing together reads as the
      // shadow where they meet rather than a line of light.
      if (seam)
        out.push({
          ...facet,
          shade: round(facet.shade * UNDERLAY_SHADE, 3),
          gloss: 0,
          sheen: 0,
          points: grown(points, UNDERLAY_UNITS),
          depth: -Infinity,
        })
    }
  })
  return out.sort((p, q) => p.depth - q.depth)
}

// The blue lines around what the pose lights. Each runs the padding's distance out from the discs its
// bodies cover on screen, and as the bodies blend it rounds off the waists between them, so a ring
// round one body is a circle and the rings of fusing bodies fuse too.
export function outlines(pose: Pose, motion: Motion): Outline[] {
  const placed = place(pose, motion)
  const neck = OUTLINE.neckUnits * clamp(pose.blend, 0, 1)
  return pose.highlights.map(({ bodies, opacity }) => {
    const discs = bodies.map(body => {
      const { x, y, z, r } = placed[body]
      const [cx, cy] = project(x, y, z)
      const reach = 1 + RELIEF[body] + OUTLINE.swellRadii * pose.blend
      return { x: cx, y: cy, r: r * reach * (EYE_UNITS / (EYE_UNITS - z)) + OUTLINE.padUnits }
    })
    const depth = bodies.reduce((sum, body) => sum + placed[body].z, 0) / bodies.length
    return { segments: contour(discs, neck), opacity, depth, bodies }
  })
}

type Edge = "top" | "right" | "bottom" | "left"
// Where the contour crosses a cell, by which of its corners are inside: top left 8, top right 4,
// bottom right 2, bottom left 1. A saddle (5, 10) is split here; JOINED_SADDLES joins it.
const CONTOUR_EDGES: [Edge, Edge][][] = [
  [],
  [["left", "bottom"]],
  [["bottom", "right"]],
  [["left", "right"]],
  [["top", "right"]],
  [
    ["top", "right"],
    ["left", "bottom"],
  ],
  [["top", "bottom"]],
  [["left", "top"]],
  [["left", "top"]],
  [["top", "bottom"]],
  [
    ["left", "top"],
    ["bottom", "right"],
  ],
  [["top", "right"]],
  [["left", "right"]],
  [["bottom", "right"]],
  [["left", "bottom"]],
  [],
]
const JOINED_SADDLES: Record<number, [Edge, Edge][]> = {
  5: [
    ["left", "top"],
    ["bottom", "right"],
  ],
  10: [
    ["top", "right"],
    ["left", "bottom"],
  ],
}

// Marching squares over how far inside the discs a point is, their union's edge rounded by a smooth
// minimum of neck units: a distance, so the crossings fall evenly.
function contour(discs: { x: number; y: number; r: number }[], neck: number): number[] {
  const margin = Math.max(...discs.map(disc => disc.r)) * 0.6
  const x0 = Math.min(...discs.map(disc => disc.x - disc.r)) - margin
  const y0 = Math.min(...discs.map(disc => disc.y - disc.r)) - margin
  const cell = OUTLINE.cellUnits
  const columns = Math.ceil((Math.max(...discs.map(disc => disc.x + disc.r)) + margin - x0) / cell)
  const rows = Math.ceil((Math.max(...discs.map(disc => disc.y + disc.r)) + margin - y0) / cell)
  // Above 0 inside.
  const values = new Float64Array((columns + 1) * (rows + 1))
  for (let row = 0; row <= rows; row += 1) {
    for (let column = 0; column <= columns; column += 1) {
      const [x, y] = [x0 + column * cell, y0 + row * cell]
      let outside = Infinity
      for (const disc of discs) outside = smoothMin(outside, Math.hypot(x - disc.x, y - disc.y) - disc.r, neck)
      values[row * (columns + 1) + column] = -outside
    }
  }
  const at = (column: number, row: number) => values[row * (columns + 1) + column]
  const segments: number[] = []
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      // Corners clockwise from the top left.
      const a = at(column, row)
      const b = at(column + 1, row)
      const c = at(column + 1, row + 1)
      const d = at(column, row + 1)
      const shape = (a > 0 ? 8 : 0) | (b > 0 ? 4 : 0) | (c > 0 ? 2 : 0) | (d > 0 ? 1 : 0)
      if (shape === 0 || shape === 15) continue
      const [left, top] = [x0 + column * cell, y0 + row * cell]
      const cut = (p: number, q: number) => p / (p - q)
      const edges: Record<Edge, number[]> = {
        top: [left + cut(a, b) * cell, top],
        right: [left + cell, top + cut(b, c) * cell],
        bottom: [left + cut(d, c) * cell, top + cell],
        left: [left, top + cut(a, d) * cell],
      }
      // A saddle joins its inside corners through the middle when the middle is inside.
      const pairs = (a + b + c + d) / 4 > 0 ? (JOINED_SADDLES[shape] ?? CONTOUR_EDGES[shape]) : CONTOUR_EDGES[shape]
      for (const [from, to] of pairs)
        segments.push(round(edges[from][0]), round(edges[from][1]), round(edges[to][0]), round(edges[to][1]))
    }
  }
  return segments
}

// The smaller of two distances, rounded off over k where they are close (Quilez's polynomial).
function smoothMin(a: number, b: number, k: number): number {
  if (k <= 0 || !Number.isFinite(a)) return Math.min(a, b)
  const h = Math.max(k - Math.abs(a - b), 0) / k
  return Math.min(a, b) - (h * h * k) / 4
}

// A triangle's corners each moved this far out from its middle, so a sliver grows no spike.
function grown(points: number[], units: number): number[] {
  const cx = (points[0] + points[2] + points[4]) / 3
  const cy = (points[1] + points[3] + points[5]) / 3
  const out: number[] = []
  for (let at = 0; at < 6; at += 2) {
    const [dx, dy] = [points[at] - cx, points[at + 1] - cy]
    const length = Math.hypot(dx, dy) || 1
    out.push(round(points[at] + (dx / length) * units), round(points[at + 1] + (dy / length) * units))
  }
  return out
}

// What a polished facet facing this way, at this point in space and on screen, shows, each 0 to 1:
// the studio it reflects, its glint, its rim and the sweep passing over it.
function polish(facing: number[], at: number[], points: number[], clockMs: number) {
  const eye = normalize([CENTER.x - at[0], CENTER.y - at[1], EYE_UNITS - at[2]])
  const half = normalize([LIGHT[0] + eye[0], LIGHT[1] + eye[1], LIGHT[2] + eye[2]])
  const toward = dot(facing, eye)
  // The reflected view: how far up it looks, y being down.
  const up = -(2 * toward * facing[1] - eye[1])
  const sky = smoothstep(STUDIO.skyFrom, STUDIO.skyTo, up)
  // A facet turned nearly edge-on reflects the horizon at a graze; there the band, as a glint would,
  // draws a bright scratch, so neither shows on one.
  const square = smoothstep(0.15, 0.45, toward)
  const band = Math.exp(-(((up - STUDIO.bandAt) / STUDIO.bandWidth) ** 2)) * square
  const studio = clamp(STUDIO.floor + STUDIO.sky * sky + STUDIO.band * band, 0, 1)
  const glint = GLINT.strength * Math.max(0, dot(facing, half)) ** GLINT.power * square
  const rim = RIM.strength * (1 - clamp(toward, 0, 1)) ** RIM.power
  const progress = (clockMs % SWEEP.periodMs) / SWEEP.crossingMs
  let sweep = 0
  if (progress < 1) {
    const front = lerp(SWEEP_SPAN.from, SWEEP_SPAN.to, progress)
    const offset = along((points[0] + points[2] + points[4]) / 3, (points[1] + points[3] + points[5]) / 3) - front
    // Facets turned up to the studio's sky catch it most, so it glitters rather than wipes.
    sweep = SWEEP.strength * Math.exp(-((offset / SWEEP.widthUnits) ** 2)) * (0.35 + 0.65 * studio)
  }
  return { studio, glint, rim, sweep }
}

// How far along the sweep's diagonal a point on screen is.
function along(x: number, y: number): number {
  return x * Math.cos(SWEEP.angle) + y * Math.sin(SWEEP.angle)
}

// Where the band's middle starts and ends: two widths beyond the box's corners, so it enters and
// leaves unseen.
const SWEEP_SPAN = (() => {
  const reaches = [0, BACKDROP_WIDTH_UNITS].flatMap(x => [0, BACKDROP_HEIGHT_UNITS].map(y => along(x, y)))
  return { from: Math.min(...reaches) - 2 * SWEEP.widthUnits, to: Math.max(...reaches) + 2 * SWEEP.widthUnits }
})()

function smoothstep(from: number, to: number, value: number): number {
  const t = clamp((value - from) / (to - from), 0, 1)
  return t * t * (3 - 2 * t)
}

// How far a facet's color goes from the sky toward its hue, 0 to 1. By night a body glows where the
// light falls, ambient plus what faces it; at dawn (ink) it is drawn as shade on a pale sky, deepest
// on the far side. Silver's shadows run deeper.
export function tone(facet: Facet, ink = false): number {
  const ambient = AMBIENT * (1 - METAL.shadowDrop * facet.metal)
  return facet.light * (ambient + (1 - ambient) * (ink ? 1 - facet.shade : facet.shade))
}

// How far a facet's hue goes toward the metal's, 0 to 1.
export function silvering(facet: Facet): number {
  return METAL.hueMix * facet.metal
}

function cross(a: number[], b: number[]): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

function dot(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

function round(value: number, digits = 1): number {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}
