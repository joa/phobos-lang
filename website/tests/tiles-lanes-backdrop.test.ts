import { describe, expect, it } from "vitest"
import {
  BACKDROP_HEIGHT_UNITS,
  BACKDROP_WIDTH_UNITS,
  beatPose,
  facets,
  inside,
  lerpPose,
  makeMeshes,
  orbitSpeed,
  axisSpeed,
  outlines,
  place,
  project,
  settled,
  silvering,
  START_MOTION,
  tone,
  type Facet,
  type Motion,
  type Pose,
} from "../src/components/home/tiles-lanes/backdrop"
import { makeMesh, multiply, rotateX, rotateY, rotateZ } from "../src/components/home/tiles-lanes/backdrop-mesh"
import { easeInOut } from "../src/lib/figure/math"

const meshes = makeMeshes()
const END_MS = 1e6
const TAU = 2 * Math.PI
// Orbit angles around a whole turn, the flattened orbits pointing several ways, and clocks across
// the fused body's float and sway.
const ANGLES = Array.from({ length: 24 }, (_, step) => (step / 24) * TAU)
const AXES = [0, 0.8, 1.6, 2.4]
const CLOCKS_MS = [0, 1800, 3650, 5500]
const motions: Motion[] = ANGLES.flatMap((orbitRad, step) =>
  CLOCKS_MS.map(clockMs => ({ orbitRad, axisRad: AXES[step % AXES.length], clockMs }))
)

// Whether the surface joins two bodies: every point on the line between their centers is inside.
function joined(pose: Pose, motion: Motion, a: number, b: number): boolean {
  const placed = place(pose, motion)
  const [p, q] = [placed[a], placed[b]]
  for (let step = 0; step <= 100; step += 1) {
    const t = step / 100
    const at = [p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t, p.z + (q.z - p.z) * t]
    if (inside(placed, pose.blend, at[0], at[1], at[2]) <= 0) return false
  }
  return true
}

const PAIRS = [
  [0, 1],
  [0, 2],
  [1, 2],
]

// A body's facets, while the bodies are apart: the ones nearly all its color.
function own(shown: Facet[], body: number): Facet[] {
  return shown.filter(facet => facet.weights[body] > 0.9)
}

function area(facet: Facet): number {
  const [ax, ay, bx, by, cx, cy] = facet.points
  return Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2
}

describe("tiles-lanes backdrop meshes", () => {
  it("are the same for the same seed, and closed geodesic spheres", () => {
    expect(makeMesh(3, 3, 0.3)).toEqual(makeMesh(3, 3, 0.3))
    for (const mesh of meshes) {
      expect(mesh.faces.length / 3).toBe(20 * 3 * 3)
      expect(mesh.relief.length).toBe(10 * 3 * 3 + 2)
      for (let vertex = 0; vertex < mesh.relief.length; vertex += 1) {
        const [x, y, z] = mesh.directions.subarray(vertex * 3, vertex * 3 + 3)
        expect(Math.hypot(x, y, z)).toBeCloseTo(1, 9)
        expect(Math.abs(mesh.relief[vertex])).toBeLessThanOrEqual(1)
      }
      // Closed: every edge is shared by exactly two faces.
      const edges = new Map<string, number>()
      for (let at = 0; at < mesh.faces.length; at += 3) {
        for (let side = 0; side < 3; side += 1) {
          const [a, b] = [mesh.faces[at + side], mesh.faces[at + ((side + 1) % 3)]]
          const key = a < b ? `${a},${b}` : `${b},${a}`
          edges.set(key, (edges.get(key) ?? 0) + 1)
        }
      }
      expect([...edges.values()].every(count => count === 2)).toBe(true)
    }
  })

  it("turn rigidly", () => {
    const m = multiply(multiply(rotateZ(0.4), rotateX(-0.3)), rotateY(1.7))
    for (let col = 0; col < 3; col += 1) expect(Math.hypot(m[col], m[3 + col], m[6 + col])).toBeCloseTo(1, 12)
  })
})

describe("tiles-lanes backdrop bodies", () => {
  it("are spheres while apart", () => {
    const pose = beatPose("tiles", 0)
    for (const motion of [START_MOTION, { orbitRad: 1.6, axisRad: 0, clockMs: 4000 }]) {
      const placed = place(pose, motion)
      for (const { x, y, z, r } of placed) {
        for (const [dx, dy, dz] of [
          [1, 0, 0],
          [0, -1, 0],
          [0, 0.6, 0.8],
          [-0.48, 0.6, -0.64],
        ])
          expect(Math.abs(inside(placed, pose.blend, x + r * dx, y + r * dy, z + r * dz))).toBeLessThan(0.01)
      }
    }
  })

  it("draw each lone body within its disc on screen, and fill it", () => {
    const pose = beatPose("tiles", 0)
    const motion = { orbitRad: 0.3, axisRad: 0, clockMs: 2000 }
    const shown = facets(pose, motion, meshes)
    place(pose, motion).forEach(({ x, y, z, r }, body) => {
      const [cx, cy] = project(x, y, z)
      const [ex] = project(x + r, y, z)
      const radius = ex - cx
      const mine = own(shown, body)
      for (const facet of mine)
        for (let at = 0; at < 6; at += 2)
          expect(Math.hypot(facet.points[at] - cx, facet.points[at + 1] - cy)).toBeLessThan(radius * 1.1)
      const filled = mine.reduce((sum, facet) => sum + area(facet), 0) / (Math.PI * radius * radius)
      expect(filled).toBeGreaterThan(0.85)
      expect(filled).toBeLessThan(1.1)
    })
  })

  it("keeps every facet in range: shares of the hues, shade, tone, silver and shine", () => {
    for (const pose of [beatPose("tiles", 0), beatPose("mixed", 2600), beatPose("mixed", END_MS), settled().pose]) {
      for (const facet of facets(pose, { orbitRad: 0.9, axisRad: 0, clockMs: 1800 }, meshes)) {
        expect(facet.weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 2)
        for (const value of [facet.shade, facet.metal, facet.gloss, facet.sheen, silvering(facet)]) {
          expect(value).toBeGreaterThanOrEqual(0)
          expect(value).toBeLessThanOrEqual(1)
        }
        for (const ink of [false, true]) {
          expect(tone(facet, ink)).toBeGreaterThanOrEqual(0)
          expect(tone(facet, ink)).toBeLessThanOrEqual(1)
        }
        for (const value of facet.points) expect(Number.isFinite(value)).toBe(true)
      }
    }
  })

  it("draws every pose inside its box, back to front", () => {
    for (const pose of [beatPose("tiles", 0), beatPose("mixed", 2600), beatPose("mixed", END_MS)]) {
      for (const motion of motions.filter((_, index) => index % 3 === 0)) {
        const shown = facets(pose, motion, meshes)
        for (let at = 1; at < shown.length; at += 1) expect(shown[at].depth).toBeGreaterThanOrEqual(shown[at - 1].depth)
        for (const facet of shown) {
          for (let at = 0; at < 6; at += 2) {
            expect(facet.points[at]).toBeGreaterThan(0)
            expect(facet.points[at]).toBeLessThan(BACKDROP_WIDTH_UNITS)
            expect(facet.points[at + 1]).toBeGreaterThan(0)
            expect(facet.points[at + 1]).toBeLessThan(BACKDROP_HEIGHT_UNITS)
          }
        }
      }
    }
  })
})

describe("tiles-lanes backdrop motion", () => {
  it("orbits in depth: the first body passes in front of Phobos without touching it", () => {
    const pose = beatPose("tiles", 0)
    const transit = ANGLES.find(orbitRad => {
      const [first, , phobos] = place(pose, { orbitRad, axisRad: 0, clockMs: 0 })
      const [fx, fy] = project(first.x, first.y, first.z)
      const [px, py] = project(phobos.x, phobos.y, phobos.z)
      return first.z > phobos.z && Math.hypot(fx - px, fy - py) < first.r + phobos.r
    })
    expect(transit).toBeDefined()
    expect(joined(pose, { orbitRad: transit ?? 0, axisRad: 0, clockMs: 0 }, 0, 2)).toBe(false)
  })

  it("draws a body larger near the viewer than behind", () => {
    const pose = beatPose("tiles", 0)
    const drawn = (orbitRad: number) =>
      own(facets(pose, { orbitRad, axisRad: 0, clockMs: 0 }, meshes), 0).reduce((sum, facet) => sum + area(facet), 0)
    // At a quarter turn the first body is nearest, at three quarters furthest.
    expect(place(pose, { orbitRad: Math.PI / 2, axisRad: 0, clockMs: 0 })[0].z).toBeGreaterThan(200)
    expect(drawn(Math.PI / 2) / drawn((3 * Math.PI) / 2)).toBeGreaterThan(1.3)
  })

  it("turns each body about its own axis", () => {
    const pose = beatPose("tiles", 0)
    const phobos = (clockMs: number) => own(facets(pose, { orbitRad: 0, axisRad: 0, clockMs }, meshes), 2)
    const [before, after] = [phobos(0), phobos(4000)]
    expect(place(pose, { orbitRad: 0, axisRad: 0, clockMs: 0 })[2]).toMatchObject({ x: 500, y: 350, z: 0 })
    expect(place(pose, { orbitRad: 0, axisRad: 0, clockMs: 4000 })[2]).toMatchObject({ x: 500, y: 350, z: 0 })
    expect(after.map(facet => facet.points)).not.toEqual(before.map(facet => facet.points))
  })

  it("floats and rocks the fused body up and down, and only that", () => {
    const heights = (pose: Pose, body: number) =>
      Array.from({ length: 40 }, (_, step) => place(pose, { orbitRad: 0, axisRad: 0, clockMs: step * 500 })[body].y)
    const spread = (values: number[]) => Math.max(...values) - Math.min(...values)
    const fused = beatPose("mixed", END_MS)
    expect(spread(heights(fused, 2))).toBeGreaterThan(25)
    // At the orbit's side, where the tilt alone would hold a lobe level.
    expect(spread(heights(fused, 0))).toBeGreaterThan(50)
    expect(spread(heights(beatPose("tiles", 0), 2))).toBe(0)
    expect(spread(heights(beatPose("tiles", 0), 0))).toBe(0)
  })

  it("swings quicker in beat 3 than it circles, and turns the swing's line only there", () => {
    const home = orbitSpeed(beatPose("tiles", 0))
    const swing = orbitSpeed(beatPose("mixed", END_MS))
    expect(home).toBeGreaterThan(0)
    expect(swing).toBeGreaterThan(home * 2)
    expect(axisSpeed(beatPose("tiles", 0))).toBe(0)
    expect(axisSpeed(beatPose("lanes", 0))).toBe(0)
    expect(axisSpeed(beatPose("mixed", END_MS))).toBeGreaterThan(0)
    expect(axisSpeed(beatPose("mixed", END_MS))).toBeLessThan(swing)
  })
})

describe("tiles-lanes backdrop outline", () => {
  // Every segment end, as x, y pairs.
  const ends = (segments: number[]) =>
    Array.from({ length: segments.length / 2 }, (_, at) => [segments[at * 2], segments[at * 2 + 1]])

  it("rings the beat's own body: a circle a padding out from it, and from its relief", () => {
    for (const [beat, body] of [
      ["tiles", 0],
      ["lanes", 1],
    ] as const) {
      const pose = beatPose(beat, 0)
      for (const motion of motions.filter((_, index) => index % 5 === 0)) {
        const [outline, ...rest] = outlines(pose, motion)
        expect(rest).toEqual([])
        expect(outline).toMatchObject({ bodies: [body], opacity: 1 })
        const { x, y, z, r } = place(pose, motion)[body]
        const [cx, cy] = project(x, y, z)
        const radius = project(x + r, y, z)[0] - cx
        expect(outline.depth).toBe(z)
        for (const [px, py] of ends(outline.segments)) {
          expect(Math.hypot(px - cx, py - cy) - radius).toBeGreaterThan(14)
          expect(Math.hypot(px - cx, py - cy) - radius).toBeLessThan(18 + radius * 0.07)
        }
      }
    }
  })

  it("rings the fused body whole, and never cuts into it", () => {
    for (const tMs of [600, 1200, END_MS]) {
      const pose = beatPose("mixed", tMs)
      for (const motion of motions.filter((_, index) => index % 3 === 0)) {
        const [outline] = outlines(pose, motion)
        expect(outline.bodies).toEqual([0, 1, 2])
        const corners = facets(pose, motion, meshes).flatMap(facet => ends(facet.points))
        for (const [px, py] of ends(outline.segments)) {
          const nearest = Math.min(...corners.map(([qx, qy]) => Math.hypot(px - qx, py - qy)))
          expect(nearest).toBeGreaterThan(8)
        }
      }
    }
  })

  it("closes every line, inside the box", () => {
    for (const pose of [beatPose("tiles", 0), beatPose("mixed", 600), beatPose("mixed", END_MS)]) {
      for (const motion of motions.filter((_, index) => index % 4 === 0)) {
        for (const outline of outlines(pose, motion)) {
          const count = new Map<string, number>()
          for (const [px, py] of ends(outline.segments)) {
            count.set(`${px},${py}`, (count.get(`${px},${py}`) ?? 0) + 1)
            expect(px).toBeGreaterThan(0)
            expect(px).toBeLessThan(BACKDROP_WIDTH_UNITS)
            expect(py).toBeGreaterThan(0)
            expect(py).toBeLessThan(BACKDROP_HEIGHT_UNITS)
          }
          expect([...count.values()].every(seen => seen % 2 === 0)).toBe(true)
        }
      }
    }
  })

  it("crosses over from ring to ring between beats", () => {
    const halfway = lerpPose(beatPose("tiles", 0), beatPose("lanes", 0), 0.5).highlights
    expect(halfway).toEqual([
      { bodies: [0], opacity: 0.5 },
      { bodies: [1], opacity: 0.5 },
    ])
    // A ring both beats share holds.
    const replay = lerpPose(beatPose("mixed", END_MS), beatPose("mixed", 0), 0.3).highlights
    expect(replay).toEqual([{ bodies: [0, 1, 2], opacity: 1 }])
    expect(lerpPose(beatPose("tiles", 0), beatPose("lanes", 0), 1).highlights).toEqual([{ bodies: [1], opacity: 1 }])
    // The page without script rings the fused body, so nothing passes in front of its line.
    const { pose, motion } = settled()
    expect(outlines(pose, motion).map(outline => outline.bodies)).toEqual([[0, 1, 2]])
  })
})

describe("tiles-lanes backdrop story", () => {
  it("makes Gluon the smallest body", () => {
    const [tiles, gluon, phobos] = place(beatPose("tiles", 0), START_MOTION).map(body => body.r)
    expect(gluon).toBeLessThan(Math.min(tiles, phobos))
  })

  it("lights the beat's own body", () => {
    const [tiles, lanes] = [beatPose("tiles", 0).lights, beatPose("lanes", 0).lights]
    expect(tiles[0]).toBeGreaterThan(Math.max(tiles[1], tiles[2]))
    expect(lanes[1]).toBeGreaterThan(Math.max(lanes[0], lanes[2]))
    const mixed = beatPose("mixed", END_MS).lights
    expect(Math.min(...mixed)).toBeGreaterThan(Math.max(tiles[1], tiles[2]))
  })

  it("keeps the bodies apart and unblended through tiles and lanes, all the way round", () => {
    for (const beat of ["tiles", "lanes"] as const) {
      const pose = beatPose(beat, 0)
      expect(pose.blend).toBe(0)
      for (const motion of motions) for (const [a, b] of PAIRS) expect(joined(pose, motion, a, b)).toBe(false)
    }
    // Between the two beats only the light moves.
    for (const t of [0.25, 0.5, 0.75]) expect(lerpPose(beatPose("tiles", 0), beatPose("lanes", 0), t).blend).toBe(0)
  })

  it("starts beat 3 where the others leave the bodies, and blends them by its end", () => {
    const start = beatPose("mixed", 0)
    expect([start.blend, start.squash]).toEqual([0, 0])
    expect(start.orbitUnits).toBe(beatPose("lanes", 0).orbitUnits)
    expect(beatPose("mixed", END_MS).blend).toBe(1)
    // The page without script shows both bodies joined to Phobos, one sunk into it.
    const { pose, motion } = settled()
    expect(joined(pose, motion, 0, 2)).toBe(true)
    expect(joined(pose, motion, 1, 2)).toBe(true)
  })

  it("keeps a body fused with Phobos from early in beat 3, whatever the swing", () => {
    for (const tMs of [1600, END_MS])
      for (const motion of motions)
        expect(joined(beatPose("mixed", tMs), motion, 0, 2) || joined(beatPose("mixed", tMs), motion, 1, 2)).toBe(true)
  })

  it("swings each body right through Phobos and out until its neck snaps", () => {
    const pose = beatPose("mixed", END_MS)
    expect(pose.squash).toBe(1)
    for (const body of [0, 1]) {
      for (const axisRad of AXES) {
        const turn = Array.from({ length: 96 }, (_, step) => ({ orbitRad: (step / 96) * TAU, axisRad, clockMs: 0 }))
        const gap = (motion: Motion) => {
          const placed = place(pose, motion)
          return Math.hypot(placed[body].x - placed[2].x, placed[body].y - placed[2].y, placed[body].z - placed[2].z)
        }
        // Through the middle...
        expect(Math.min(...turn.map(gap))).toBeLessThan(10)
        // ...and out the far side and back, apart at each end of the swing.
        expect(turn.filter(motion => !joined(pose, motion, body, 2)).length).toBeGreaterThan(1)
        expect(turn.filter(motion => joined(pose, motion, body, 2)).length).toBeGreaterThan(turn.length / 2)
      }
    }
  })

  it("turns only the fused body to polished silver", () => {
    for (const beat of ["tiles", "lanes"] as const) {
      expect(beatPose(beat, 0).shine).toBe(0)
      for (const facet of facets(beatPose(beat, 0), START_MOTION, meshes))
        expect([facet.metal, facet.gloss, facet.sheen, silvering(facet)]).toEqual([0, 0, 0, 0])
    }
    expect(beatPose("mixed", 0).shine).toBe(0)
    const fused = beatPose("mixed", END_MS)
    expect(fused.shine).toBe(1)
    const shown = facets(fused, { orbitRad: 0.9, axisRad: 0, clockMs: 1800 }, meshes)
    // Silver, with a hint of each body's hue left.
    for (const facet of shown) {
      expect(silvering(facet)).toBeGreaterThan(0.6)
      expect(silvering(facet)).toBeLessThan(1)
    }
    // Metal's contrast: facets that mirror the dark ground beside ones that mirror the bright sky.
    const shades = shown.map(facet => facet.shade)
    expect(Math.min(...shades)).toBeLessThan(0.3)
    expect(Math.max(...shades)).toBeGreaterThan(0.85)
    // As it turns, a few facets at a time glint past how faint the bodies are drawn, most do not.
    const frames = motions.filter((_, index) => index % 12 === 0).map(motion => facets(fused, motion, meshes))
    const glinting = frames.map(frame => frame.filter(facet => facet.sheen > 0.4).length)
    expect(glinting.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(0)
    frames.forEach((frame, index) => expect(glinting[index]).toBeLessThan(frame.length / 4))
    // Gone by the time the pills leave beat 3.
    expect(lerpPose(fused, beatPose("tiles", 0), easeInOut(0.5)).shine).toBe(0)
  })

  it("sweeps a band of light across the silver now and then", () => {
    const fused = beatPose("mixed", END_MS)
    const shine = (clockMs: number) =>
      facets(fused, { orbitRad: 0.4, axisRad: 0, clockMs }, meshes).reduce((sum, facet) => sum + facet.sheen, 0)
    const mean = (clocks: number[]) => clocks.reduce((sum, clockMs) => sum + shine(clockMs), 0) / clocks.length
    // One period is 5.2 s and the band takes 1.5 s to cross; it is over the body mid-crossing.
    const crossing = [5800, 5900, 6000, 6100, 6200]
    const resting = [7000, 7600, 8200, 8800, 9400]
    expect(mean(crossing)).toBeGreaterThan(mean(resting) * 1.5)
  })

  it("parts the bodies before the pills leave beat 3", () => {
    // The player switches the pills and the code halfway through a transition.
    const leaving = (p: number) => lerpPose(beatPose("mixed", END_MS), beatPose("tiles", 0), easeInOut(p))
    for (const motion of motions) {
      expect(joined(leaving(0), motion, 0, 2) || joined(leaving(0), motion, 1, 2)).toBe(true)
      for (const p of [0.5, 0.6, 0.8, 1])
        for (const [a, b] of PAIRS) expect(joined(leaving(p), motion, a, b)).toBe(false)
    }
  })

  it("keeps the page's static frame small", () => {
    const { pose, motion } = settled()
    const shown = facets(pose, motion, meshes)
    expect(shown.length).toBeGreaterThan(120)
    expect(shown.length).toBeLessThan(320)
  })
})
