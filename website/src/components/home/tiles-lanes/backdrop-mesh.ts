// The backdrop's bodies as meshes: a geodesic sphere, its vertices nudged off the regular grid so the
// facets are uneven, each carrying a little relief, so a turning body shows it in its outline too.
// Plus the 3 x 3 rotations that turn them. Pure, so the build and the browser draw the same bodies.

import { mulberry32 } from "../../../lib/figure/math"

export type Vec3 = [number, number, number]
// Row-major.
export type Mat3 = [number, number, number, number, number, number, number, number, number]

export interface Mesh {
  // A unit direction per vertex, x y z.
  directions: Float64Array
  // Per vertex, -1 to 1: how far it stands out of or sinks into the surface.
  relief: Float64Array
  // Three vertex indices per face.
  faces: Uint16Array
}

const PHI = (1 + Math.sqrt(5)) / 2
const ICOSAHEDRON_VERTICES: Vec3[] = [
  [-1, PHI, 0],
  [1, PHI, 0],
  [-1, -PHI, 0],
  [1, -PHI, 0],
  [0, -1, PHI],
  [0, 1, PHI],
  [0, -1, -PHI],
  [0, 1, -PHI],
  [PHI, 0, -1],
  [PHI, 0, 1],
  [-PHI, 0, -1],
  [-PHI, 0, 1],
]
const ICOSAHEDRON_FACES = [
  [0, 11, 5],
  [0, 5, 1],
  [0, 1, 7],
  [0, 7, 10],
  [0, 10, 11],
  [1, 5, 9],
  [5, 11, 4],
  [11, 10, 2],
  [10, 7, 6],
  [7, 1, 8],
  [3, 9, 4],
  [3, 4, 2],
  [3, 2, 6],
  [3, 6, 8],
  [3, 8, 9],
  [4, 9, 5],
  [2, 4, 11],
  [6, 2, 10],
  [8, 6, 7],
  [9, 8, 1],
]

// Each face of an icosahedron cut into frequency^2 triangles, every vertex pushed out onto the sphere.
// A vertex moves by up to jitter of an edge along the surface.
export function makeMesh(seed: number, frequency: number, jitter: number): Mesh {
  const random = mulberry32(seed)
  const points: Vec3[] = []
  const index = new Map<string, number>()
  const vertex = (v: Vec3) => {
    const unit = normalize(v)
    const key = unit.map(c => c.toFixed(5)).join(",")
    let at = index.get(key)
    if (at === undefined) {
      at = points.length
      points.push(unit)
      index.set(key, at)
    }
    return at
  }
  const faces: number[] = []
  for (const [ia, ib, ic] of ICOSAHEDRON_FACES) {
    const [a, b, c] = [ICOSAHEDRON_VERTICES[ia], ICOSAHEDRON_VERTICES[ib], ICOSAHEDRON_VERTICES[ic]]
    const at = (i: number, j: number) =>
      vertex([0, 1, 2].map(k => a[k] + ((b[k] - a[k]) * i + (c[k] - a[k]) * j) / frequency) as Vec3)
    for (let i = 0; i < frequency; i += 1) {
      for (let j = 0; i + j < frequency; j += 1) {
        faces.push(at(i, j), at(i + 1, j), at(i, j + 1))
        if (i + j < frequency - 1) faces.push(at(i + 1, j), at(i + 1, j + 1), at(i, j + 1))
      }
    }
  }
  // An edge of the cut faces, on the unit sphere, about 1.1 / frequency.
  const step = (jitter * 1.1) / frequency
  const directions = new Float64Array(points.length * 3)
  const relief = new Float64Array(points.length)
  points.forEach((point, at) => {
    const moved = normalize([
      point[0] + (random() - 0.5) * 2 * step,
      point[1] + (random() - 0.5) * 2 * step,
      point[2] + (random() - 0.5) * 2 * step,
    ])
    directions.set(moved, at * 3)
    relief[at] = random() * 2 - 1
  })
  return { directions, relief, faces: Uint16Array.from(faces) }
}

export function normalize(v: Vec3): Vec3 {
  const length = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / length, v[1] / length, v[2] / length]
}

export function multiply(a: Mat3, b: Mat3): Mat3 {
  const out = new Array(9).fill(0) as Mat3
  for (let row = 0; row < 3; row += 1)
    for (let col = 0; col < 3; col += 1)
      out[row * 3 + col] = a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col]
  return out
}

export function rotateX(angle: number): Mat3 {
  const [c, s] = [Math.cos(angle), Math.sin(angle)]
  return [1, 0, 0, 0, c, -s, 0, s, c]
}

export function rotateY(angle: number): Mat3 {
  const [c, s] = [Math.cos(angle), Math.sin(angle)]
  return [c, 0, s, 0, 1, 0, -s, 0, c]
}

export function rotateZ(angle: number): Mat3 {
  const [c, s] = [Math.cos(angle), Math.sin(angle)]
  return [c, -s, 0, s, c, 0, 0, 0, 1]
}
