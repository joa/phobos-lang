// A small perspective camera for the tiles-and-lanes scene. World units are tensor elements, y is up.
// The camera orbits its target: yaw turns it around the vertical axis, pitch raises it, and focal is
// fixed while distance changes, so moving closer zooms. A view then fits the projection into the
// viewBox. Pure, so the build renders the same frame the browser starts from.

import { lerp } from "../../../lib/figure/math"

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface Camera {
  target: Vec3
  yawDeg: number
  pitchDeg: number
  distance: number
  focal: number
}

// Where the projection's origin lands in the viewBox and how many units one projected unit is.
export interface View {
  scale: number
  cx: number
  cy: number
}

export interface Projected {
  x: number
  y: number
  // How much nearer than the target the point is: above 1 nearer, below 1 further.
  near: number
}

export function vec(x: number, y: number, z: number): Vec3 {
  return { x, y, z }
}

const DEG = Math.PI / 180
// Points behind the camera are held just in front of it rather than flipped.
const MIN_DEPTH = 1

export function project(point: Vec3, camera: Camera, view: View): Projected {
  const yaw = camera.yawDeg * DEG
  const pitch = camera.pitchDeg * DEG
  const dx = point.x - camera.target.x
  const dy = point.y - camera.target.y
  const dz = point.z - camera.target.z
  const x1 = dx * Math.cos(yaw) - dz * Math.sin(yaw)
  const z1 = dx * Math.sin(yaw) + dz * Math.cos(yaw)
  const y2 = dy * Math.cos(pitch) - z1 * Math.sin(pitch)
  const z2 = dy * Math.sin(pitch) + z1 * Math.cos(pitch)
  const depth = Math.max(MIN_DEPTH, camera.distance - z2)
  const perspective = camera.focal / depth
  return {
    x: view.cx + view.scale * x1 * perspective,
    y: view.cy - view.scale * y2 * perspective,
    near: camera.distance / depth,
  }
}

export interface Box {
  x: number
  y: number
  width: number
  height: number
}

// The view that fits the points' projection into the box, centered.
export function fitView(points: Vec3[], camera: Camera, box: Box): View {
  const raw = points.map(point => project(point, camera, { scale: 1, cx: 0, cy: 0 }))
  const minX = Math.min(...raw.map(p => p.x))
  const maxX = Math.max(...raw.map(p => p.x))
  const minY = Math.min(...raw.map(p => p.y))
  const maxY = Math.max(...raw.map(p => p.y))
  const scale = Math.min(box.width / (maxX - minX || 1), box.height / (maxY - minY || 1))
  return {
    scale,
    cx: box.x + box.width / 2 - scale * ((minX + maxX) / 2),
    cy: box.y + box.height / 2 - scale * ((minY + maxY) / 2),
  }
}

export function lerpVec(a: Vec3, b: Vec3, t: number): Vec3 {
  return { x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), z: lerp(a.z, b.z, t) }
}

export function lerpCamera(a: Camera, b: Camera, t: number): Camera {
  return {
    target: lerpVec(a.target, b.target, t),
    yawDeg: lerp(a.yawDeg, b.yawDeg, t),
    pitchDeg: lerp(a.pitchDeg, b.pitchDeg, t),
    // Distance eases geometrically, so a dolly feels even rather than rushing at the end.
    distance: a.distance * (b.distance / a.distance) ** t,
    focal: lerp(a.focal, b.focal, t),
  }
}

export function lerpView(a: View, b: View, t: number): View {
  return { scale: lerp(a.scale, b.scale, t), cx: lerp(a.cx, b.cx, t), cy: lerp(a.cy, b.cy, t) }
}

// A point on the quadratic Bezier from a through control c to b.
export function bezier3(a: Vec3, c: Vec3, b: Vec3, t: number): Vec3 {
  const u = 1 - t
  return {
    x: u * u * a.x + 2 * u * t * c.x + t * t * b.x,
    y: u * u * a.y + 2 * u * t * c.y + t * t * b.y,
    z: u * u * a.z + 2 * u * t * c.z + t * t * b.z,
  }
}
