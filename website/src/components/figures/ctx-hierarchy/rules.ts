// SPEC's rules for the ctx fields, bounds and masking, as the ctx and language figures draw them.
// Pure, so the figures and the tests share one source:
//   ctx.warp = ctx.thread / ctx.lanes, ctx.lane = ctx.thread % ctx.lanes, 32 lanes on NVIDIA,
//   @launch thread counts are positive multiples of 32, grid extents are dims over tiles rounded up,
//   a slice past a tensor's runtime extent is masked unless @aligned promises whole tiles.

export const LANES = 32
export const DEFAULT_THREADS = 256
export const MAX_THREADS = 1024
// Tensor extents are assumed to be multiples of this.
export const EXTENT_MULTIPLE = 4

export function isWholeWarps(threads: number, lanes = LANES): boolean {
  return Number.isInteger(threads) && threads > 0 && threads % lanes === 0
}

export function warpsOf(threads: number, lanes = LANES): number {
  if (!isWholeWarps(threads, lanes)) throw new Error(`${threads} threads are not a whole number of warps`)
  return threads / lanes
}

export function warpOf(thread: number, lanes = LANES): number {
  return Math.floor(thread / lanes)
}

export function laneOf(thread: number, lanes = LANES): number {
  return thread % lanes
}

export function threadOf(warp: number, lane: number, lanes = LANES): number {
  return warp * lanes + lane
}

// The CTAs a launch needs along one axis to cover an extent with tiles of a size.
export function gridExtent(extent: number, tile: number): number {
  return Math.ceil(extent / tile)
}

// The values of `for k in range(start, end, step)`.
export function strided(start: number, end: number, step: number): number[] {
  const values: number[] = []
  for (let k = start; k < end; k += step) values.push(k)
  return values
}

// The part of the slice `start :+ length` that lies inside an extent of `extent`.
export function inBounds(start: number, length: number, extent: number): number {
  return Math.max(0, Math.min(length, extent - start))
}

// Whether element `index` of the slice `start :+ length` is masked: past the extent, and the extent
// is not promised whole tiles.
export function isMasked(start: number, index: number, extent: number, aligned: boolean): boolean {
  return !aligned && start + index >= extent
}

export interface Block {
  x: number
  y: number
}

export interface ThreadPath {
  block: Block
  warp: number
  lane: number
  thread: number
}

// How the readouts name one thread, such as "block (2, 1), warp 3, lane 17, thread 113".
export function describeThread({ block, warp, lane, thread }: ThreadPath): string {
  return `block (${block.x}, ${block.y}), warp ${warp}, lane ${lane}, thread ${thread}`
}
