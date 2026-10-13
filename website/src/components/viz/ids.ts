// Ids for visualizer figures: unique on a page however many kernels it shows.
let counter = 0

export function nextVizId(name: string): string {
  counter += 1
  return `viz-${name.replace(/[^\w-]/g, "-")}-${counter}`
}
