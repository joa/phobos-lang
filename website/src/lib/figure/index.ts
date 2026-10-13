// The figure library: one import for figure modules.
export {
  ChartFigure,
  type FigureConstructor,
  type FigureState,
  type StateValue,
  type StaticRender,
} from "./chart-figure"
export * from "./draw"
export * from "./layout"
export * from "./math"
export {
  Motion,
  ARRIVAL_GLOW_MS,
  parseDurationMs,
  prefersReducedMotion,
  readMotionTokens,
  setBreathing,
  sweepWire,
  viewTransition,
  type MotionTokens,
} from "./motion"
export { StarPacket } from "./packet"
export { Polyline, elbow, type Point } from "./polyline"
export { SVG_NS, setAttributes, svgEl, svgText, type Attributes } from "./svg"
export { Timeline, type TimelineStep } from "./timeline"
