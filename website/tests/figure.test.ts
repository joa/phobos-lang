import { describe, expect, it } from "vitest"
import { cubicBezier, easeInOut, easeInOutSine, easeOut, easeOutCubic, mulberry32 } from "../src/lib/figure/math"
import {
  graticulePoints,
  layoutFor,
  MIN_LABEL_UNITS,
  NARROW_BELOW_PX,
  pxPerUnit,
  registrationTicksPath,
} from "../src/lib/figure/layout"
import { elbow, Polyline } from "../src/lib/figure/polyline"
import { LOOP_HOLD_MS, LoopClock, Timeline } from "../src/lib/figure/timeline"
import { parseDurationMs } from "../src/lib/figure/motion"
import { ChartFigure } from "../src/lib/figure/chart-figure"
import { plate } from "../src/lib/figure/draw"
import { cloudParticles, cloudSpread, MAX_PARTICLES, StarPacket } from "../src/lib/figure/packet"

describe("Timeline", () => {
  function recorder() {
    const calls: string[] = []
    const step = (name: string, durationMs: number) => ({
      durationMs,
      apply: (progress: number) => calls.push(`${name}:${Math.round(progress * 100) / 100}`),
    })
    return { calls, step }
  }

  it("runs steps back to back and carries leftover time", () => {
    const { calls, step } = recorder()
    const timeline = new Timeline([step("a", 100), step("b", 100)])
    expect(timeline.advance(50)).toBe(true)
    expect(timeline.advance(100)).toBe(true)
    expect(calls).toEqual(["a:0.5", "a:1", "b:0.5"])
    expect(timeline.advance(100)).toBe(false)
    expect(timeline.isFinished).toBe(true)
  })

  it("applies a 0 ms step at once", () => {
    const { calls, step } = recorder()
    const timeline = new Timeline([step("reset", 0), step("a", 100)])
    timeline.advance(0)
    expect(calls).toEqual(["reset:1", "a:0"])
  })

  it("holds at the end", () => {
    const { step } = recorder()
    const timeline = new Timeline([step("a", 100)], { holdMs: 200 })
    expect(timeline.totalMs).toBe(300)
    expect(timeline.advance(150)).toBe(true)
    expect(timeline.advance(150)).toBe(false)
  })

  it("seekEnd applies every step at 1 in order", () => {
    const { calls, step } = recorder()
    const timeline = new Timeline([step("a", 100), step("b", 0), step("c", 50)], { holdMs: 1000 })
    timeline.advance(30)
    timeline.seekEnd()
    expect(calls.slice(1)).toEqual(["a:1", "b:1", "c:1"])
    expect(timeline.isFinished).toBe(true)
  })

  it("reset rewinds without applying", () => {
    const { calls, step } = recorder()
    const timeline = new Timeline([step("a", 100)])
    timeline.advance(200)
    const count = calls.length
    timeline.reset()
    expect(calls.length).toBe(count)
    expect(timeline.isFinished).toBe(false)
  })
})

describe("LoopClock", () => {
  it("stays idle until the figure has run", () => {
    const clock = new LoopClock()
    expect(clock.tick(false, 5000)).toBe("idle")
  })

  it("holds the end state, then restarts once", () => {
    const clock = new LoopClock({ holdMs: 1500 })
    expect(clock.tick(true, 16)).toBe("run")
    expect(clock.tick(false, 1000)).toBe("hold")
    expect(clock.tick(false, 499)).toBe("hold")
    expect(clock.tick(false, 1)).toBe("restart")
    // A restart that played nothing does not loop again.
    expect(clock.tick(false, 5000)).toBe("idle")
  })

  it("starts the hold over when the figure runs again", () => {
    const clock = new LoopClock({ holdMs: 1500 })
    clock.tick(true, 16)
    clock.tick(false, 1400)
    expect(clock.tick(true, 16)).toBe("run")
    expect(clock.tick(false, 1400)).toBe("hold")
    expect(clock.heldMs).toBe(1400)
  })

  it("holds for about a second and a half by default", () => {
    expect(LOOP_HOLD_MS).toBe(1500)
  })
})

describe("packet cloud", () => {
  it("is deterministic per seed and capped", () => {
    expect(cloudParticles(14, 7)).toEqual(cloudParticles(14, 7))
    expect(cloudParticles(14, 7)).not.toEqual(cloudParticles(14, 8))
    expect(cloudParticles(500, 1)).toHaveLength(MAX_PARTICLES)
    expect(cloudParticles(0, 1)).toHaveLength(1)
  })

  it("puts particles on three depth layers, far first, sized and faded by depth", () => {
    const particles = cloudParticles(18, 3)
    expect(new Set(particles.map(particle => particle.layer))).toEqual(new Set([0, 1, 2]))
    expect(particles.map(particle => particle.layer)).toEqual([...particles.map(particle => particle.layer)].sort())
    for (const particle of particles) {
      expect(particle.radiusUnits).toBeGreaterThanOrEqual(0.8)
      expect(particle.radiusUnits).toBeLessThanOrEqual(2.2)
      expect(particle.opacity).toBeGreaterThan(0)
      expect(particle.opacity).toBeLessThanOrEqual(1)
      expect(particle.lagUnits).toBeGreaterThanOrEqual(0)
    }
    const near = particles.filter(particle => particle.layer === 2)
    const far = particles.filter(particle => particle.layer === 0)
    expect(Math.max(...near.map(particle => particle.radiusUnits))).toBeGreaterThan(
      Math.max(...far.map(particle => particle.radiusUnits))
    )
  })

  it("gathers to a point at both ends of a line", () => {
    expect(cloudSpread(0, 100, 13.5)).toBe(0)
    expect(cloudSpread(100, 100, 13.5)).toBe(0)
    expect(cloudSpread(50, 100, 13.5)).toBe(1)
    expect(cloudSpread(5, 100, 13.5)).toBeGreaterThan(0)
    expect(cloudSpread(5, 100, 13.5)).toBeLessThan(1)
  })
})

describe("easing", () => {
  const easings = { easeInOutSine, easeOutCubic, easeOut, easeInOut }
  for (const [name, easing] of Object.entries(easings)) {
    it(`${name} starts at 0, ends at 1 and never overshoots`, () => {
      expect(easing(0)).toBeCloseTo(0, 6)
      expect(easing(1)).toBeCloseTo(1, 6)
      let previous = 0
      for (let t = 0; t <= 1.0001; t += 0.01) {
        const value = easing(t)
        expect(value).toBeGreaterThanOrEqual(previous - 1e-9)
        expect(value).toBeLessThanOrEqual(1 + 1e-9)
        previous = value
      }
    })
  }

  it("cubicBezier matches the linear curve and known points", () => {
    const linear = cubicBezier(0, 0, 1, 1)
    for (const t of [0.1, 0.37, 0.5, 0.9]) expect(linear(t)).toBeCloseTo(t, 5)
    // ease-in-out is symmetric around its middle
    expect(easeInOut(0.5)).toBeCloseTo(0.5, 5)
    expect(easeInOut(0.25) + easeInOut(0.75)).toBeCloseTo(1, 5)
    // --ease-out arrives fast: well past halfway at a quarter of the time
    expect(easeOut(0.25)).toBeGreaterThan(0.7)
  })

  it("parses duration tokens", () => {
    expect(parseDurationMs("240ms", 0)).toBe(240)
    expect(parseDurationMs(" 0.9s", 0)).toBe(900)
    expect(parseDurationMs("fast", 7)).toBe(7)
  })
})

describe("mulberry32", () => {
  it("is deterministic and in [0, 1)", () => {
    const a = mulberry32(42)
    const b = mulberry32(42)
    for (let index = 0; index < 1000; index += 1) {
      const value = a()
      expect(value).toBe(b())
      expect(value).toBeGreaterThanOrEqual(0)
      expect(value).toBeLessThan(1)
    }
  })
})

describe("layout", () => {
  it("switches at the 600 px container breakpoint", () => {
    expect(layoutFor(NARROW_BELOW_PX - 1)).toBe("narrow")
    expect(layoutFor(NARROW_BELOW_PX)).toBe("wide")
    expect(layoutFor(390)).toBe("narrow")
  })

  it("keeps the smallest label at 10 px or more at either side of the breakpoint", () => {
    expect(MIN_LABEL_UNITS * pxPerUnit(NARROW_BELOW_PX)).toBeGreaterThanOrEqual(10)
    expect(MIN_LABEL_UNITS * pxPerUnit(340)).toBeGreaterThanOrEqual(10)
  })

  it("renders the smallest label at 11 px in a 640 px docs column", () => {
    expect(MIN_LABEL_UNITS * pxPerUnit(640)).toBeGreaterThanOrEqual(11)
  })

  it("centers the graticule", () => {
    const points = graticulePoints(100, 50, 16)
    const xs = [...new Set(points.map(([x]) => x))]
    expect(xs[0]).toBeCloseTo(100 - xs[xs.length - 1], 6)
    for (const [x, y] of points) {
      expect(x).toBeGreaterThan(0)
      expect(x).toBeLessThan(100)
      expect(y).toBeGreaterThan(0)
      expect(y).toBeLessThan(50)
    }
  })

  it("draws four corner ticks", () => {
    expect(registrationTicksPath(100, 50).match(/M/g)).toHaveLength(4)
  })
})

describe("Polyline", () => {
  const line = new Polyline([
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 20 },
  ])

  it("measures and walks at constant speed", () => {
    expect(line.lengthUnits).toBe(30)
    expect(line.pointAt(5)).toEqual({ x: 5, y: 0 })
    expect(line.pointAt(20)).toEqual({ x: 10, y: 10 })
    expect(line.pointAtFraction(1)).toEqual({ x: 10, y: 20 })
    expect(line.pointAt(-4)).toEqual({ x: 0, y: 0 })
  })

  it("routes an elbow with right angles", () => {
    const route = elbow({ x: 0, y: 0 }, { x: 20, y: 10 })
    expect(route.points).toEqual([
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
      { x: 20, y: 10 },
    ])
  })

  it("drops straight-run vertices when simplified", () => {
    const straight = elbow({ x: 0, y: 5 }, { x: 20, y: 5.0000000001 }).simplified()
    expect(straight.points.length).toBe(2)
    expect(line.simplified().points.length).toBe(3)
  })
})

class SampleFigure extends ChartFigure {
  override build(): void {
    this.setViewBox(720, 200)
    plate(this.drawing, this.id, { widthUnits: 720, heightUnits: 200, label: "FIG / SAMPLE", schematic: true })
    const packet = new StarPacket(this.drawing, { space: "shared" })
    packet.place(
      new Polyline([
        { x: 0, y: 0 },
        { x: 100, y: 0 },
      ]),
      50
    )
  }

  override settle(): void {
    this.setReadout(`mode ${this.state.mode}`)
  }
}

describe("static rendering", () => {
  it("runs build and settle without a browser", () => {
    const render = SampleFigure.renderStatic("wide", { mode: "a" })
    expect(render.viewBox).toBe("0 0 720 200")
    expect(render.readout).toBe("mode a")
    expect(render.drawing).toContain('class="chart-plate-text"')
    expect(render.drawing).toContain("FIG / SAMPLE")
    expect(render.drawing).toContain('class="chart-packet sp-shared"')
    expect(render.drawing).toContain('transform="translate(50 0)"')
    expect(render.drawing).not.toMatch(/style="[^"]*(color|fill|stroke):/)
  })

  it("prefixes ids inside the drawing with the given id", () => {
    expect(SampleFigure.renderStatic("wide", {}, "fig-one").drawing).toContain('id="fig-one-graticule"')
    expect(SampleFigure.renderStatic("narrow", {}, "fig-one-narrow").drawing).toContain('id="fig-one-narrow-graticule"')
  })
})
