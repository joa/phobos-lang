// The kernel visualizer: a trace stepped statement by statement, drawn as a star chart. The tile level
// shows the grid of CTAs, the global tensors with the picked CTA's boxes, shared memory as a byte
// ruler, the barriers in program order and the CTA's threads; the lane level shows warp 0's lanes,
// the lines they touch, shuffles as arcs and each value's uniformity.
//
// The page names the trace on the wrapper ([data-viz] with data-trace and data-chip); the module
// loads it, then starts the figure base. Packets move in the frame loop; every other change is a class
// or a transform that CSS transitions. Readouts change per step, never per frame.

import { ChartFigure, type StateValue } from "../figure/chart-figure"
import type { Layout } from "../figure/layout"
import { clamp } from "../figure/math"
import { elbow, Polyline, type Point } from "../figure/polyline"
import { Timeline, type TimelineStep } from "../figure/timeline"
import { buildElements, setRect, type SlotElements, type VizElements } from "./elements"
import { boxRect, kindName, slotRects, vizGeometry, type Rect, type VizGeometry } from "./geometry"
import { iterLabel, KernelModel, LINE_BYTES, type LineFacts, type StepState } from "./model"
import { precomputed, type TraceSource } from "./source"
import type { Access, Cta, LaneEvent } from "./trace"

type Zoom = "tile" | "lane"
type Origin = "reader" | "follow" | "tour"

const SPEED_UNITS_PER_MS = 0.3
const PACKET_STAGGER_MS = 110
const OP_STAGGER_MS = 320
const LANE_TRAVEL_MS = 520
const LANE_STAGGER_MS = 12
const STAGE_MS = 760
const PLAY_HOLD_MS = 950
const SWEEP_HOLD_MS = 700
const REDUCED_PLAY_MS = 1400
const LANE_SLOT_UNITS = 150

interface Route {
  line: Polyline
  delayMs: number
  travelMs: number
  space: "global" | "shared" | "reg"
}

export interface StepDetail {
  key: string
  fraction: number
  zoom: Zoom
  ctaIndex: number
}

export default class KernelViz extends ChartFigure {
  static source: TraceSource = precomputed

  #shell: HTMLElement | null = null
  #models = new Map<string, KernelModel>()
  #model: KernelModel | null = null
  #geometry: VizGeometry | null = null
  #els: VizElements | null = null
  #cta = 0
  #step = 0
  #zoom: Zoom = "tile"
  #allCtas = false
  #timeline: Timeline | null = null
  #isAutoPlaying = false
  #autoHoldMs = 0
  #sweepIndex = -1
  #sweepHoldMs = 0
  #reducedTimer = 0
  #lastStage = -1
  #codeBox: HTMLElement | null = null
  #codeLines: HTMLElement[] = []
  #marker: HTMLElement | null = null
  #lineMap: number[] = []

  override start(): void {
    // Two scripts on one page can each start the page's figures; the first start wins.
    if (this.root.hasAttribute("data-viz-booting")) return
    this.root.setAttribute("data-viz-booting", "")
    void this.#boot()
  }

  get model(): KernelModel | null {
    return this.#model
  }

  override get tour(): Record<string, StateValue>[] {
    if (!this.#shell?.hasAttribute("data-tour") || !this.#model) return []
    const count = this.#model.stepCount(this.#cta)
    const lanes = this.#model.hasLaneEvents
    const collective = this.#model
      .statements(this.#cta)
      .findIndex(statement => statement.steps.some(step => step.lanes?.kind === "collective"))
    if (lanes && collective >= 0) {
      return [{ step: 1 }, { zoom: "lane" }, { step: Math.min(count, 2) }, { step: collective + 1 }, { zoom: "tile" }]
    }
    return [{ step: 1 }, { step: Math.min(count, 2) }, { step: Math.min(count, 3) }, { zoom: "lane" }, { zoom: "tile" }]
  }

  async #boot(): Promise<void> {
    this.#shell = this.root.closest<HTMLElement>("[data-viz]")
    const name = this.#shell?.dataset.trace
    const pressedChip = this.root.querySelector<HTMLElement>('[data-group="chip"][aria-pressed="true"]')
    const chip = pressedChip?.dataset.value ?? this.#shell?.dataset.chip ?? "sm_75"
    if (!name) return
    try {
      this.#model = await this.#load(name, chip)
    } catch (error) {
      console.error(`Visualizer: ${String(error)}`)
      return
    }
    this.#readUrlState()
    this.#pressZoomButton()
    this.#setRangeMax()
    // Shown first, so the base measures the real width and builds the right layout once.
    this.#shell?.setAttribute("data-viz-live", "")
    super.start()
    this.#attachCode()
    this.#attachKeys()
    this.root.addEventListener("viz:follow", event => this.#onFollow(event as CustomEvent<StepDetail>))
    this.#sync()
  }

  async #load(name: string, chip: string): Promise<KernelModel> {
    const cached = this.#models.get(chip)
    if (cached) return cached
    const model = new KernelModel(await KernelViz.source.load(name, chip))
    this.#models.set(chip, model)
    return model
  }

  override build(layout: Layout): void {
    const model = this.#model
    if (!model) return
    this.#timeline = null
    this.#geometry = vizGeometry(model, layout)
    this.setViewBox(this.#geometry.widthUnits, this.#geometry.heightUnits)
    this.#els = buildElements(this.drawing, this.id, model, this.#geometry)
    this.#els.tileGroup.classList.toggle("viz-off", this.#zoom !== "tile")
    this.#els.laneGroup.classList.toggle("viz-off", this.#zoom !== "lane")
  }

  override settle(): void {
    this.motion.finishAll()
    this.#timeline?.seekEnd()
    this.#timeline = null
    this.#applyState()
    this.#settleLanes()
  }

  override update(dtMs: number): boolean {
    if (this.#timeline) {
      if (this.#timeline.advance(dtMs)) return true
      this.#timeline = null
    }
    if (this.#sweepIndex >= 0) {
      this.#sweepHoldMs += dtMs
      if (this.#sweepHoldMs < SWEEP_HOLD_MS) return true
      this.#sweepHoldMs = 0
      this.#sweepIndex += 1
      if (this.#sweepIndex >= (this.#model?.ctas.length ?? 0)) {
        this.#sweepIndex = -1
        return false
      }
      this.#pickCta(this.#sweepIndex, "reader")
      return true
    }
    if (this.#isAutoPlaying) {
      this.#autoHoldMs += dtMs
      if (this.#autoHoldMs < PLAY_HOLD_MS) return true
      this.#autoHoldMs = 0
      this.#advancePlay()
      return this.#isAutoPlaying || this.#timeline !== null
    }
    return false
  }

  override onFirstVisible(): void {
    if (!this.reducedMotion && this.#els) {
      const lines = this.#els.tileGroup.querySelectorAll<SVGGeometryElement>(".chart-box, .chart-wire")
      this.motion.drawIn([...lines].slice(0, 60))
      this.motion.enter(
        this.#els.entrance.filter(element => element.tagName === "text"),
        { delayMs: 240 }
      )
    }
    super.onFirstVisible()
    // Without a tour, the step the figure opens on plays once, so a butterfly is seen running.
    if (this.tour.length === 0) this.#animateStep()
  }

  override onInput(param: string, value: number): void {
    if (param === "step") this.#go(value - 1, "reader")
  }

  override onChange(group: string, value: string): void {
    if (group === "zoom") this.#setZoom(value === "lane" ? "lane" : "tile", "reader")
    else if (group === "cta") this.#pickCta(Number(value), "reader")
    else if (group === "chip") void this.#switchChip(value)
  }

  override onAction(action: string): void {
    const model = this.#model
    if (!model) return
    if (action !== "play") this.#stopPlay()
    switch (action) {
      case "next":
        this.#go(this.#step + 1, "reader")
        break
      case "prev":
        this.#go(this.#step - 1, "reader")
        break
      case "first":
        this.#go(0, "reader")
        break
      case "iter-next":
        this.#go(model.iterationStep(this.#cta, this.#step, 1), "reader")
        break
      case "iter-prev":
        this.#go(model.iterationStep(this.#cta, this.#step, -1), "reader")
        break
      case "cta-next":
        this.#pickCta((this.#cta + 1) % model.ctas.length, "reader")
        break
      case "cta-prev":
        this.#pickCta((this.#cta + model.ctas.length - 1) % model.ctas.length, "reader")
        break
      case "play":
        if (this.#isAutoPlaying) this.#stopPlay()
        else this.#startPlay()
        break
      case "sweep":
        this.#toggleSweep()
        break
    }
  }

  // Steps

  #go(stepIndex: number, origin: Origin, animate = true): void {
    const model = this.#model
    if (!model) return
    const count = model.stepCount(this.#cta)
    const next = clamp(Math.round(stepIndex), 0, count - 1)
    const moved = next !== this.#step
    this.#step = next
    this.setState({ step: next + 1 })
    this.#resetMotion()
    this.#applyState()
    this.#sync()
    if (origin !== "follow") this.#announce()
    if (animate && (moved || origin !== "follow")) this.#animateStep()
  }

  #pickCta(ctaIndex: number, origin: Origin): void {
    const model = this.#model
    if (!model) return
    const key = model.statements(this.#cta)[this.#step]?.key
    this.#cta = clamp(ctaIndex, 0, model.ctas.length - 1)
    const matched = key ? model.findByKey(this.#cta, key) : null
    this.#setRangeMax()
    this.#go(matched ?? this.#step, origin)
  }

  #setZoom(zoom: Zoom, origin: Origin): void {
    const els = this.#els
    this.setState({ zoom })
    if (!els || zoom === this.#zoom) {
      this.#zoom = zoom
      return
    }
    this.#zoom = zoom
    this.#resetMotion()
    const focus = this.#focusPoint()
    const entering = zoom === "lane" ? els.laneGroup : els.tileGroup
    const leaving = zoom === "lane" ? els.tileGroup : els.laneGroup
    entering.classList.remove("viz-off")
    const tokens = this.motion.tokens
    const zoomed = (scale: number) =>
      `translate(${focus.x}px, ${focus.y}px) scale(${scale}) translate(${-focus.x}px, ${-focus.y}px)`
    const inScale = zoom === "lane" ? 0.55 : 1.6
    const outScale = zoom === "lane" ? 1.6 : 0.55
    const options = { duration: tokens.dur3Ms, easing: tokens.easeInOut }
    this.motion.animate(
      entering,
      [
        { opacity: 0, transform: zoomed(inScale) },
        { opacity: 1, transform: "none" },
      ],
      options
    )
    const leave = this.motion.animate(
      leaving,
      [
        { opacity: 1, transform: "none" },
        { opacity: 0, transform: zoomed(outScale) },
      ],
      options
    )
    const hide = () => leaving.classList.toggle("viz-off", this.#zoom !== (leaving === els.laneGroup ? "lane" : "tile"))
    if (leave) leave.addEventListener("finish", hide)
    else hide()
    this.#applyState()
    this.#sync()
    if (origin !== "follow") this.#announce()
    this.#animateStep()
  }

  async #switchChip(chip: string): Promise<void> {
    const name = this.#shell?.dataset.trace
    if (!name) return
    this.#stopPlay()
    try {
      this.#model = await this.#load(name, chip)
    } catch (error) {
      console.error(`Visualizer: ${String(error)}`)
      return
    }
    this.#cta = clamp(this.#cta, 0, this.#model.ctas.length - 1)
    this.#step = clamp(this.#step, 0, this.#model.stepCount(this.#cta) - 1)
    this.#setRangeMax()
    this.drawing.replaceChildren()
    this.build(this.layout)
    this.settle()
    this.#sync()
  }

  #startPlay(): void {
    const model = this.#model
    if (!model) return
    if (this.#step >= model.stepCount(this.#cta) - 1) this.#go(0, "reader")
    this.#isAutoPlaying = true
    this.#autoHoldMs = 0
    this.#playButton(true)
    if (this.reducedMotion) {
      const tick = () => {
        if (!this.#isAutoPlaying) return
        this.#advancePlay()
        if (this.#isAutoPlaying) this.#reducedTimer = window.setTimeout(tick, REDUCED_PLAY_MS)
      }
      this.#reducedTimer = window.setTimeout(tick, REDUCED_PLAY_MS)
      return
    }
    this.play()
  }

  #advancePlay(): void {
    const model = this.#model
    if (!model || this.#step >= model.stepCount(this.#cta) - 1) {
      this.#stopPlay()
      return
    }
    this.#go(this.#step + 1, "reader")
  }

  #stopPlay(): void {
    this.#isAutoPlaying = false
    window.clearTimeout(this.#reducedTimer)
    this.#playButton(false)
  }

  #playButton(isPlaying: boolean): void {
    const button = this.root.querySelector<HTMLElement>('[data-action="play"]')
    if (!button) return
    button.setAttribute("aria-pressed", String(isPlaying))
    button.textContent = isPlaying ? "Pause" : "Play"
  }

  #toggleSweep(): void {
    this.#allCtas = !this.#allCtas
    this.root.querySelector('[data-action="sweep"]')?.setAttribute("aria-pressed", String(this.#allCtas))
    if (this.#allCtas && !this.reducedMotion && (this.#model?.ctas.length ?? 0) > 1) {
      this.#sweepIndex = 0
      this.#sweepHoldMs = 0
      this.#pickCta(0, "reader")
      this.play()
      return
    }
    this.#sweepIndex = -1
    this.#applyState()
  }

  // State to classes. Runs on every step change and after every build; never per frame.

  #applyState(): void {
    const model = this.#model
    const els = this.#els
    const geometry = this.#geometry
    if (!model || !els || !geometry) return
    const state = model.stateAt(this.#cta, this.#step)
    const tile = geometry.tile

    els.ctaNodes.forEach((node, index) => {
      node.classList.toggle("is-on", true)
      node.classList.toggle("is-hot", index === this.#cta)
      node.setAttribute("r", index === this.#cta ? "4.6" : "3.2")
    })
    const picked = tile.grid.nodes[this.#cta]
    const firstNode = tile.grid.nodes[0]
    moveTo(els.ctaGlow, picked.x - firstNode.x, picked.y - firstNode.y)
    els.ctaText.textContent = `${model.ctaName(model.ctas[this.#cta])}, ${this.#cta + 1} of ${model.ctas.length}`

    this.#applyTensors(state)
    const touchedBuffers = new Set(
      [...state.reads, ...state.writes].flatMap(access => (access.buffer === undefined ? [] : [access.buffer]))
    )
    for (const [id, { rect, text }] of els.buffers) {
      const live = state.liveBuffers.includes(id)
      rect.classList.toggle("is-on", live)
      rect.classList.toggle("is-hot", touchedBuffers.has(id))
      rect.classList.toggle("is-idle", !live)
      text?.classList.toggle("is-on", live)
    }

    const statementOps = new Set(state.statement.steps.map(step => step.op))
    model.barriers.forEach((barrier, index) => {
      els.barrierTicks[index]?.group.classList.toggle("viz-current", statementOps.has(barrier.afterOp))
    })
    const lastOp = state.statement.steps[state.statement.steps.length - 1].op
    const program = tile.program
    const opSpan = Math.max(1, model.opCount - 1)
    moveTo(els.programCursor, program.x0 + (model.orderOf(lastOp) / opSpan) * (program.x1 - program.x0), 0)

    this.#applyThreads(state)
    this.#applyLaneLevel(state)
    this.setReadout(model.readout(this.#cta, this.#step))
  }

  #applyTensors(state: StepState): void {
    const model = this.#model
    const els = this.#els
    const tile = this.#geometry?.tile
    if (!model || !els || !tile) return
    const across = this.#allCtas ? model.accessesAcross(this.#step) : []
    tile.tensors.forEach((place, index) => {
      const tensor = els.tensors[index]
      const written = state.written[place.name] ?? []
      tensor.written.forEach((rect, slot) => {
        const box = written[slot]
        rect.classList.toggle("is-hidden", !box)
        if (box) setRect(rect, boxRect(place, box))
      })
      const current = [
        ...state.reads.filter(access => access.tensor === place.name).map(access => ({ access, isWrite: false })),
        ...state.writes.filter(access => access.tensor === place.name).map(access => ({ access, isWrite: true })),
      ]
      tensor.current.forEach((rect, slot) => {
        const entry = current[slot]
        rect.classList.toggle("is-hidden", !entry)
        if (!entry) return
        rect.classList.toggle("is-outline", !entry.isWrite)
        setRect(rect, boxRect(place, entry.access.box))
      })
      tensor.others.forEach((rect, ctaIndex) => {
        const other = across[ctaIndex]
        const access =
          other && ctaIndex !== this.#cta
            ? [...other.reads, ...other.writes].find(candidate => candidate.tensor === place.name)
            : undefined
        rect.classList.toggle("is-hidden", !access)
        if (access) setRect(rect, boxRect(place, access.box))
      })
      const masked = current.reduce((sum, { access }) => sum + (access.maskedElems ?? 0), 0)
      tensor.masked.textContent = masked > 0 ? `${masked} masked` : ""
    })
  }

  #applyThreads(state: StepState): void {
    const model = this.#model
    const els = this.#els
    const tile = this.#geometry?.tile
    if (!model || !els || !tile) return
    const lanes = model.trace.launch.lanes
    const laneEvent = state.lanes[0]?.event
    const active = state.mapped
      ? state.mapped.threads.activeThreads
      : laneEvent
        ? Math.min(lanes, laneEvent.runs?.length ?? lanes)
        : 0
    els.threadDots.forEach((dot, thread) => dot.classList.toggle("is-on", thread < active))
    if (state.mapped) {
      const facts = state.mapped.threads
      const who =
        facts.activeThreads < facts.threads
          ? `${facts.activeThreads} of ${facts.threads} threads`
          : `${facts.threads} threads`
      const passes = facts.passes > 1 ? `, ${facts.passes} passes` : ""
      els.threadText.textContent = `${who}, ${facts.perThread} ${facts.elem} each${passes}`
    } else if (laneEvent) {
      els.threadText.textContent = `warp 0, ${laneEvent.kind.replace(/_/g, " ")}`
    } else {
      els.threadText.textContent = "no thread map recorded for this op"
    }
    const lines = this.#linesOf(state)
    this.#applySlots(els.tileSlots, tile.threads.slots, lines?.facts ?? null)
    els.tileSlotsText.textContent = lines ? linesSentence(lines.facts, lines.tensor) : ""
  }

  #applyLaneLevel(state: StepState): void {
    const model = this.#model
    const els = this.#els
    const lane = this.#geometry?.lane
    if (!model || !els || !lane) return
    const lines = this.#linesOf(state)
    const slots = this.#applySlots(els.laneSlots, lane.slots, lines?.facts ?? null, LANE_SLOT_UNITS)
    els.laneSlotsText.textContent = lines
      ? linesSentence(lines.facts, lines.tensor)
      : "no global memory access at this step"
    const laneSlots = lines?.facts.laneLines ?? []
    els.laneWires.forEach((wire, laneIndex) => {
      const slot = slots[laneSlots[laneIndex]?.[0] ?? -1]
      wire.classList.toggle("is-hidden", !slot)
      if (!slot) return
      const node = lane.lanes[laneIndex]
      const to = laneTarget(slot, lines?.facts.laneBytes[laneIndex])
      wire.setAttribute("d", `M${round(node.x)} ${round(node.y + 3)}L${round(to.x)} ${round(to.y)}`)
    })
    const collective = state.lanes.find(entry => entry.event.stages)
    const anyLanes = state.lanes.length > 0 || state.mapped !== null
    els.laneNodes.forEach(node => node.classList.toggle("is-on", anyLanes))
    for (const arcs of els.arcs) {
      const isCurrent = collective?.opId === arcs.opId
      for (const stage of arcs.stages) {
        for (const pair of stage) {
          pair.path.classList.toggle("is-hidden", !isCurrent)
          pair.path.classList.remove("is-on")
          pair.path.classList.toggle("is-idle", isCurrent)
        }
      }
    }
    els.stageText.textContent = collective ? stagesSentence(collective.event) : "no shuffle at this step"
    els.idleText.textContent =
      state.lanes.length === 0 && !state.mapped ? "this op has no per-lane record in the trace" : ""
    const current = new Set(state.values.map(value => value.name))
    model.values.slice(0, els.kindNames.length).forEach((value, index) => {
      els.kindNames[index].classList.toggle("is-hot", current.has(value.name))
      els.kindLabels[index].classList.toggle("is-hot", current.has(value.name))
      els.kindLabels[index].textContent = kindName(value.kind)
    })
  }

  #settleLanes(): void {
    for (const packet of this.#els?.lanePackets ?? []) packet.hide()
    this.#lastStage = -1
  }

  #linesOf(state: StepState): { facts: LineFacts; tensor: string } | null {
    const model = this.#model
    if (!model) return null
    for (const { event } of state.lanes) {
      const facts = model.laneRunLines(event)
      const tensor = event.runs?.[0]?.access.tensor
      if (facts && tensor) return { facts, tensor }
    }
    const mapped = state.mapped
    if (mapped?.lines && mapped.access.tensor) return { facts: mapped.lines, tensor: mapped.access.tensor }
    return null
  }

  #applySlots(pool: SlotElements[], region: Rect, facts: LineFacts | null, maxUnits = 26): Rect[] {
    const count = Math.min(pool.length, facts?.touched.length ?? 0)
    const rects = slotRects(region, count, maxUnits)
    pool.forEach((slot, index) => {
      const rect = rects[index]
      const line = facts?.touched[index]
      slot.group.classList.toggle("is-hidden", !rect || !line)
      if (!rect || !line) return
      setRect(slot.outline, rect)
      setRect(slot.used, { ...rect, width: Math.max(1, (rect.width * line.usedBytes) / LINE_BYTES) })
      const wasted = line.usedBytes < LINE_BYTES
      slot.outline.classList.toggle("is-warn", wasted)
      slot.group.classList.toggle("sp-cost", wasted)
      slot.group.classList.toggle("sp-global", !wasted)
    })
    return rects
  }

  // Motion

  #resetMotion(): void {
    this.#timeline = null
    const els = this.#els
    if (!els) return
    for (const packet of [...els.packets, ...els.lanePackets]) packet.hide()
    for (const route of els.routes) {
      route.path.classList.add("is-hidden")
      route.sweep.classList.remove("is-on")
    }
  }

  #animateStep(): void {
    if (this.reducedMotion || !this.#model || !this.#els) return
    const state = this.#model.stateAt(this.#cta, this.#step)
    const steps = this.#zoom === "tile" ? this.#tileSteps(state) : this.#laneSteps(state)
    if (steps.length === 0) return
    this.#timeline = new Timeline(steps)
    this.play()
  }

  #tileSteps(state: StepState): TimelineStep[] {
    const els = this.#els
    if (!els) return []
    const routes = this.#routes(state).slice(0, els.routes.length)
    if (routes.length === 0) return []
    const perRoute = Math.floor(els.packets.length / routes.length)
    routes.forEach((route, index) => {
      const wire = els.routes[index]
      const pathData = route.line.toPathData()
      wire.path.setAttribute("d", pathData)
      wire.sweep.setAttribute("d", pathData)
      wire.path.setAttribute("class", `chart-wire is-on sp-${route.space} is-hidden`)
    })
    const packetsSpan = (perRoute - 1) * PACKET_STAGGER_MS
    const totalMs = Math.max(...routes.map(route => route.delayMs + route.travelMs + packetsSpan)) + 160
    return [
      {
        durationMs: totalMs,
        apply: progress => {
          const elapsedMs = progress * totalMs
          routes.forEach((route, index) => {
            const localMs = elapsedMs - route.delayMs
            const isActive = progress < 1 && localMs >= 0 && localMs <= route.travelMs + packetsSpan
            els.routes[index].path.classList.toggle("is-hidden", !isActive)
            els.routes[index].sweep.classList.toggle("is-on", isActive)
            for (let copy = 0; copy < perRoute; copy += 1) {
              const packet = els.packets[index * perRoute + copy]
              const fraction = (localMs - copy * PACKET_STAGGER_MS) / route.travelMs
              if (progress >= 1) packet.hide()
              else packet.placeAtFraction(route.line, fraction)
            }
          })
        },
      },
    ]
  }

  // Where each op's data comes from and goes: tensor boxes, buffer slots, the threads band.
  #routes(state: StepState): Route[] {
    const model = this.#model
    const geometry = this.#geometry
    if (!model || !geometry) return []
    const threads = geometry.tile.threads.rect
    const threadsPort = { x: threads.x + threads.width / 2, y: threads.y }
    const port = (access: Access): Point | null => {
      if (access.tensor !== undefined) {
        const place = geometry.tile.tensors.find(candidate => candidate.name === access.tensor)
        if (!place) return null
        const rect = boxRect(place, access.box)
        return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }
      }
      const placed = geometry.tile.shared.buffers.find(candidate => candidate.id === access.buffer)
      return placed ? { x: placed.rect.x + placed.rect.width / 2, y: placed.rect.y + placed.rect.height / 2 } : null
    }
    const spaceOf = (access: Access) => (access.tensor !== undefined ? "global" : "shared")
    const routes: Route[] = []
    state.statement.steps.forEach((step, opIndex) => {
      const reads = (step.reads ?? []).flatMap(access => {
        const at = port(access)
        return at ? [{ at, space: spaceOf(access) as Route["space"] }] : []
      })
      const writes = (step.writes ?? []).flatMap(access => {
        const at = port(access)
        return at ? [at] : []
      })
      const sources = reads.length > 0 ? reads : [{ at: threadsPort, space: "reg" as const }]
      const targets = writes.length > 0 ? writes : [threadsPort]
      for (const source of sources) {
        for (const target of targets.slice(0, 2)) {
          if (Math.hypot(source.at.x - target.x, source.at.y - target.y) < 4) continue
          const line = elbow(roundPoint(source.at), roundPoint(target), "v").simplified()
          const travelMs = clamp(line.lengthUnits / SPEED_UNITS_PER_MS, 320, 1100)
          routes.push({ line, delayMs: opIndex * OP_STAGGER_MS, travelMs, space: source.space })
        }
      }
    })
    return routes
  }

  #laneSteps(state: StepState): TimelineStep[] {
    const els = this.#els
    const lane = this.#geometry?.lane
    if (!els || !lane) return []
    const steps: TimelineStep[] = []
    const lines = this.#linesOf(state)
    if (lines) {
      const slots = slotRects(lane.slots, Math.min(els.laneSlots.length, lines.facts.touched.length), LANE_SLOT_UNITS)
      const isStore =
        state.lanes.some(entry => entry.event.kind === "lane_store") ||
        (state.mapped !== null && state.writes.includes(state.mapped.access))
      const paths = lines.facts.laneLines.map((slotIndexes, laneIndex) => {
        const slot = slots[slotIndexes[0] ?? -1]
        if (!slot) return null
        const from = laneTarget(slot, lines.facts.laneBytes[laneIndex])
        const to = lane.lanes[laneIndex]
        return isStore ? new Polyline([to, from]) : new Polyline([from, to])
      })
      const totalMs = LANE_TRAVEL_MS + paths.length * LANE_STAGGER_MS + 120
      steps.push({
        durationMs: totalMs,
        apply: progress => {
          paths.forEach((line, laneIndex) => {
            const packet = els.lanePackets[laneIndex]
            if (!line || progress >= 1) {
              packet.hide()
              return
            }
            packet.placeAtFraction(line, (progress * totalMs - laneIndex * LANE_STAGGER_MS) / LANE_TRAVEL_MS)
          })
        },
      })
    }
    const collective = state.lanes.find(entry => entry.event.stages)
    const arcs = collective ? els.arcs.find(candidate => candidate.opId === collective.opId) : undefined
    if (collective && arcs) {
      const stageCount = arcs.stages.length
      steps.push({
        durationMs: STAGE_MS * stageCount,
        apply: progress => this.#applyStages(arcs.stages, collective.event, progress),
      })
    }
    return steps
  }

  #applyStages(stages: VizElements["arcs"][number]["stages"], event: LaneEvent, progress: number): void {
    const els = this.#els
    if (!els) return
    const stageFloat = progress * stages.length
    const current = Math.min(stages.length - 1, Math.floor(stageFloat))
    const local = progress >= 1 ? 1 : stageFloat - current
    stages.forEach((stage, index) => {
      for (const pair of stage) {
        const isCurrent = index === current && progress < 1
        pair.path.classList.toggle("is-hidden", index > current && progress < 1)
        pair.path.classList.toggle("is-on", isCurrent)
        pair.path.classList.toggle("is-idle", !isCurrent)
        pair.path.setAttribute("stroke-dashoffset", isCurrent ? String(round(1 - Math.min(1, local / 0.3))) : "0")
        pair.path.setAttribute("stroke-dasharray", isCurrent ? "1 1" : "none")
      }
    })
    const pairs = stages[current]
    els.lanePackets.forEach((packet, index) => {
      const pair = pairs[Math.floor(index / 2)]
      if (!pair || progress >= 1) {
        packet.hide()
        return
      }
      packet.placeAtFraction(index % 2 === 0 ? pair.line : pair.reverse, (local - 0.22) / 0.6)
    })
    if (current !== this.#lastStage || progress >= 1) {
      this.#lastStage = progress >= 1 ? -1 : current
      const distance = event.stages?.[current]?.[0] ?? 0
      els.stageText.textContent =
        progress >= 1
          ? stagesSentence(event)
          : `stage ${current + 1} of ${stages.length}: ${event.mode ?? "xor"} ${distance}`
    }
  }

  #focusPoint(): Point {
    const geometry = this.#geometry
    const model = this.#model
    if (!geometry || !model) return { x: 0, y: 0 }
    const state = model.stateAt(this.#cta, this.#step)
    const access = state.mapped?.access ?? [...state.reads, ...state.writes].find(candidate => candidate.tensor)
    const place = access?.tensor ? geometry.tile.tensors.find(candidate => candidate.name === access.tensor) : undefined
    if (access && place) {
      const rect = boxRect(place, access.box)
      return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }
    }
    const threads = geometry.tile.threads.rect
    return { x: threads.x + threads.width / 2, y: threads.y }
  }

  // The page around the drawing: code marker, labels, URL, lockstep, keys.

  #sync(): void {
    const model = this.#model
    if (!model) return
    const statement = model.statements(this.#cta)[this.#step]
    if (!statement) return
    const iter = iterLabel(statement.iter)
    const position = this.root.querySelector("[data-viz-pos]")
    if (position) {
      position.textContent = `step ${this.#step + 1} of ${model.stepCount(this.#cta)}${iter ? `, ${iter}` : ""}`
    }
    const ctaLabel = this.root.querySelector("[data-viz-cta]")
    if (ctaLabel) ctaLabel.textContent = model.ctaName(model.ctas[this.#cta])
    this.#syncCode(statement.span.line, statement.span.endLine)
    if (this.#shell?.hasAttribute("data-url-state")) {
      const cta = model.ctas[this.#cta]
      const query = `?step=${this.#step + 1}&cta=${cta[0]},${cta[1]}&zoom=${this.#zoom}`
      if (location.search !== query) history.replaceState(history.state, "", query)
    }
  }

  #announce(): void {
    const model = this.#model
    if (!model) return
    const statement = model.statements(this.#cta)[this.#step]
    const count = model.stepCount(this.#cta)
    const detail: StepDetail = {
      key: statement?.key ?? "",
      fraction: count > 1 ? this.#step / (count - 1) : 0,
      zoom: this.#zoom,
      ctaIndex: this.#cta,
    }
    this.root.dispatchEvent(new CustomEvent("viz:step", { bubbles: true, detail }))
  }

  #onFollow(event: CustomEvent<StepDetail>): void {
    const model = this.#model
    if (!model) return
    const { key, fraction, zoom, ctaIndex } = event.detail
    this.#stopPlay()
    if (ctaIndex < model.ctas.length && ctaIndex !== this.#cta) {
      this.#cta = ctaIndex
      this.#setRangeMax()
    }
    const count = model.stepCount(this.#cta)
    const matched = model.findByKey(this.#cta, key)
    if (zoom !== this.#zoom) this.#setZoom(zoom, "follow")
    this.#go(matched ?? Math.round(fraction * (count - 1)), "follow")
  }

  #attachCode(): void {
    const shell = this.#shell
    if (!shell) return
    this.#codeBox = shell.querySelector<HTMLElement>("[data-viz-code]")
    this.#codeLines = [...(this.#codeBox?.querySelectorAll<HTMLElement>(".ec-line") ?? [])]
    this.#marker = shell.querySelector<HTMLElement>(".viz-marker")
    try {
      this.#lineMap = JSON.parse(shell.dataset.lineMap ?? "[]")
    } catch {
      this.#lineMap = []
    }
    if (this.#codeBox) new ResizeObserver(() => this.#sync()).observe(this.#codeBox)
  }

  #syncCode(line: number, endLine: number): void {
    const box = this.#codeBox
    const marker = this.#marker
    if (!box || !marker || this.#codeLines.length === 0) return
    const shown = (original: number) => {
      const index = this.#lineMap.indexOf(original)
      return index >= 0 ? index : original - 1
    }
    const first = this.#codeLines[shown(line)]
    const last = this.#codeLines[shown(Math.max(line, endLine))] ?? first
    if (!first || !last) return
    const boxRectPx = box.getBoundingClientRect()
    const topPx = first.getBoundingClientRect().top - boxRectPx.top + box.scrollTop
    const heightPx = last.getBoundingClientRect().bottom - first.getBoundingClientRect().top
    marker.style.transform = `translateY(${round(topPx)}px)`
    marker.style.height = `${round(heightPx)}px`
    marker.classList.add("is-on")
    if (box.scrollHeight > box.clientHeight + 1) {
      const target = topPx - (box.clientHeight - heightPx) / 2
      box.scrollTo({ top: Math.max(0, target), behavior: this.reducedMotion ? "auto" : "smooth" })
    }
  }

  #attachKeys(): void {
    const shell = this.#shell
    if (!shell) return
    const target: EventTarget = shell.hasAttribute("data-global-keys") ? document : shell
    target.addEventListener("keydown", event => this.#onKey(event as KeyboardEvent))
  }

  #onKey(event: KeyboardEvent): void {
    if (event.altKey || event.ctrlKey || event.metaKey) return
    const element = event.target as Element | null
    if (element?.closest("input, textarea, select, [contenteditable]")) return
    const other = element?.closest("[data-viz]")
    if (other && other !== this.#shell) return
    const actions: Record<string, string> = {
      ArrowRight: "next",
      ArrowLeft: "prev",
      " ": "play",
      "]": "iter-next",
      "[": "iter-prev",
    }
    const action = actions[event.key]
    if (!action) return
    const button = this.root.querySelector<HTMLButtonElement>(`[data-action="${action}"]`)
    if (!button) return
    event.preventDefault()
    button.click()
  }

  #readUrlState(): void {
    const model = this.#model
    if (!model) return
    const initial = Number(this.#shell?.dataset.step)
    if (initial > 0) this.#step = clamp(initial - 1, 0, model.stepCount(this.#cta) - 1)
    if (!this.#shell?.hasAttribute("data-url-state")) return
    const params = new URLSearchParams(location.search)
    const cta = params.get("cta")
    if (cta) {
      const [x, y] = cta.split(",").map(Number)
      const index = model.ctaIndex([x || 0, y || 0, 0] as Cta)
      if (index >= 0) this.#cta = index
    }
    const step = Number(params.get("step"))
    if (step > 0) this.#step = clamp(step - 1, 0, model.stepCount(this.#cta) - 1)
    if (params.get("zoom") === "lane") this.#zoom = "lane"
  }

  #pressZoomButton(): void {
    if (this.#zoom !== "lane") {
      const pressed = this.root.querySelector<HTMLElement>('[data-group="zoom"][aria-pressed="true"]')
      if (pressed?.dataset.value === "lane") this.#zoom = "lane"
    }
    for (const button of this.root.querySelectorAll<HTMLElement>('[data-group="zoom"]')) {
      button.setAttribute("aria-pressed", String(button.dataset.value === this.#zoom))
    }
  }

  #setRangeMax(): void {
    const model = this.#model
    const range = this.root.querySelector<HTMLInputElement>('input[data-param="step"]')
    if (!model || !range) return
    range.max = String(model.stepCount(this.#cta))
    range.value = String(this.#step + 1)
  }
}

function moveTo(element: Element, dx: number, dy: number): void {
  ;(element as SVGElement).style.transform = `translate(${round(dx)}px, ${round(dy)}px)`
}

function round(value: number): number {
  return Math.round(value * 100) / 100
}

function roundPoint(point: Point): Point {
  return { x: round(point.x), y: round(point.y) }
}

// Where a lane's run lands in its line's slot: the middle of the bytes it uses.
function laneTarget(slot: Rect, bytes: [number, number] | undefined): Point {
  const middle = bytes ? (bytes[0] + bytes[1]) / 2 : LINE_BYTES / 2
  return { x: slot.x + (slot.width * middle) / LINE_BYTES, y: slot.y }
}

function linesSentence(facts: LineFacts, tensor: string): string {
  const lines = `${facts.lines} line${facts.lines === 1 ? "" : "s"}`
  const extra = facts.lines > facts.idealLines ? `, ${facts.idealLines} would do` : ""
  return `warp 0, ${tensor}: ${lines} for ${facts.requestedBytes} bytes${extra}`
}

function stagesSentence(event: LaneEvent): string {
  const distances = (event.stages ?? []).map(stage => stage[0])
  return `${distances.length} ${event.mode ?? "xor"} stages: ${distances.join(", ")}`
}
