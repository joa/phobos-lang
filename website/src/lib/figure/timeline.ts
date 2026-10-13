// Steps run back to back. apply(progress) must be a pure function of progress that sets everything
// the step owns, so apply(1) is the step's end state and seekEnd() can replay all of them.

export interface TimelineStep {
  durationMs: number
  apply(progress: number): void
}

export class Timeline {
  #steps: TimelineStep[]
  #holdMs: number
  #stepIndex = 0
  #stepElapsedMs = 0
  #heldMs = 0

  constructor(steps: TimelineStep[], { holdMs = 0 }: { holdMs?: number } = {}) {
    this.#steps = steps
    this.#holdMs = holdMs
  }

  get totalMs(): number {
    return this.#steps.reduce((sumMs, step) => sumMs + Math.max(0, step.durationMs), this.#holdMs)
  }

  get isFinished(): boolean {
    return this.#stepIndex >= this.#steps.length && this.#heldMs >= this.#holdMs
  }

  reset(): void {
    this.#stepIndex = 0
    this.#stepElapsedMs = 0
    this.#heldMs = 0
  }

  // Returns true while a step runs or the hold lasts, false once finished.
  advance(dtMs: number): boolean {
    let remainingMs = Math.max(0, dtMs)
    while (this.#stepIndex < this.#steps.length) {
      const step = this.#steps[this.#stepIndex]
      const leftInStepMs = step.durationMs - this.#stepElapsedMs
      if (remainingMs < leftInStepMs) {
        this.#stepElapsedMs += remainingMs
        step.apply(this.#stepElapsedMs / step.durationMs)
        return true
      }
      remainingMs -= Math.max(0, leftInStepMs)
      step.apply(1)
      this.#stepIndex += 1
      this.#stepElapsedMs = 0
    }
    this.#heldMs = Math.min(this.#holdMs, this.#heldMs + remainingMs)
    return this.#heldMs < this.#holdMs
  }

  seekEnd(): void {
    for (const step of this.#steps) step.apply(1)
    this.#stepIndex = this.#steps.length
    this.#stepElapsedMs = 0
    this.#heldMs = this.#holdMs
  }
}

// How long a figure rests on its end state before it plays again, and how long the end state takes
// to cross-fade into the start of the next run.
export const LOOP_HOLD_MS = 1500
export const LOOP_FADE_MS = 900

export type LoopTick = "idle" | "run" | "hold" | "restart"

// Decides when a figure that has finished playing plays again. The figure reports every frame
// whether it animated; once it stops, the clock holds for holdMs and then says "restart". A figure
// that never animated since its last restart stays idle, so a run that turned out to have nothing to
// play does not restart forever. Pure, driven by dtMs, so it pauses with the frame loop.
export class LoopClock {
  #holdMs: number
  #heldMs = 0
  #hasRun = false

  constructor({ holdMs = LOOP_HOLD_MS }: { holdMs?: number } = {}) {
    this.#holdMs = holdMs
  }

  get heldMs(): number {
    return this.#heldMs
  }

  reset(): void {
    this.#heldMs = 0
    this.#hasRun = false
  }

  tick(isRunning: boolean, dtMs: number): LoopTick {
    if (isRunning) {
      this.#hasRun = true
      this.#heldMs = 0
      return "run"
    }
    if (!this.#hasRun) return "idle"
    this.#heldMs += Math.max(0, dtMs)
    if (this.#heldMs < this.#holdMs) return "hold"
    this.reset()
    return "restart"
  }
}
