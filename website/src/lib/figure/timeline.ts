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
