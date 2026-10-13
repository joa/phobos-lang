// The scene's journey: dwell on a station, move to the next, wrap from the last back to the first. A
// jump to a station moves there at once and rests on it longer, so a reader's click is not overrun by
// the next advance. Driven by dtMs, so it pauses with the frame loop. Pure.

export interface ClockOptions {
  count: number
  dwellMs: (station: number) => number
  moveMs?: number
  jumpMoveMs?: number
  /** Share of a move after which the next station comes on. */
  onAt?: number
  /** Least rest on a station a reader jumped to. */
  jumpRestMs?: number
}

export class PipelineClock {
  readonly count: number
  #dwellMs: (station: number) => number
  #moveMs: number
  #jumpMoveMs: number
  #onAt: number
  #jumpRestMs: number

  #from = 0
  #station = 0
  #moving = false
  #moveElapsedMs = 0
  #currentMoveMs = 0
  #stationMs = 0
  #restMs = 0

  constructor(options: ClockOptions) {
    this.count = options.count
    this.#dwellMs = options.dwellMs
    this.#moveMs = options.moveMs ?? 1700
    this.#jumpMoveMs = options.jumpMoveMs ?? 1100
    this.#onAt = options.onAt ?? 0.55
    this.#jumpRestMs = options.jumpRestMs ?? 12000
    this.#restMs = this.#dwellMs(0)
  }

  /** The station on screen, or the one being moved to. */
  get station(): number {
    return this.#station
  }

  /** The station a move leaves; equal to station at rest. */
  get from(): number {
    return this.#moving ? this.#from : this.#station
  }

  get moving(): boolean {
    return this.#moving
  }

  /** 0 to 1 through the current move; 1 at rest. */
  get moveProgress(): number {
    return this.#moving ? Math.min(1, this.#moveElapsedMs / this.#currentMoveMs) : 1
  }

  /** Time since the station came on; negative early in a move, before it does. */
  get stationMs(): number {
    return this.#stationMs
  }

  /** 0 to 1 through the rest on the current station, for the rail's progress. */
  get restProgress(): number {
    if (this.#moving) return 0
    return Math.min(1, Math.max(0, this.#stationMs / this.#restMs))
  }

  /** Starts the station over, as when the scene first comes into view. */
  restart(station = 0): void {
    this.#from = station
    this.#station = station
    this.#moving = false
    this.#stationMs = 0
    this.#restMs = this.#dwellMs(station)
  }

  advance(dtMs: number): void {
    const dt = Math.max(0, dtMs)
    this.#stationMs += dt
    if (this.#moving) {
      this.#moveElapsedMs += dt
      if (this.#moveElapsedMs >= this.#currentMoveMs) this.#moving = false
      return
    }
    if (this.#stationMs >= this.#restMs) this.#move((this.#station + 1) % this.count, this.#moveMs, this.#dwellMs)
  }

  jump(station: number): void {
    const target = ((station % this.count) + this.count) % this.count
    this.#move(target, this.#jumpMoveMs, index => Math.max(this.#dwellMs(index), this.#jumpRestMs))
  }

  #move(target: number, moveMs: number, rest: (station: number) => number): void {
    this.#from = this.#station
    this.#station = target
    this.#moving = true
    this.#moveElapsedMs = 0
    this.#currentMoveMs = moveMs
    this.#stationMs = -this.#onAt * moveMs
    this.#restMs = rest(target)
  }
}
