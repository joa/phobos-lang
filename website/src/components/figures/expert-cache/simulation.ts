// The router and the expert cache of one mixture-of-experts layer, without any DOM. Simulated: expert
// popularity is Zipf-like and each token keeps some of the previous token's picks, which is enough to
// show why a cache much smaller than the expert count hits most of the time. The real routing is not
// modeled and no rate from it is claimed.
//
// What follows phobos-gguf: 256 experts and 8 picks a token (MOE_USED), least recently used eviction
// within a block, and the two paths. A decode step computes its misses on the CPU from host memory and
// queues one refill, its most heavily weighted miss, to copy into the cache for later tokens. A prompt
// copies its misses over PCIe into cache slots.

import { mulberry32 } from "../../../lib/figure/math"

export const EXPERT_COUNT = 256
export const PICKS_PER_TOKEN = 8
// Drawn, never printed: the cache's real size depends on the card's free memory.
export const SLOT_COUNT = 40
export const WARMUP_TOKENS = 300
export const SIMULATION_SEED = 5
export const KEEP_PROBABILITY = 0.4
export const ZIPF_EXPONENT = 1.25

export type MissPath = "decode" | "prompt"

export interface Pick {
  expert: number
  slot: number | null
  isHit: boolean
}

export interface Copy {
  expert: number
  slot: number
  // A decode step's refill: copied for later tokens while the CPU computes this token's misses.
  isRefill: boolean
}

export interface TokenResult {
  number: number
  path: MissPath
  picks: Pick[]
  copies: Copy[]
  hitCount: number
  hitRate: number
}

// Vose's alias method: O(n) to build, then a draw is one uniform column and one biased coin.
export class AliasSampler {
  #probabilities: Float64Array
  #aliases: Uint16Array

  constructor(weights: number[]) {
    const count = weights.length
    const total = weights.reduce((sum, weight) => sum + weight, 0)
    const scaled = weights.map(weight => (weight * count) / total)
    this.#probabilities = new Float64Array(count).fill(1)
    this.#aliases = new Uint16Array(count)
    const underfull: number[] = []
    const overfull: number[] = []
    scaled.forEach((value, index) => (value < 1 ? underfull : overfull).push(index))
    while (underfull.length > 0 && overfull.length > 0) {
      const small = underfull.pop()!
      const large = overfull.pop()!
      this.#probabilities[small] = scaled[small]
      this.#aliases[small] = large
      scaled[large] -= 1 - scaled[small]
      if (scaled[large] < 1) underfull.push(large)
      else overfull.push(large)
    }
  }

  sample(random: () => number): number {
    const column = Math.floor(random() * this.#probabilities.length)
    return random() < this.#probabilities[column] ? column : this.#aliases[column]
  }
}

// Weight 1 / rank^s, with the ranks dealt by a seeded Fisher-Yates shuffle so the popular experts are
// scattered over the grid.
function zipfPopularity(random: () => number, exponent: number): number[] {
  const ranks = Array.from({ length: EXPERT_COUNT }, (_, index) => index)
  for (let index = EXPERT_COUNT - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1))
    ;[ranks[index], ranks[other]] = [ranks[other], ranks[index]]
  }
  return ranks.map(rank => 1 / (rank + 1) ** exponent)
}

export interface SimulationOptions {
  path?: MissPath
  seed?: number
  keepProbability?: number
  zipfExponent?: number
  warmupTokens?: number
  slotCount?: number
}

export class RouterSimulation {
  #random: () => number
  #popularity: number[]
  #sampler: AliasSampler
  #keepProbability: number
  #slotCount: number
  // Expert to slot. A Map iterates in insertion order, so re-inserting on use keeps the least
  // recently used expert first.
  #slotByExpert = new Map<number, number>()
  #previousPicks: number[] = []
  #path: MissPath
  #tokenCount = 0
  #hitCount = 0

  constructor({
    path = "decode",
    seed = SIMULATION_SEED,
    keepProbability = KEEP_PROBABILITY,
    zipfExponent = ZIPF_EXPONENT,
    warmupTokens = WARMUP_TOKENS,
    slotCount = SLOT_COUNT,
  }: SimulationOptions = {}) {
    this.#random = mulberry32(seed)
    this.#keepProbability = keepProbability
    this.#slotCount = slotCount
    this.#popularity = zipfPopularity(this.#random, zipfExponent)
    this.#sampler = new AliasSampler(this.#popularity)
    this.#path = path
    for (let token = 0; token < warmupTokens; token += 1) this.step()
    this.#resetStats()
  }

  get path(): MissPath {
    return this.#path
  }

  get hitRate(): number {
    return this.#tokenCount === 0 ? 0 : this.#hitCount / (this.#tokenCount * PICKS_PER_TOKEN)
  }

  // Expert in each slot, null where a slot is empty.
  get slots(): (number | null)[] {
    const slots: (number | null)[] = Array.from({ length: this.#slotCount }, () => null)
    for (const [expert, slot] of this.#slotByExpert) slots[slot] = expert
    return slots
  }

  // The cache survives a change of path; only the running statistics start over.
  setPath(path: MissPath): void {
    this.#path = path
    this.#resetStats()
  }

  step(): TokenResult {
    const picks = this.#route().map(expert => {
      const slot = this.#slotByExpert.get(expert) ?? null
      return { expert, slot, isHit: slot !== null }
    })
    const hits = picks.filter(pick => pick.isHit)
    // Touch every hit first, so a token's misses never evict its own hits.
    for (const { expert, slot } of hits) {
      this.#slotByExpert.delete(expert)
      this.#slotByExpert.set(expert, slot!)
    }
    const isRefill = this.#path === "decode"
    const copies = this.#chooseCopies(picks.filter(pick => !pick.isHit)).map(({ expert }) => ({
      expert,
      slot: this.#insert(expert),
      isRefill,
    }))
    this.#tokenCount += 1
    this.#hitCount += hits.length
    return Object.freeze({
      number: this.#tokenCount,
      path: this.#path,
      picks,
      copies,
      hitCount: hits.length,
      hitRate: this.hitRate,
    })
  }

  #resetStats(): void {
    this.#tokenCount = 0
    this.#hitCount = 0
  }

  // Each of the previous token's picks comes back with keepProbability; the rest are drawn by
  // popularity, rejecting duplicates.
  #route(): number[] {
    const picks = this.#previousPicks.filter(() => this.#random() < this.#keepProbability)
    while (picks.length < PICKS_PER_TOKEN) {
      const expert = this.#sampler.sample(this.#random)
      if (!picks.includes(expert)) picks.push(expert)
    }
    this.#previousPicks = picks
    return picks
  }

  // A prompt copies every miss. Decode copies one, standing in for the most heavily weighted miss.
  #chooseCopies(misses: Pick[]): Pick[] {
    if (this.#path === "prompt" || misses.length === 0) return misses
    const popularity = this.#popularity
    return [misses.reduce((best, miss) => (popularity[miss.expert] > popularity[best.expert] ? miss : best))]
  }

  #insert(expert: number): number {
    let slot = this.#slotByExpert.size
    if (slot >= this.#slotCount) {
      const [leastRecent, freed] = this.#slotByExpert.entries().next().value as [number, number]
      this.#slotByExpert.delete(leastRecent)
      slot = freed
    }
    this.#slotByExpert.set(expert, slot)
    return slot
  }
}
