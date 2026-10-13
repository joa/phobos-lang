// The arithmetic behind the bandwidth-bound figure, from the roofline model (Williams, Waterman and
// Patterson, CACM 2009): below the card's balance point of operations per byte the memory bus sets the
// pace, above it the math units do. Pure.

// A Q8_0 block, phobos-gguf/src/quant/q8_0.rs: 32 int8 weights and an f16 scale in 34 bytes.
export const Q8_0_BLOCK_WEIGHTS = 32
export const Q8_0_BLOCK_BYTES = 34
// One multiply-add per weight per token.
export const OPS_PER_WEIGHT = 2
// The longer prompt scripts/bench.py measures.
export const PROMPT_TOKENS = 512

// The card the results were measured on, from NVIDIA's specifications for the RTX 2080 SUPER: 48 SMs of
// 64 fp32 lanes at a 1815 MHz boost clock, and 15.5 Gbps GDDR6 on a 256-bit bus.
export const CARD = {
  name: "NVIDIA GeForce RTX 2080 SUPER",
  smCount: 48,
  fp32LanesPerSm: 64,
  boostClockMhz: 1815,
  memoryGbps: 15.5,
  busWidthBits: 256,
} as const

export const PEAK_FLOPS = CARD.smCount * CARD.fp32LanesPerSm * OPS_PER_WEIGHT * CARD.boostClockMhz * 1e6
export const BANDWIDTH_BYTES_PER_S = (CARD.memoryGbps * 1e9 * CARD.busWidthBits) / 8
export const BALANCE_OPS_PER_BYTE = PEAK_FLOPS / BANDWIDTH_BYTES_PER_S

export const DECODE_OPS_PER_BYTE = (OPS_PER_WEIGHT * Q8_0_BLOCK_WEIGHTS) / Q8_0_BLOCK_BYTES
export const PROMPT_OPS_PER_BYTE = DECODE_OPS_PER_BYTE * PROMPT_TOKENS

// The share of the time each side can be busy: the slower side runs flat out, the other waits.
export const DECODE_MATH_BUSY = DECODE_OPS_PER_BYTE / BALANCE_OPS_PER_BYTE
export const PROMPT_BUS_BUSY = BALANCE_OPS_PER_BYTE / PROMPT_OPS_PER_BYTE
export const DECODE_BUSY_SMS = Math.max(1, Math.round(CARD.smCount * DECODE_MATH_BUSY))

export type Phase = "decode" | "prompt"

export function opsPerByte(phase: Phase): number {
  return phase === "decode" ? DECODE_OPS_PER_BYTE : PROMPT_OPS_PER_BYTE
}

export function percent(fraction: number): string {
  const value = fraction * 100
  return `${value < 10 ? value.toFixed(0) : Math.round(value)}%`
}

export function approx(value: number): string {
  if (value >= 100) return String(Math.round(value / 10) * 10)
  if (value >= 10) return String(Math.round(value))
  return value.toFixed(1)
}

export const PEAK_TFLOPS_LABEL = (PEAK_FLOPS / 1e12).toFixed(1)
export const BANDWIDTH_LABEL = `${Math.round(BANDWIDTH_BYTES_PER_S / 1e9)} GB/s`

export const READOUTS: Record<Phase, string> = {
  decode:
    `Decode: about ${approx(DECODE_OPS_PER_BYTE)} operations per byte, and the card needs about ` +
    `${approx(BALANCE_OPS_PER_BYTE)} before its math sets the pace. The bus runs flat out, the SMs mostly wait.`,
  prompt:
    `A ${PROMPT_TOKENS}-token prompt uses every weight ${PROMPT_TOKENS} times: about ` +
    `${approx(PROMPT_OPS_PER_BYTE)} operations per byte. Now the SMs run flat out and the bus mostly waits.`,
}

// x of a value on a log axis of three decades starting at 1.
export function logX(value: number, x0: number, x1: number): number {
  return x0 + (Math.log10(value) / 3) * (x1 - x0)
}
