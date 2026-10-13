// TraceSource: how the visualizer reaches a trace. `precomputed` serves the committed traces in
// src/data/traces, one lazily loaded chunk each; a live in-browser build would be a second source.

import type { Trace } from "./trace"

export interface TraceSource {
  load(name: string, chip: string): Promise<Trace>
}

const modules = import.meta.glob<Trace>("/src/data/traces/*.json", { import: "default" })

export const precomputed: TraceSource = {
  async load(name, chip) {
    const load = modules[`/src/data/traces/${name}.${chip}.json`]
    if (!load) throw new Error(`no trace ${name}.${chip}`)
    return load()
  },
}
