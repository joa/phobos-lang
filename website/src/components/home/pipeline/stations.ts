// The seven stations from source to the GPU: what each is called, the line that says what the compiler
// holds there, how long the journey rests on it, and its drawing in either layout. Pure.

import { astStation } from "./ast"
import { structure } from "./data"
import { graphStation } from "./graph"
import { dimsText, gpuStation } from "./gpu"
import { mlirStation, regionDialects } from "./mlir"
import { passesStation, SWEEP_MS, SWEEP_START_MS } from "./passes"
import { ptxStation } from "./ptx"
import { sourceStation } from "./source"
import type { Layout, StationGeometry } from "./types"

export type StationId = "source" | "ast" | "graph" | "passes" | "mlir" | "ptx" | "gpu"

export interface Station {
  id: StationId
  label: string
  caption: string
  plate: string
  dwellMs: number
}

const graph = structure.graph
const mlirOps = Object.values(regionDialects(structure.mlir.root)).reduce((sum, count) => sum + count, 0)
const realNodes = graph.nodes.length

export const STATIONS: Station[] = [
  {
    id: "source",
    label: ".ph source",
    caption: "The kernel as written: an accumulator, a k loop over two tiles, and the store.",
    plate: "FIG / GEMM_FP32 / SOURCE",
    dwellMs: 5200,
  },
  {
    id: "ast",
    label: "AST",
    caption: `Parsed and checked, ${structure.ast.nodeCount} nodes: the for kept whole, every other statement folded to one.`,
    plate: "FIG / GEMM_FP32 / AST",
    dwellMs: 5600,
  },
  {
    id: "graph",
    label: "graph IR",
    caption: `Ops and the values between them, before any memory is placed: ${realNodes} nodes, ${graph.edges.length} edges, one value carried round the loop.`,
    plate: "FIG / GEMM_FP32 / GRAPH IR",
    dwellMs: 9000,
  },
  {
    id: "passes",
    label: "passes",
    caption: "Staging, pipelining, shared-memory placement and barrier elision, shown for the build without @aligned.",
    plate: "FIG / GEMM_FP32 WITHOUT @ALIGNED / SHARED",
    dwellMs: SWEEP_START_MS + SWEEP_MS + 2200,
  },
  {
    id: "mlir",
    label: "MLIR",
    caption: `The gpu, scf, vector, memref and arith dialects, ${mlirOps} ops in the kernel's regions, lowered through NVVM to LLVM IR.`,
    plate: "FIG / GEMM_FP32 / MLIR REGIONS",
    dwellMs: 6000,
  },
  {
    id: "ptx",
    label: "PTX",
    caption: `${structure.ptx.instructions} instructions from LLVM's NVPTX backend for ${structure.ptx.target.replace(".target ", "")}, by family.`,
    plate: "FIG / GEMM_FP32 / PTX MIX",
    dwellMs: 5600,
  },
  {
    id: "gpu",
    label: "GPU",
    caption: `Loaded and launched through CUDA: ${structure.gpu.ctas} CTAs of ${structure.gpu.threads} threads, the grid the host picks for ${dimsText(structure.gpu.dims)}.`,
    plate: "FIG / GEMM_FP32 / LAUNCH",
    dwellMs: 6200,
  },
]

/** The station the page shows without script or motion. */
export const OVERVIEW_STATION = STATIONS.findIndex(station => station.id === "graph")

export function stationGeometry(id: StationId, layout: Layout): StationGeometry {
  switch (id) {
    case "source":
      return sourceStation(layout)
    case "ast":
      return astStation(layout)
    case "graph":
      return graphStation(layout).geometry
    case "passes":
      return passesStation(layout)
    case "mlir":
      return mlirStation(layout)
    case "ptx":
      return ptxStation(layout)
    case "gpu":
      return gpuStation(layout)
  }
}

export function allGeometry(layout: Layout): StationGeometry[] {
  return STATIONS.map(station => stationGeometry(station.id, layout))
}
