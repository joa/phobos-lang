// The compiler's output for examples/gemm_fp32.ph, as the scene reads it.

import data from "../../../data/pipeline/gemm_fp32.json"
import type { PipelineData, PipelineStructure } from "../pipeline-data"

export const structure: PipelineStructure = (data as unknown as PipelineData).structure

/** The passes view the scene animates: the variant without @aligned, which has buffers to pack. */
export function stagedPasses(view: PipelineStructure = structure) {
  return view.passes.find(pass => pass.recommended) ?? view.passes[0]
}

/** The kernel as written, for the numbers beside the variant. */
export function writtenPasses(view: PipelineStructure = structure) {
  return view.passes.find(pass => pass.variant === null) ?? view.passes[view.passes.length - 1]
}
