# Phobos

A tile-based GPU kernel language for distributed tensor algebra compiled through MLIR to PTX, executed via CUDA.

One compiler with two codegen paths:

1. AST -> PTX for leaf nodes
2. AST -> DAG for cluster instruction scheduling

Keep any parser/codegen changes in sync with `SPEC.md`.

## Rust
- Write elegant, idiomatic and clippy-clean rust code
- Always attach the unit to variable names when appropriate (e.g. time_millis, mem_bytes)
- Only comment what's non-obvious.
- Do NOT use unicode in comments or code (no em-dashes, arrows, smart quotes, or other non-ASCII characters). Prefer rephrasing into a comma-joined clause or a separate sentence over an ASCII dash; use `--` sparse and only when it makes sense. Do not use markdown formatting (exception: [`Rust`] doc links). 

## Verifying changes
- `cargo test`: the whole enchilada
- `cargo test -p phobos-lang`: parser + codegen tests; codegen tests run the MLIR verifier on emitted modules and grep the printed IR
- `cargo run -p phobos-lang --example emit [file]`: prints the MLIR for a source file; fastest way to eyeball codegen output
- `cargo run -p phobos-bench`: Full GPU smoke test

## Workspace layout
- **phobos-lang**: codegen via MLIR. See also `SPEC.md`
- **phobos-mlir**: `gen_ptx` runs the GPU lowering pipeline (gpu -> NVVM -> LLVM IR -> PTX via inkwell/NVPTX)
- **phobos-bench**: stand-alone compiler + benchmark binary: builds a kernel, compiles to PTX, launches it with `cust` (CUDA). Requires an NVIDIA GPU + CUDA toolkit at runtime
- **phobos-base**: shared config & logger (`Context`, GPU target config).
- **phobos-cluster**: distributed execution codegen and utils
- **phobos-sched**: global scheduler
- **phobos-pod**: node runtime
