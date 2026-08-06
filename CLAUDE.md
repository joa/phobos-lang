# Phobos

A tile-based GPU kernel language for distributed tensor algebra compiled through MLIR to PTX, executed via CUDA.

One compiler with two codegen paths:

1. AST -> PTX for leaf nodes
2. AST -> DAG for cluster instruction scheduling

Two model front ends sit on top of that language, GGUF and ONNX, and
`phobos-inference` runs either one on the kernels it emits.

Keep any parser/codegen changes in sync with `SPEC.md`.

## Rust
- Write elegant, idiomatic and clippy-clean rust code
- Always attach the unit to variable names when appropriate (e.g. time_millis, mem_bytes)
- Only comment what's non-obvious.
- Do NOT use unicode in comments or code (no em-dashes, arrows, smart quotes, or other non-ASCII characters). Prefer rephrasing into a comma-joined clause or a separate sentence over an ASCII dash; use `--` sparse and only when it makes sense. Do not use markdown formatting (exception: [`Rust`] doc links). 

## Verifying changes
- `cargo test`: the whole enchilada
- `cargo test -p phobos-lang`: parser + codegen tests; codegen tests run the MLIR verifier on emitted modules and grep the printed IR
- `cargo test -p phobos-gguf -p phobos-onnx -p phobos-inference`: the front ends, the host reference paths, the tokenizers and the sampler. No GPU, no MLIR toolchain
- `cargo run -p phobos-lang --example emit [file]`: prints the MLIR for a source file; fastest way to eyeball codegen output
- `cargo run -p phobos-bench`: Full GPU smoke test

The model path on the GPU needs `--features cuda`, and always `--release`: a
forward pass is a few hundred GFLOP and the debug build is unusable. Its checks
live in `phobos-inference/examples`:
- `backend_check`: every device op against the host reference
- `batch_check`: a batched pass against the same tokens fed one at a time
- `model_check`: whole-model logits, device against host
- `bench`: the two numbers `llama-bench` reports, `pp<N>` and `tg<N>`

Two habits that this project learned the hard way. An op that matches in
isolation can still be wrong, because a kernel writing past its output damages
the *next* allocation and never its own result, so a whole-model check is not
redundant with an op-by-op one. And a performance number is only comparable to
one measured in the same session on the same card; anything else is a
remembered figure and has to be labelled as such.

## Workspace layout
- **phobos-lang**: codegen via MLIR. See also `SPEC.md`
- **phobos-mlir**: `gen_ptx` runs the GPU lowering pipeline (gpu -> NVVM -> LLVM IR -> PTX via inkwell/NVPTX)
- **phobos-bench**: stand-alone compiler + benchmark binary: builds a kernel, compiles to PTX, launches it with `cust` (CUDA). Requires an NVIDIA GPU + CUDA toolkit at runtime
- **phobos-base**: shared config & logger (`Context`, GPU target config).
- **phobos-cluster**: distributed execution codegen and utils
- **phobos-sched**: global scheduler
- **phobos-pod**: node runtime
- **phobos-gguf**: GGUF container, dequantization, byte-level BPE, and the `qwen35` and `llama` forward passes.
- **phobos-onnx**: ONNX proto -> graph IR -> shape inference, folding, fusion -> Phobos kernels, with a host interpreter as the oracle
- **phobos-inference**: the CLI, the REPL, the OpenAI-compatible server, the sampler, and `DeviceBackend`, which implements phobos-gguf's `Backend` on Phobos kernels (TODO: move out of phobos-inference)