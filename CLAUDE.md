# Phobos

A tile-based GPU kernel language for distributed tensor algebra compiled through MLIR to PTX, executed via CUDA.

One compiler with two codegen paths:

1. AST -> PTX for leaf nodes
2. AST -> DAG for cluster instruction scheduling

Two model front ends sit on top of that language, GGUF and ONNX. Each carries a
host backend and a GPU one, and each implements the traits `phobos-inference`
defines, so the runtime above them, the sampler, the chat rendering and the
server, never names a model format. `phobos-cli` is the binary that picks one.

Keep any parser/codegen changes in sync with `SPEC.md`.

## Rust
- Write elegant, idiomatic and clippy-clean rust code
- Always attach the unit to variable names when appropriate (e.g. time_millis, mem_bytes)
- Only comment what's non-obvious.
- Do NOT use unicode in comments or code (no em-dashes, arrows, smart quotes, or other non-ASCII characters). Prefer rephrasing into a comma-joined clause or a separate sentence over an ASCII dash; use `--` sparse and only when it makes sense. Do not use markdown formatting (exception: [`Rust`] doc links). 

## Verifying changes
- `cargo test`: the whole enchilada
- `cargo test -p phobos-lang`: parser + codegen tests; codegen tests run the MLIR verifier on emitted modules and grep the printed IR
- `cargo test -p phobos-gguf -p phobos-onnx -p phobos-inference -p phobos-kernels`: the front ends, the host reference paths, the tokenizers, the chat rendering and the sampler. No GPU, no MLIR toolchain
- `cargo run -p phobos-lang --example emit [file]`: prints the MLIR for a source file; fastest way to eyeball codegen output
- `cargo run -p phobos-bench`: Full GPU smoke test

The model path on the GPU needs `--features cuda`, and always `--release`: a
forward pass is a few hundred GFLOP and the debug build is unusable. Its checks
live in `phobos-gguf/examples`:
- `backend_check`: every device op against the host reference
- `batch_check`: a batched pass against the same tokens fed one at a time
- `model_check`: whole-model logits, device against host
- `bench`: the two numbers `llama-bench` reports, `pp<N>` and `tg<N>`

The ONNX device path has its own, in `phobos-onnx/examples`: `mm_check`,
`run_gpt2_gpu` and `chain_gpt2`.

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
- **phobos-kernels**: what both front ends need to reach a GPU: the launch ABI, the compile step, the launcher, the allocation pool and the plain f32 matmul. The `cuda` feature gates everything that talks to the driver, so the ABI and the kernel sources still compile without one
- **phobos-gguf**: GGUF container, dequantization, byte-level BPE, the `qwen35` and `llama` forward passes, and `backend/` with the host and device implementations of its `Backend` trait
- **phobos-onnx**: ONNX proto -> graph IR -> shape inference, folding, fusion -> Phobos kernels, and `backend/` with the host interpreter as the oracle alongside the device paths. Ships the GPT-2 tokenizer its exports are paired with
- **phobos-inference**: the traits (`Model`, `Session`, `Tokenizer`), the sampler, the generation loop, the chat rendering and the OpenAI-compatible server. Depends on neither front end, which is what lets both depend on it
- **phobos-cli**: the `phobos-cli` binary. Argument parsing, the REPL, and the one `match` that decides GGUF or ONNX
