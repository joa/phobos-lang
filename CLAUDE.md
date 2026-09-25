# Phobos

A tile-based GPU kernel language for distributed tensor algebra compiled through MLIR to PTX, executed via CUDA.

One compiler with two codegen paths:

1. AST -> PTX for leaf nodes
2. AST -> DAG for cluster instruction scheduling

Two model front ends sit on top of that language, GGUF and ONNX. Each carries a
host backend and a GPU one, and each implements the traits `phobos-inference`
defines, so the runtime above them, the sampler, the chat rendering and the
server, never names a model format. `phobos-cli` is the binary that picks one.

Keep any parser/codegen changes in sync with `SPEC.md`. Every `PHOBOS_*`
environment variable the tree reads is documented in `ENV.md`, with the
tree's two toggle spellings (`env_flag` opt-in, `env_flag_on` opt-out);
a new one is added there in the same change, and read through one of those
two helpers rather than a spelling of its own.

## Rust
- Write elegant, idiomatic and clippy-clean rust code
- Always attach the unit to variable names when appropriate (e.g. time_millis, mem_bytes)
- Only comment what's non-obvious.
- Do NOT use unicode in comments or code (no em-dashes, arrows, smart quotes, or other non-ASCII characters). Prefer rephrasing into a comma-joined clause or a separate sentence over an ASCII dash; use `--` sparse and only when it makes sense. Do not use markdown formatting (exception: [`Rust`] doc links). 
- No `.rs` over 900 lines and no inline `mod tests` over 150; past that the tests
  move into a `tests.rs` beside the module and the module itself gets split.
  `phobos-base/tests/source_size.rs` enforces both. It carries a list of files
  that predate the cap, and that list is a ratchet: an entry may shrink, never
  grow, and has to be deleted once its file is under the cap. Do not add to it.
  The cap drops to 700 the day it is empty

## Verifying changes
- `cargo test`: the whole enchilada
- `cargo test -p phobos-lang`: parser + codegen tests; codegen tests run the MLIR verifier on emitted modules and grep the printed IR
- `cargo test -p phobos-gguf -p phobos-onnx -p phobos-inference -p phobos-kernels`: the front ends, the host reference paths, the tokenizers, the chat rendering and the sampler. No GPU, no MLIR toolchain
- `cargo run -p phobos-lang --example emit [file]`: prints the MLIR for a source
  file; fastest way to eyeball codegen output. `PHOBOS_CHIP` and
  `PHOBOS_INDEX_BITS` override the target, which is how to reach a path the
  default never takes: mma.sync and cp.async both want 64-bit indices, and the
  two diverge again between sm_75 and sm_80. A refactor of a code generator that
  changes the generated code is not a refactor, so diff this over every `.ph` in
  the tree, under each of those targets, before and after
- `cargo run -p phobos-kbench`: Full GPU smoke test

The model path on the GPU needs `--features cuda`, and always `--release`: a
forward pass is a few hundred GFLOP and the debug build is unusable. Its checks
live in `phobos-gguf/examples`:
- `backend_check`: every device op against the host reference
- `batch_check`: a batched pass against the same tokens fed one at a time
- `model_check`: whole-model logits, device against host
- `fuse_check`: the fused decode path against the launched one, both on the
  device in one session, since the device-against-host bound is too wide to see
  a fusion under
- `attndecode`: the decode attention path alone against cache length, over a
  working set the size of a real model's caches, with the card's measured copy
  bandwidth beside it. Warms the card itself: an idle GPU sits at its lowest
  clock and a short benchmark measures the ramp rather than the kernel
- `phobos-bench`, its own crate: the two numbers `llama-bench` reports,
  `pp<N>` and `tg<N>`; `python scripts/bench.py` runs it interleaved against
  llama.cpp on a card it checks for contention first
- `quant_check`: a quantized file against a wider one of the same model,
  tensor by tensor. Needs no GPU. This is what says a new format in
  `src/quant/` decodes correctly, and nothing else does: a wrong decoder still
  generates fluent text. Make the pair with
  `llama-quantize --allow-requantize WIDE.gguf NARROW.gguf Q4_K_M`

The ONNX device path has its own, in `phobos-onnx/examples`: `mm_check`,
`run_gpt2_gpu` and `chain_gpt2`.

Two habits that this project learned the hard way. An op that matches in
isolation can still be wrong, because a kernel writing past its output damages
the *next* allocation and never its own result, so a whole-model check is not
redundant with an op-by-op one. And a performance number is only comparable to
one measured in the same session on the same card; anything else is a
remembered figure and has to be labelled as such.

## Workspace layout
- **phobos-lang**: codegen via MLIR. See also `SPEC.md`. Everything the portable
  dialects cannot express goes through `codegen/target/`: `Isa` is the
  instruction vocabulary and `nvidia.rs` the one implementation, so the rest of
  `codegen/` never names an instruction and never reads `gpu_config`
- **phobos-mlir**: `gen_code` runs the GPU lowering pipeline (gpu -> NVVM -> LLVM
  IR -> PTX via inkwell/NVPTX), taking the pass pipeline and the LLVM machine
  from the target's `phobos_base::backend::Backend`. `gen_ptx` is a thin wrapper
- **phobos-kbench**: stand-alone kernel benchmark against cuBLAS: builds a kernel, compiles to PTX, launches it with `cust` (CUDA). Requires an NVIDIA GPU + CUDA toolkit at runtime
- **phobos-compile**: a kernel source to PTX for one chip, the compiler a release ships
- **phobos-bench**: a GGUF model's `pp<N>` and `tg<N>`, as `llama-bench` reports them
- **phobos-cache**: the compiled-kernel cache's tool, split one directory per chip. `warm` replays a manifest recorded with `PHOBOS_KERNEL_MANIFEST` (`scripts/record_kernels.py` records every GGUF model) for every supported chip with no GPU, one child process per job; `list`, `clear` and `prune` manage it. `scripts/release.py TAG` records, warms and packages a release around the binaries `.github/workflows/release.yml` builds
- **phobos-base**: shared config & logger (`Context`, GPU target config), `progress` and `log`, each a sink a front end installs so a library can report without knowing what draws it, the
  `Backend` trait that owns a target's lowering pipeline and its post-processing
  of the generated text, plus utilities used across the crates
- **phobos-cluster**: distributed execution codegen and utils
- **phobos-sched**: global scheduler
- **phobos-pod**: node runtime
- **phobos-kernels**: what both front ends need to reach a GPU: the launch ABI, the compile step, the launcher, the allocation pool and the plain f32 matmul. The `cuda` feature gates everything that talks to the driver, so the ABI and the kernel sources still compile without one
- **phobos-gguf**: GGUF container, byte-level BPE, the `qwen35` and `llama` forward passes, `quant/` with one file per quantized format behind a `Quant`/`Spec` registry, and `backend/` with the host and device implementations plus `fuse/`, the pass that lowers a chain of decode stages into one persistent kernel. A format with no kernel dequantizes at upload and takes the dense path, so a new one lands correct before it lands fast
- **phobos-onnx**: ONNX proto -> graph IR -> shape inference, folding, fusion -> Phobos kernels, and `backend/` with the host interpreter as the oracle alongside the device paths
- **phobos-inference**: inference traits, the byte-level BPE both front ends merge with (`bpe::ByteBpe`), the sampler, the generation loop, the chat rendering and the OpenAI-compatible server, plus `telemetry::Meter`, the shared record the engine writes progress into and a viewer reads snapshots of. What a front end can report about itself, and what it cannot, is the default-`None` half of the `Model` trait: `footprint`, `device_memory`, `device_info`, `architecture`, `cache_stats`, and `Session::truncate`, whose default accepts only the request that drops nothing, so a backend that cannot rewind still gets prefix reuse by extension
- **phobos-cli**: Argument parsing, the REPL, the one `match` that decides GGUF or ONNX, and `tui/`, the full-screen dashboard `--listen` puts on the terminal. The dashboard runs on its own thread and reads only a `Meter`, so it never names a model format and never blocks a decode; `--no-tui`, or output that is not a terminal, turns it off
