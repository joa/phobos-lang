---
title: Compile and benchmark
description: Turn a kernel into PTX with phobos-compile, read its MLIR, and time it on the GPU with phobos-kbench.
sidebar:
  order: 3
---

Three tools take a kernel from source to a number. `phobos-compile` turns a
source file into PTX. The `emit` example prints the MLIR the compiler
generates, which is the quickest way to see what a change to a kernel does.
`phobos-kbench` compiles, autotunes and times the bundled kernels on the GPU,
next to cuBLAS where there is a comparison. Only the last one needs a GPU.

## phobos-compile

```text
usage: phobos-compile FILE.ph [--chip CHIP] [--index-bits 32|64] [-o OUT.ptx]
```

`phobos-compile` reads one source file and writes PTX for every kernel in it,
to standard output or to the file `-o` names:

```sh
phobos-compile examples/gemm_fp32.ph -o gemm.ptx
```

From a checkout, run it through cargo:

```sh
cargo run -r -p phobos-compile -- examples/gemm_fp32.ph --chip sm_80 -o gemm.ptx
```

The options:

- `--chip` picks the GPU architecture to compile for. The default is `sm_75`,
  which every supported card runs. The chips a release supports are `sm_75`,
  `sm_80`, `sm_86`, `sm_89`, `sm_90` and `sm_120`.
- `--index-bits` sets the width of index arithmetic, 32 by default. A module
  that needs 64-bit indices is widened to 64 the way the runtime would widen
  it, so a kernel compiles here exactly when it compiles there: one that loads
  tensor-core operands with `ldmatrix`, uses `load_acquire` or
  `store_release`, or runs `mma.sync` on `sm_80` and newer.
- `PHOBOS_PRINT_PHASES=1` in the environment prints the IR after each lowering
  phase.

Tile constants declared with `@autotune` take the first value of their search
space, since nothing is tuned here.

The PTX is what the CUDA driver loads. To launch it from your own program you
need the argument layout Phobos uses, which the
[`phobos-kernels`](https://github.com/joa/phobos-lang/tree/main/phobos-kernels)
crate implements together with a launcher.

## Read the MLIR

From a checkout, the `emit` example of `phobos-lang` prints the MLIR module
the compiler generates for a source file, after running the MLIR verifier on
it. It needs no GPU:

```sh
cargo run -p phobos-lang --example emit -- examples/saxpy_fp32.ph
```

Without a file, it prints a built-in matmul.

Two environment variables pick the target, as `--chip` and `--index-bits` do
for `phobos-compile`:

| Variable | Takes | Default |
| --- | --- | --- |
| `PHOBOS_CHIP` | an SM target, such as `sm_75` or `sm_80` | `sm_75` |
| `PHOBOS_INDEX_BITS` | `32` or `64` | `32` |

They matter because some code paths are only taken on some targets.
`mma.sync` and `cp.async` both want 64-bit indices, and the two diverge again
between `sm_75` and `sm_80`. To see the tensor-core gemm the way an Ampere
card would run it:

```sh
PHOBOS_CHIP=sm_80 PHOBOS_INDEX_BITS=64 cargo run -p phobos-lang --example emit -- examples/gemm_fp16tc_fp32acc.ph
```

In PowerShell, set the variables first with `$env:PHOBOS_CHIP = "sm_80"` and
`$env:PHOBOS_INDEX_BITS = "64"`. Every variable the tree reads is listed in
[Environment variables](/docs/reference/environment/).

## phobos-kbench

`phobos-kbench` needs an NVIDIA GPU and the CUDA driver. From a checkout:

```sh
cargo run -r -p phobos-kbench
```

It compiles and autotunes each bundled kernel, runs the winner, and prints
its throughput, next to cuBLAS where a comparison exists. The kernels are the
files in [`examples/`](https://github.com/joa/phobos-lang/tree/main/examples):

| Benchmark | Source | What it is |
| --- | --- | --- |
| `saxpy_fp32` | `saxpy_fp32.ph` | `out = alpha * x + y` |
| `gemm_fp32` | `gemm_fp32.ph` | f32 matrix multiply |
| `gemm_fp16tc_fp32acc` | `gemm_fp16tc_fp32acc.ph` | the same on tensor cores, f16 operands, f32 accumulation |
| `gemm_fp16` | `gemm_fp16.ph` | f16 inputs, output and accumulation |
| `flash_fp32` | `flash_attention_fp32.ph` | attention with an online softmax |
| `flash_fp16` | `flash_attention_fp16.ph` | f16 Q, K, V and O, f32 softmax state |

It runs all of them by default. The useful flags:

| Flag | Effect |
| --- | --- |
| `--bench NAME` | Run one benchmark. |
| `--autotune "DIM=VAL ..."` | Fix the tuning values for the selected `--bench` and skip the search, such as `--bench gemm_fp16 --autotune "TILE_M=256 TILE_N=128 TILE_K=16"`. |
| `--csv [PATH]` | Write the achieved GFLOP/s against the theoretical peak to a CSV file, `phobos-kbench.csv` by default. |
| `--peak-fp32`, `--peak-fp16tc`, `--peak-fp16tcf32acc TFLOPS` | Override the detected peak throughput the CSV compares against. |
| `--help` | List every flag. |

A benchmark is only comparable with one measured in the same session on the
same card. A GPU that is busy with something else, or that has not yet
clocked up, gives lower numbers. The published results, with the card they
were measured on, are in [Benchmarks](/docs/reference/benchmarks/).

## Next

- The [examples](/docs/examples/) explain the bundled kernels one by one.
- The [language guide](/docs/language/) covers each concept in turn.
- The [language spec](/docs/reference/spec/) is the complete reference.
