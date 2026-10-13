---
title: Examples
description: Five kernels from the repository, from a vector add to flash attention, explained.
sidebar:
  label: Overview
  order: 0
---

Each page shows one kernel from the repository as it is, and explains what it
computes and how. They are ordered so that each adds a little to the one
before.

| Kernel | What it adds |
| --- | --- |
| [saxpy](/docs/examples/saxpy-fp32/) | One dimension, one slice per CTA, elementwise math, masking. |
| [gemm, f32](/docs/examples/gemm-fp32/) | Two dimensions, a loop over k, tiles in shared memory, `dot`, autotuning. |
| [Q8_0 matvec](/docs/examples/q8-matvec/) | Quantized weights, a conversion, per-block scales, broadcasting. |
| [gemm on tensor cores](/docs/examples/gemm-fp16tc-fp32acc/) | The same gemm with `@tensorcore`: f16 operands, f32 accumulation. |
| [Flash attention, f32](/docs/examples/flash-attention-fp32/) | Several tiles carried across a loop, row reductions, an online softmax. |

All five compile with `phobos-compile`, and all but the matvec are among the
kernels `phobos-kbench` times; see
[Compile and benchmark](/docs/getting-started/compile-and-benchmark/).
