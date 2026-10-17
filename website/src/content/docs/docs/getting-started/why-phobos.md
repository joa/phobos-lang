---
title: Why Phobos?
description: How a project to learn GPUs grew a compiler, and then an inference engine to test it.
sidebar:
  order: 4
---

Phobos is a personal project by Joa Ebert. It started as a way to learn how
GPUs actually work. I believe you only understand something once you have done
the work, and since my background is compilers, optimization and LLVM, writing
a small kernel language for my old RTX 2080 SUPER was the approachable way in: first hand-written PTX, then MLIR
down to NVPTX, then tiles, tensor cores and autotuning.

Once SGEMM reached 76% of cuBLAS, more GEMM performance was no longer teaching
me much; it was diminishing returns on what the card's ISA offers. It has since
reached 86% on the same card, at `M=N=K=4096`. Inference
came next, to eat my own dog food: a real workload that would show whether the
compiler holds up outside a benchmark, and where it does not. Every kernel a
model runs is compiled from Phobos at runtime, so every gap in the language
shows up as a slow or wrong token.

Both steps are written up on my blog:

- [Phobos: A tiny scale-free kernel language](https://www.joa-ebert.com/posts/2026-07-07-phobos-lang/):
  the language, the compiler and the road to 76% of cuBLAS.
- [Filling in the blanks: Inference using Phobos](https://www.joa-ebert.com/posts/2026-10-06-phobos-inference/):
  from a compiler to an inference engine, and why most of the speed came from
  fighting memory.
