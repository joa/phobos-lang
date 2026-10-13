---
title: Reference
description: The language spec, environment variables, benchmarks and command-line tools.
sidebar:
  label: Overview
  order: 0
---

The reference pages are the repository's own files, rendered here as they
are. A change to one of them in the repository shows up on the site with the
next build.

| Page | Source | What it holds |
| --- | --- | --- |
| [Language spec](/docs/reference/spec/) | `SPEC.md` | The grammar, types, statements, built-ins and attributes. The language guide explains; the spec is what the compiler implements. |
| [Environment variables](/docs/reference/environment/) | `ENV.md` | Every `PHOBOS_*` variable the tree reads, what it takes and what it does. |
| [Benchmarks](/docs/reference/benchmarks/) | `BENCHMARKS.md` | Measured results against llama.cpp and cuBLAS, with the card, driver and commands. |
| [Command-line tools](/docs/reference/cli/) | the binaries' `--help` | The options of `phobos-cli`, `phobos-compile`, `phobos-bench` and `phobos-cache`. |
| [Building on Windows](/docs/reference/windows/) | `WINDOWS.md` | Building LLVM and MLIR with MSVC, and the environment the build reads. |
