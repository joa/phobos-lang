---
title: Troubleshooting
description: Slow runs, models that do not fit, Windows paging, a busy GPU, and kernels that compile on every start.
sidebar:
  order: 6
---

## Everything is very slow, even a small model

Check how `phobos-cli` was built. Without the `cuda` feature it runs on the
host backend, which exists to verify the GPU path against and is very slow.
A debug build is too slow for a forward pass as well. Build with both:

```sh
cargo build --release --features cuda -p phobos-cli
```

A release download is built this way already.

## The first start takes minutes

A run compiles every kernel it does not find in the
[kernel cache](/docs/inference/kernel-cache/). With a release, keep the
`kernel-cache/` folder beside the binaries: it holds every kernel the
supported models need, and a first run then loads instead of compiling. The
cache only serves the build that filled it, so a binary built from source
compiles its own kernels once and keeps them in `~/.phobos/kernel-cache`.

## Kernels fail to load

Phobos emits PTX ISA 9.0, which needs a driver from the CUDA 13 line, R580 or
newer. An sm_120 card's kernels, compiled with libNVVM,
are PTX ISA 9.3 and need a driver that runs CUDA 13.3. The GPU must be of compute capability 7.5 or newer (Turing on). Update
the driver first.

## The model does not fit, or barely fits

A model needs room on the card for its weights, plus its cache, which grows
with the context, plus scratch space for each pass. To see what a model will
need before loading it, run the `footprint` example from a checkout:

```sh
cargo run -r --features cuda -p phobos-gguf --example footprint -- MODEL.gguf
```

It reports the weights, the cache per token, and what the card has free.

When a model only just fits:

- Close other programs that use the GPU. A desktop, a browser or a second
  model all take video memory, and the amount changes while you work.
- Start `phobos-cli` with `--no-prefix-cache`. The server then gives up
  reusing the previous request's session, and its buffers go back to the
  pass as scratch.
- For a mixture-of-experts model, `--expert-cache SIZE` caps the device memory
  its expert cache takes. Experts that are not cached are computed on the CPU
  instead. See [Models and formats](/docs/inference/models-and-formats/#models-larger-than-the-card).

`PHOBOS_VRAM=1` prints the free device memory at every pass boundary and
every large allocation, which shows where the memory goes. It turns the
dashboard off, since its lines would draw over it.

## On Windows, a model that ran fast becomes many times slower

On Windows, the driver does not fail an allocation when the card's memory
runs out. It moves allocations to host memory and reads them back over PCIe
when they are needed. A model that no longer fits keeps running, but every
token waits on the bus. The two 27B models in the
[Benchmarks](/docs/reference/benchmarks/) tables only reach their published
speed while the desktop uses little video memory, for this reason.

The fix is the same as above: free video memory, or run a smaller model or
quantization. Check what the desktop uses with `nvidia-smi` before a run.

Phobos already allocates its weights in a few large slabs rather than one
allocation per tensor, because the Windows driver keeps a small set of large
allocations resident where it fails a large set of small ones
(`PHOBOS_ARENA`, on by default).

## Benchmark numbers are lower than published

- **Another program is using the GPU.** Any other work on the card lowers the
  figures. `scripts/bench.py` refuses to start on a busy card and checks
  again between rounds; a hand-run `phobos-bench` does not.
- **A `PHOBOS_*` variable is set.** A variable left in the shell, such as
  `PHOBOS_VRAM`, changes what runs. `scripts/bench.py` prints every one it
  finds as `phobos env:` before it starts. Unset them for a clean run.
- **Memory is tight.** A model near the card's limit can be paged, as above.
- **The numbers come from different sessions.** A number is only comparable
  with one measured in the same session on the same card. The published
  tables name their card and driver.

See [Benchmarking](/docs/inference/benchmarking/).

## The dashboard does not appear

`--listen` draws a dashboard only when standard output is a terminal and
neither `PHOBOS_VRAM` nor `PHOBOS_PASS_REPORT` is set; otherwise it prints why
and logs to the terminal instead. `--tui` forces the dashboard where the
terminal was not recognized. See [Serving](/docs/inference/serving/#the-dashboard).

## "No forward pass is implemented for the ... architecture"

The file's architecture is not one Phobos runs. The supported ones are
listed in [Models and formats](/docs/inference/models-and-formats/#architectures).
