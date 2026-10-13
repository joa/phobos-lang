---
title: Models and formats
description: The model architectures phobos-cli runs, the models it has been tested with, and the GGUF weight formats it reads.
sidebar:
  order: 1
---

`phobos-cli` reads GGUF files. A GGUF file names its architecture and carries
its own tokenizer and, for most chat models, its own chat template, so one
`--gguf FILE` is all a model needs.

## Architectures

Phobos picks the forward pass from the architecture the file declares in
`general.architecture`:

| Architecture | Forward pass |
| --- | --- |
| `llama` | The Llama family. |
| `qwen35` | Qwen3.5 and its relatives, including the gated delta-rule layers. |
| `qwen35moe` | The mixture-of-experts variant of `qwen35`, with experts streamed from host memory when they do not fit on the card. |
| `glm4` | GLM-4. |

Any other architecture stops the load with "no forward pass is implemented
for the '...' architecture". To see what a file declares before loading it,
run the `inspect` example from a checkout. It needs no GPU:

```sh
cargo run -r -p phobos-gguf --example inspect -- MODEL.gguf
```

It prints the metadata, the tensors by shape, and a histogram of the
quantization types the file uses.

## Tested models

Inference has been tested with these files:

- [MiniCPM5-1B-Q8_0](https://huggingface.co/Abiray/MiniCPM5-1B-GGUF)
- [Qwen3.5-0.8B-Q8_0](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF)
- [Qwen3.5-4B-Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
- [Qwen3.8-27B-UD-IQ1_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [Ternary-Bonsai-2-27B-PTQ1_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
- [Qwen3.6-35B-A3B-UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [GLM-4.6V-Flash-Q4_K_M](https://huggingface.co/unsloth/GLM-4.6V-Flash-GGUF), text only

The GPU path has been verified against llama.cpp: token for token where
greedy decoding is stable, and by the gaps between the top next-token logits
where it is not. Measured speeds for these models are on the
[Benchmarks](/docs/reference/benchmarks/) page.

The two 27B models fill most of an 8 GB card, and the 35B mixture of experts
does not fit at all. See [Troubleshooting](/docs/inference/troubleshooting/)
for what that means in practice, and the next section for how the 35B runs.

## Models larger than the card

A mixture-of-experts model (`qwen35moe`) holds most of its weights in its
experts, and only a few experts run for each token. Phobos keeps everything
but the experts on the GPU, keeps the experts in host memory, and holds a
cache of experts on the card. During decoding, experts in the cache run on
the GPU while CPU threads compute the missing ones from host memory at the
same time. There is no manual split to choose.

The cache takes the device memory the resident weights leave, less a reserve.
`--expert-cache SIZE` sets it instead, as `2g`, `1500m` or a byte count:

```sh
phobos-cli --gguf models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --expert-cache 2g
```

When another program takes memory on the card while the model runs, the
cache gives it back, and more experts are computed on the CPU. The
environment variables that tune this path are listed under
[the device backend: streamed experts](/docs/reference/environment/#the-device-backend-streamed-experts).

## Weight formats

A quantized GGUF file stores most weights in blocks: a run of values sharing
one scale, packed into a few bits each. A file named for one format, such as
`Q4_K_M`, usually mixes several; `inspect` shows the mix.

Phobos reads these formats. Formats with a kernel are computed on directly
from their packed blocks. The others are expanded when the model is uploaded
to the card and then take the dense path, which costs more device memory than
the file size suggests.

| Format | Values per block | Bytes per block | Bits per weight | On the GPU |
| --- | ---: | ---: | ---: | --- |
| `Q8_0` | 32 | 34 | 8.5 | kernel |
| `Q5_0` | 32 | 22 | 5.5 | kernel |
| `Q4_0` | 32 | 18 | 4.5 | expanded at upload |
| `Q4_1` | 32 | 20 | 5.0 | expanded at upload |
| `Q5_1` | 32 | 24 | 6.0 | expanded at upload |
| `Q8_1` | 32 | 36 | 9.0 | expanded at upload |
| `Q2_K` | 256 | 84 | 2.625 | kernel |
| `Q3_K` | 256 | 110 | 3.4375 | kernel |
| `Q4_K` | 256 | 144 | 4.5 | kernel |
| `Q5_K` | 256 | 176 | 5.5 | kernel |
| `Q6_K` | 256 | 210 | 6.5625 | kernel |
| `IQ1_S` | 256 | 50 | 1.5625 | kernel |
| `IQ1_M` | 256 | 56 | 1.75 | kernel |
| `IQ2_XXS` | 256 | 66 | 2.0625 | kernel |
| `IQ2_XS` | 256 | 74 | 2.3125 | kernel |
| `IQ2_S` | 256 | 82 | 2.5625 | kernel |
| `IQ3_XXS` | 256 | 98 | 3.0625 | kernel |
| `IQ3_S` | 256 | 110 | 3.4375 | kernel |
| `IQ4_XS` | 256 | 136 | 4.25 | kernel |
| `PTQ1_0` | 128 | 28 | 1.75 | kernel |

Unquantized tensors in `F32`, `F16` and `BF16` load as well. `PTQ1_0` is
PrismML's ternary format, which no upstream GGML type code carries; its
weights are `-d`, `0` or `+d`. `Q8_K` and `IQ4_NL` are recognized type codes
that Phobos does not decode.

The kernels behind these formats are written in the Phobos language. Some are
built-ins of the compiler, listed in the spec under
[Quantized built-ins](/docs/reference/spec/#quantized-built-ins); the
[language guide](/docs/language/quantized/) explains them.

## How much memory a model needs

From a checkout, the `footprint` example reports what a model will occupy on
the GPU: the weights, how much of that is `f32`, and the cache per token of
context. Built with `--features cuda`, it also reports what the card has
free:

```sh
cargo run -r --features cuda -p phobos-gguf --example footprint -- MODEL.gguf
```
