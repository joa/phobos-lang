# Benchmarks

Phobos offers `phobos-bench` which measures prompt processing (pp) and
token generation (tg) throughput.

See [details](#details) for information on how to run these benchmarks locally.
`scripts/bench.py` measures Phobos and llama.cpp interleaved, and checks the
card for contention first.

![Phobos vs llama.cpp, tokens per second](results/inference.svg)

![Phobos vs cuBLAS, achieved GFLOP/s](results/bench.svg)

Qwen3.5-0.8B-Q8_0 on an RTX 2080 SUPER, driver 610.88, tokens per second:

| test   | llama.cpp CUDA[^1]  | Phobos GPU         |
| ------ | ------------------: | -----------------: |
| pp128  |  6865.64 +/-  20.15 | 5598.11 +/- 192.96 |
| pp512  | 10871.30 +/-   9.37 | 8491.03 +/-  17.11 |
| tg32   |   241.50 +/-   0.18 |  329.45 +/-   0.27 |
| tg128  |   260.12 +/-   0.09 |  328.37 +/-   0.14 |
| tg512  |   263.78 +/-   0.10 |  328.63 +/-   0.05 |
| tg1024 |   263.35 +/-   0.09 |  327.67 +/-   0.07 |
| tg2048 |   262.47 +/-   0.05 |  326.14 +/-   0.08 |

MiniCPM5-1B-Q8_0 on an RTX 2080 SUPER, driver 610.88, tokens per second:

| test   | llama.cpp CUDA[^1]  | Phobos GPU          |
| ------ | ------------------: | ------------------: |
| pp128  |  8613.27 +/-  30.14 |  7574.73 +/-  27.89 |
| pp512  | 16235.99 +/-  34.08 | 10114.45 +/-  20.40 |
| tg32   |   280.79 +/-   0.15 |   318.95 +/-   0.33 |
| tg128  |   281.81 +/-   0.18 |   317.78 +/-   0.18 |
| tg512  |   280.80 +/-   0.13 |   316.35 +/-   0.12 |
| tg1024 |   280.01 +/-   0.16 |   313.77 +/-   0.13 |
| tg2048 |   277.07 +/-   0.14 |   308.68 +/-   0.12 |

Qwen3.5-4B-Q4_K_M on an RTX 2080 SUPER, driver 610.88, tokens per second:

| test   | llama.cpp CUDA[^1] | Phobos GPU         |
| ------ | -----------------: | -----------------: |
| pp128  | 2317.77 +/-   1.83 | 2307.98 +/-   3.33 |
| pp512  | 2956.77 +/-   3.66 | 2352.83 +/-  24.31 |
| tg32   |  107.04 +/-   0.07 |  113.53 +/-   0.08 |
| tg128  |  110.07 +/-   0.06 |  113.29 +/-   0.05 |
| tg512  |  110.59 +/-   0.04 |  113.30 +/-   0.04 |
| tg1024 |  110.44 +/-   0.05 |  113.15 +/-   0.06 |
| tg2048 |  109.87 +/-   0.03 |  112.60 +/-   0.05 |

Qwen3.8-27B-UD-IQ1_M on an RTX 2080 SUPER, driver 610.88, tokens per second:

| test  | llama.cpp CUDA[^1] | Phobos GPU      |
| ----- | -----------------: | --------------: |
| pp128 |   476.96 +/-  1.56 | 532.60 +/- 0.74 |
| tg128 |    22.00 +/-  0.00 |  31.36 +/- 0.09 |

Ternary-Bonsai-2-27B-PTQ1_0 on an RTX 2080 SUPER, driver 610.88, tokens per second:

| test  | llama.cpp-prism CUDA[^2] | Phobos GPU       |
| ----- | -----------------------: | ---------------: |
| pp128 |         302.11 +/-  0.78 | 583.49 +/- 15.04 |
| tg128 |          29.36 +/-  0.01 |  46.12 +/-  0.01 |

The two 27B models need 6.27 GiB (IQ1_M) and 5.53 GiB (PTQ1_0) for their
weights, which leaves little of the card's 8 GiB. Their numbers only hold while
the desktop uses little VRAM: 1377 MiB was in use before the IQ1_M run, and
979 MiB before the PTQ1_0 run. If the desktop uses much more, the model no
longer fits and the driver pages it over PCIe. An earlier session measured
that at 7 t/s.

## Mixture of Experts

Qwen3.6-35B-A3B-UD-Q4_K_M, a mixture of 256 experts whose 19.5 GB do not fit the card, on an RTX 2080 SUPER, driver 610.88, tokens per second:

| test  | llama.cpp CUDA[^1], `-ncmoe 31 -t 8` | Phobos GPU       |
| ----- | -----------------------------------: | ---------------: |
| pp128 |                      75.68 +/-  0.32 | 329.33 +/- 0.75  |
| pp512 |                     256.91 +/-  0.55 | 598.85 +/- 2.13  |
| tg128 |                      31.41 +/-  0.75 |  55.52 +/- 0.15  |

Both engines keep everything except the experts on the GPU. They differ in
where the experts go.

**llama.cpp** decides at load time which layers keep their experts on the CPU.
We used its fastest setting on this card, found by sweeping `llama-bench` at a
3k context:

- `-ncmoe 31`: experts of the first 31 layers on the CPU. At 30 the card is at
  the edge of running out of memory, and at 29 it starts paging.
- `-t 8`: 8 CPU threads decode 12% faster than the default.

**Phobos** keeps a cache of experts on the GPU, sized so the context still has
room to grow.

- Decode: experts in the cache run on the GPU. At the same time, CPU threads
  compute the missing experts from host memory. That takes a quarter of the
  time copying the expert to the GPU would. The CPU hands its results to the
  GPU through mapped memory.
- Prompt processing: Phobos copies experts to the GPU over PCIe and computes
  the rest on the CPU at the same time.

The link on this machine is PCIe 3.0 x8, 6.4 GB/s.

## Real-World Load

The tables above come from a synthetic benchmark. `scripts/agent_bench.py`
measures a real workload instead:

1. It runs the [pi](https://pi.dev) coding agent on the tasks in `bench/` and
   records every request the agent sends.
2. It replays the recording to each engine, with every answer capped at its
   recorded length. Both engines get the same prompts in the same order. Each
   keeps its session between requests and rewinds it to where the next prompt
   differs.

The model is Qwen3.6-35B-A3B. Both engines use the same sampler, a 16k context
and one slot, and llama.cpp uses the settings above. Results for the fib task,
8 requests, three rounds each:

| no thinking      | wall   | prompt    | decode     |
| ---------------- | -----: | --------: | ---------: |
| llama.cpp CUDA   | 71.9 s | 132 t/s   | 29.9 t/s   |
| Phobos GPU       | 44.0 s | 237 t/s   | 42.3 t/s   |

| thinking         | wall    | prompt    | decode     | tokens/round |
| ---------------- | ------: | --------: | ---------: | -----------: |
| llama.cpp CUDA   | 149.1 s | 105 t/s   | 30.1 t/s   |        2,005 |
| Phobos GPU       | 147.8 s | 193 t/s   | 42.0 t/s   |       ~4,400 |

With thinking, the two wall times cover different amounts of work. The replay
caps each answer at its recorded length, but it cannot make an engine keep
going. llama.cpp ends the long reasoning turn early, so Phobos writes about
twice as many tokens in the same time.

The context grows to 4k tokens without thinking and 8.6k with it. That is why
decode is slower here than in the benchmark tables.

## Details

The 0.8B, MiniCPM5-1B and 4B tables come from one run of both engines.

```bash
python scripts/bench.py -p 128 512 -n 32 128 512 1024 2048 -r 3 -R 5 \
  --csv results/bench.csv --json results/bench.json
```

The Qwen 27B is slow, so this run uses fewer sizes and repetitions. It is also
close to the card's memory limit, so it measures only one prompt size and one
generation size.

```bash
python scripts/bench.py -m models/Qwen3.8-27B-UD-IQ1_M.gguf -p 128 -n 128 \
  -r 1 -R 3 --csv results/bench-qwen38.csv --json results/bench-qwen38.json
```

The ternary 27B, against PrismML's llama.cpp fork.

```bash
python scripts/bench.py -m models/Ternary-Bonsai-2-27B-PTQ1_0.gguf -p 128 -n 128 \
  -r 1 -R 3 --llama-bench ${llama_cpp_prism}/llama-bench.exe \
  --csv results/bench-bonsai.csv --json results/bench-bonsai.json
```

The 35B mixture of experts model. llama.cpp runs its experts for layers 0 to 30
on the CPU (`-ncmoe 31`) with 8 threads (`-t 8`). This was its fastest
setting on the RTX 2080 SUPER at a 3k context.

```bash
python scripts/bench.py -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -p 128 512 -n 128 \
  -r 1 -R 3 --llama-args="-ncmoe 31 -t 8" \
  --csv results/bench-qwen36moe.csv --json results/bench-qwen36moe.json
```

Record a [pi](https://pi.dev) session, then replay it to both engines.

```bash
python scripts/agent_bench.py --engines phobos -r 1 [--thinking]
python scripts/agent_bench.py -r 3 [--thinking] --replay RUN/phobos-rep1-fib.requests.jsonl \
  --llama-arg=-ncmoe --llama-arg=31 --llama-arg=-t --llama-arg=8
```

Producing the SVG for the results.

```bash
python scripts/plot.py results/bench.json results/bench-qwen38.json \
  results/bench-bonsai.json results/bench-qwen36moe.json -o results/inference.svg
```

Manually invoking the engines (not a comparison).

```bash
llama-bench -p 512 -n 128 -m ${models}/Qwen3.5-0.8B-Q8_0.gguf -r 10
cargo run --features cuda --release -p phobos-bench -- -m ${models}/Qwen3.5-0.8B-Q8_0.gguf -p 512 -n 128 -r 10
```

Measuring cuBLAS/GEMM performance. It's a separate benchmark that autotunes the
kernels.

```bash
cargo run -r -p phobos-kbench -- --csv results/results.csv
python scripts/plot_bench.py results/results.csv -o results/bench.svg
```

[^1]: build: 4d19b2876 (10636)
[^2]: [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), build: 7dffb158d (10685)
