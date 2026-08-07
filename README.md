# phobos

**EXPERIMENTAL** Tile-based kernel language for distributed tensor algebra. Inspired by [Triton](https://triton-lang.org).

```plain
@cluster(TILE_M in [16384, 65536], TILE_N in [16384, 65536], TILE_K in [16384, 65536])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel gemm(A: tensor<f32>[M, K],
            B: tensor<f32>[K, N],
            C: tensor<f32>[M, N],
            alpha: f32,
            beta: f32) {
  let pm = program_id(0)
  let pn = program_id(1)

  var acc: tile<f32>[TILE_M, TILE_N] = 0.0
  
  for kt in range(0, K, TILE_K) {
    var a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    var b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
  }
  
  let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
  C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = alpha * acc + beta * c_old
}
```

SGEMM performance is at 75% throughput of cuBLAS `cublasSgemm_v2` on a 2080 SUPER[^1] for `M=N=K=4096` fp32.
The same language runs LLM inference end to end: a quantized GGUF model running on phobos kernels generates at llama.cpp's rate on that card. See [Inference](#inference).

![Phobos benchmark results](results/bench.svg)

## Language

See [SPEC](./SPEC.md) for details or check out the [`examples/`](./examples).

## Autotuning

Phobos supports autotuning for finding the optimal configuration. 

![Running the gemm_fp32 benchmark](results/autotune.gif)

## Inference

Frontends for ONNX and GGUF sit atop the language and `phobos-cli` executes them.

Inference has been tested with:

- [MiniCPM5-1B-Q8_0](https://huggingface.co/Abiray/MiniCPM5-1B-GGUF)
- [Qwen3.5-0.8B-Q8_0](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF)
- [GPT2 (ONNX)](https://github.com/onnx/models/tree/main/validated/text/machine_comprehension/gpt-2)

**Note:**
* The host backend is used for verification and *very* slow. Always build with `--features=cuda` unless you need to verify against the host oracle. Always build with `--release` if the host backend is used.
* ONNX has not seen a lot of love (as in: it runs the OG GPT-2, but it's slow).

Two model front ends, each with a host backend and a GPU one:

- **`phobos-gguf`**: This has been verified against llama.cpp (token for token where greedy decoding is stable, 
  and by next-token logit gaps where it is not).
- **`phobos-onnx`**: ONNX protobuf to a graph IR, then shape inference, constant folding, LayerNorm
  and epilogue fusion. GPT-2 runs end to end and has been verified against its bundled reference, with a KV-cache path.

Qwen3.5-0.8B-Q8_0 on an RTX 2080 SUPER, tokens per second over ten repetitions:

| test   | llama.cpp CUDA[^2]  | Phobos GPU          |
| ------ | ------------------: | ------------------: |
| pp128  |  6280.82 +/- 812.25 |  4940.30 +/- 287.79 |
| pp512  | 10145.46 +/- 958.31 |  8951.88 +/- 154.39 |
| tg32   |   237.08 +/-   1.01 |   267.31 +/-   3.28 |
| tg128  |   255.87 +/-   0.57 |   261.67 +/-   1.08 |
| tg512  |   259.58 +/-   0.44 |   254.11 +/-   0.25 |

<details>
  <summary>Invocation Details</summary>
  
```plain
# running llama.cpp
llama-bench -p 512 -n 128 -m ${models}/Qwen3.5-0.8B-Q8_0.gguf -r 10

# running phobos
cargo run --features cuda --release -p phobos-gguf --example bench -- -m ${models}/Qwen3.5-0.8B-Q8_0.gguf -p 512 -n 128 -r 10
```
</details>

#### Running Locally

1. Download [minicpm5-1b-Q8_0.gguf](https://huggingface.co/Abiray/MiniCPM5-1B-GGUF) from HuggingFace.
2. Start the inference server with recommended model settings for temp etc.
  ```plain
  cargo run --features cuda -r -p phobos-cli -- --listen 127.0.0.1:8080 --gguf ~\models\minicpm5-1b-Q8_0.gguf --temp 0.9 --top-p 0.95 -n 32768
  ```
3. [pi.dev](https://pi.dev) `~/.pi/agent/models.json` entry:
  ```json
  {
    "providers": {
      "ollama": {
        "baseUrl": "http://127.0.0.1:8080/v1",
        "api": "openai-completions",
        "apiKey": "phobos",
        "models": [
          {
            "id": "minicpm5-1b-Q8_0",
            "input": ["text"],
            "compat": {
              "supportsDeveloperRole": false,
            },
            "contextWindow": 65536,
            "maxTokens": 32768,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
          }
        ]
      }
    }
  }
  ```
4. Happy hacking!

## Clustering

Phobos supports Hierarchical AMT with lineage recovery out of the box given the language is scale-free.

- **`phobos-sched`**: A central resource manager creates the global DAG and assigns sub-DAGs to specific nodes.
- **`phobos-pod`**: A node-level runtime takes the sub-DAG and schedules the fine-grained operations (threads, network fetches, memory allocations) dynamically and out-of-order.
- Communication via gRPC (see `phobos-cluster/proto`).

### Job Definition

Jobs are defined in a text format. 

Supported I/O protocols:
- **`file://`**: Must be available to the `pod` processes (local, NFS, ...)

```plain
source = /path/to/gemm_fp32.ph
dim M = 16384
dim N = 16384
dim K = 16384
tensor A = read  f32 16384x16384 file:///data/A.bin
tensor B = read  f32 16384x16384 file:///data/B.bin
tensor C = rmw   f32 16384x16384 file:///data/C.bin
scalar alpha = f32 0.125
scalar beta  = f32 1.5
```

### Example

Running one scheduler and two pods on the same host.
Reduce VRAM to 4 GiB per pod.

1. **Start scheduler**: `cargo run -p phobos-sched -- --listen 0.0.0.0:8881 --nodes 2 --job .\examples\matmul_cluster_job.txt --autotune --vram 4294967296`
2. **Start pod 0**: `cargo run -p phobos-pod -- --id 0 --sched 127.0.0.1:8881 --listen 0.0.0.0:8882 --arena 4294967296`
3. **Start pod 1**: `cargo run -p phobos-pod -- --id 1 --sched 127.0.0.1:8881 --listen 0.0.0.0:8883 --arena 4294967296`
4. Output will be written to the job's `tensor C` URI (`file:///data/C.bin` in the example above)

```plain
$ cargo run -p phobos-sched --            \
  --listen 0.0.0.0:8881                   \
  --nodes 2                               \
  --job .\examples\matmul_cluster_job.txt \
  --autotune                              \
  --vram 4294967296
phobos-sched: listening on 0.0.0.0:8881, waiting for 2 node(s)
[  0.000s  INFO] sched: planned 'matmul' supers=[("TILE_K", 4096), ("TILE_M", 4096), ("TILE_N", 8192)] ingest=DirectLoad budget=none nodes=2 segments/node=[1, 1] peak=704MiB fetch=0MiB
[  0.077s  INFO] sched: compiled 2 leaf kernel(s)
[  0.078s  INFO] sched: waiting for 2 node(s) to register
[ 39.692s  INFO] sched: node0 registered (tile server 127.0.0.1:8882)
[ 56.587s  INFO] sched: node1 registered (tile server 127.0.0.1:8883)
[ 56.589s  INFO] sched: 2 node(s) registered: ["node0=127.0.0.1:8882", "node1=127.0.0.1:8883"]
[ 56.591s  INFO] sched: -> node0 issue segment 0 (92 instrs: 24 ALLOC, 20 LOAD, 20 COMPUTE, 4 STORE, 24 FREE; incr 704MiB)
[ 56.593s  INFO] sched: -> node1 issue segment 1 (92 instrs: 24 ALLOC, 20 LOAD, 20 COMPUTE, 4 STORE, 24 FREE; incr 704MiB)
[ 71.059s  INFO] sched: all 184 instructions accounted (184 retired, 0 abandoned)
[ 71.061s  INFO] job done; outputs:
[ 71.062s  INFO]   file:///data/C.bin
```

## Command-Line Tools

Sample kernels (`.ph`) and job files live in [`examples/`](./examples).

### `phobos-cli` (optional GPU)

```plain
cargo run [--features cuda] -r -p phobos-cli -- [--gguf <file.gguf>] [-m|--model <dir>] [--kv <dir>]
                                                [-n|--num <tokens>] [--show <int>] [--listen <host:port>]
                                                [-t|--temp <float>] [-k|--top-k <int>] [-p|--top-p <float>]
                                                [--min-p <float>]
                                                [--presence-penalty <float>] [--repetition-penalty <float>]
                                                [--seed <int>]
                                                [prompt]
```

Perform inference. `--gguf` loads a GGUF model and dispatches on the architecture the file declares;
without it the exported ONNX engines load (`--kv` when that directory is present, else `--model`).
Uses the host backend when CUDA is not selected as a build feature; both backends produce the same logits.

Given a prompt it prints the continuation and exits. Without one it starts a REPL that keeps the model warm.

`--listen` runs a minimal OpenAI-compatible server (`/v1/completions`, `/v1/chat/completions`, `/v1/models`)
with SSE streaming, rendering conversations and tool definitions through the model's own
`tokenizer.chat_template`. The sampling flags above become defaults unless specified otherwise in the query.

### `phobos-bench` (needs a GPU)

```plain
cargo run -r -p phobos-bench
```

Compiles and autotunes the bundled kernels at 4096^3 and prints throughput (against cuBLAS where a shim exists). Covers `saxpy_fp32`, `gemm_fp32`, `gemm_fp16tc_fp32acc` (tensor-core, f16 inputs with f32 accumulation), `gemm_fp16` (f16 inputs, output, and accumulation), `flash_fp32`, and `flash_fp16` (f16 Q/K/V/O with an f32 online-softmax state). Runs all of them by default; pass `--bench NAME` to run a single one, or `--help` for the full flag list. `--autotune "DIM=VAL ..."` pins the autotune dims (skipping the search) for the selected `--bench`; `--csv [PATH]` writes achieved throughput; `--peak-fp32`/`--peak-fp16tc`/`--peak-fp16tcf32acc TFLOPS` override the detected roofline peaks.

### `phobos-sched`

```plain
cargo run -r -p phobos-sched -- --listen <host:port> --nodes <n> --job <file>
                                [--budget <bytes>] [--ingest direct|home-fetch]
                                [--autotune [--vram <bytes>] [--link-bw <bytes/s>] [--leaf-flops <flop/s>]]
```

The global scheduler daemon: waits for `--nodes` pods to register, plans and dispatches the job, and prints the output tensor URIs. `--budget` enables per-node memory-budgeted segmentation; `--autotune` picks the supertile config from a cost model (overridable via `--vram`/`--link-bw`/`--leaf-flops`).

### `phobos-pod` (needs a GPU)

```plain
cargo run -r -p phobos-pod -- --id <node-id> --sched <host:port>
                               [--listen <host:port>] [--advertise <host:port>] [--arena <bytes>]
```

The node runtime daemon (one process = one GPU). Attaches to the scheduler and executes the segments it is given. Use `--listen host:0` (the default) to let the OS pick a port; `--advertise` overrides the address peers FETCH from for a multi-host cluster. `--arena` sets the device arena size (default 512 MiB).

### `phobos-tensor`

```plain
cargo run -r -p phobos-cluster --bin phobos-tensor -- init --uri <file://...> --shape <RxC|N>
                                                           [--fill zero|random|const|iota] [--value <f>] [--seed <s>]
cargo run -r -p phobos-cluster --bin phobos-tensor -- peek --uri <file://...> --shape <RxC|N>
```
Seeds and inspects the `file://` f32 tensor blobs a job reads and writes. `init` creates a blob at full size (use `--fill zero` to pre-allocate an output tensor, since STORE seeks into an existing file); `peek` prints a few well-spread elements. Shapes are row-major `RxC` (rank-2) or `N` (rank-1).

## Examples

Run with `cargo run -p <crate> --example <name> -- <args>`.

| Example | Crate | Syntax | What it does |
| --- | --- | --- | --- |
| `emit` | `phobos-lang` | `emit -- [file.ph]` | Prints the emitted MLIR (defaults to a built-in matmul). No GPU. |
| `ptx` | `phobos-lang` | `ptx -- <file.ph> [chip] [index-bitwidth]` | Compiles a source file to PTX (default chip `sm_75`). No GPU. |
| `dag_dot` | `phobos-cluster` | `dag_dot -- <file.ph> [out.dot]` | Renders a `@cluster` kernel's parametric cluster IR as Graphviz DOT. No GPU. |
| `plan_dot` | `phobos-sched` | `plan_dot -- <job.txt> [--nodes N] [--ingest direct\|home-fetch] [out.dot]` | Lowers a job to the concrete per-node instruction DAG and renders it as Graphviz DOT. No GPU. |
| `cluster_bench` | `phobos-sched` | `cluster_bench` | Analytic, CPU-only scheduler benchmark: planner throughput and cost-model quality as node count grows. No GPU. |
| `loopback` | `phobos-pod` | `loopback` | Single-node scheduler + pod smoke test over gRPC. Needs a GPU. |
| `cluster_grpc` | `phobos-pod` | `cluster_grpc` | Scheduler + two pods exercising the peer FETCH data path. Needs a GPU. |
| `budgeted` | `phobos-pod` | `budgeted` | Memory-budgeted multi-segment execution with the cluster autotuner. Needs a GPU. |
| `bench_cluster` | `phobos-pod` | `bench_cluster` | Single-GPU end-to-end cost of the cluster machinery vs. a bare full-K kernel launch. Needs a GPU. |
| `cluster_correctness` | `phobos-pod` | `cluster_correctness` | Splits one large on-disk SGEMM across several in-process nodes and sample-checks against a CPU reference. Needs a GPU. |

Render a DOT graph with Graphviz, e.g. `cargo run -p phobos-cluster --example dag_dot -- examples/matmul_cluster_fp32.ph | dot -Tsvg -o dag.svg`.

The model path has its own set. Build these `--release`, and the ones needing a GPU also `--features cuda`.

| Example | Crate | Syntax | What it does |
| --- | --- | --- | --- |
| `inspect` | `phobos-gguf` | `inspect -- MODEL.gguf [--tensors] [--dump TENSOR]` | Metadata, tensor directory by shape, quantization histogram. No GPU. |
| `generate` | `phobos-gguf` | `generate -- MODEL.gguf -n 40 "prompt"` | Host-backend generation, dispatching on the file's architecture. This is the reference path. No GPU. |
| `encode` | `phobos-gguf` | `encode -- MODEL.gguf "text"` | Token ids from the file's own tokenizer, or JSON for a JSON array of prompts. No GPU. |
| `diagnose` | `phobos-gguf` | `diagnose -- MODEL.gguf` | Per-position NLL on a repeated phrase: a flat profile means no context is reaching the current position. No GPU. |
| `inspect_onnx` | `phobos-onnx` | `inspect_onnx -- model.onnx` | Opset, inputs/outputs, op-type histogram. No GPU. |
| `run_gpt2` | `phobos-onnx` | `run_gpt2` | A real exported GPT-2 through load, fold and the host interpreter, against its bundled reference. No GPU. |
| `run_gpt2_gpu` | `phobos-onnx` | `run_gpt2_gpu` | The same model with the Gemm projections and all 25 LayerNorms on Phobos kernels. Needs a GPU. |
| `kv_check` | `phobos-onnx` | `kv_check` | A single with-past step against the last row of a full recompute. No GPU. |
| `bench` | `phobos-gguf` | `bench -- -m MODEL.gguf -p 512 -n 128 -r 5` | The two numbers `llama-bench` reports, in the same units. Needs a GPU. |
| `backend_check` | `phobos-gguf` | `backend_check` | Every device op against the host reference. Needs a GPU. |
| `batch_check` | `phobos-gguf` | `batch_check -- MODEL.gguf` | A batched pass against the same tokens one at a time, and a split prompt against a whole one. Needs a GPU. |
| `model_check` | `phobos-gguf` | `model_check -- MODEL.gguf` | Whole-model logits, device against host. Needs a GPU. |

`phobos-gguf/examples` also holds the kernel sweeps each optimization was decided by (`q8sweep`,
`ppsweep`, `attnsweep`, `deltasweep`, `dotform`); `docs/GGUF.md` says what each one answered.

```
                                             ::::
                            ::-=====++++*********+-
                        =*#%@@@@@@@@@@%%##**++===++=.
                       *@@@%%%%#@@@@@%%#*++=----=+***=.
                      =#: ==.=- .*@@@%#+==--=+*#%@@@%#*=-.
                     -#+:-+=+=+=-=+#%#*+==*%@@@@@@%#**+=-+=:
                    -%#**++*++++***#%%#%%@@@@@@%##*+=:    :==-.
                   =@@@@%#######%@@@@@@@@@@@@%#**+-:        .==-.
                  *@@@@@@@%@@%%%@@@@@@@@@@%%###*++:           .-:
                .#@@@@@@@@@@@@@@@@@@@@@@@%%##***+++=.          .-.
               .*@@@@@@@@%%@%%%####**++--:..   ..:-=*+-.       -+:
               ..       .          .               :=+**+-. :=+**=.
               =.       :=.:::-=+*#*-:.  :==:       .:=+*****%##*+:
               +.       =+-+++*##%%#+-::=@@@%+:       .-+++*##***+-
              **:   :..:**+*##%@%@@%*=--#*=-::--       .-*+=****++=.
              @=:. .:..:*+++**#####*=-:::..... ..      ..-#+=***++=:
             +#::. ::..:*===++++++=--..               ....-#*=**+==:
             #+:::.:. .:*+---====---:                 :++=-=%++*+=-:.
             %++*=:.   .=+::----:::-.              -*@@@@@@%@@-++=-:.
            *@@@@*:.    .:.:--:::.::              *@@@@@@@@@@#-:+=-:.
           *@@@@@%-.       ::..:.::      .......-@@@@@@@@%*+====++-:.
            @@@@@@-.       .. . .::  ...:::::::*@@@@@@@%+*##***+=-:.
            @@@@@@+. .-----+--=-=-:..::::.:::-#@@@@@@@+=*#*++=--:..
            +@@@@@*-+@@@@@@@@@@@@#+-. ...:::=#%%%#%@#==**++=--:...
             @@@@@%%@@@@@@@@@@@@@@%%#*=-:-=+#######=-++==---::..
           -*@@@@%@@@@@@@@@@@@@@@%%%%%%#*+=+#%#%#+:-+==-----=+=-
             =@@@@@@@@@@@@@@@@@@%%%%%%@@@@@@@*%@=.-==----=+#%%%%#*=:.
       **+=-=+@@@@%@@@*+=+==+%@@%##**#%%@%*%@--=.-=--==+*%%%#####%###*=-
      *@@@@@@@##%*#%@%*=====+#%%#**+++=+++=:. .. :==++*######**++==+****
     +*****%@@*::=*##*+=-==---=***++=-:.      ..:.-*#%%%%%%%%###*+=-:.-+
    #%*=: :#%*+: :+*++:       .====-::.         :-*@@@@@@@@@@@@%%##*+:
   #%#*=. +#*++=  -=---:::::..::::::...        .#@@@@@@@@@@@@@@@@@%#*+--
  ##**+-.=#*+++-                             -#@@@@@@@@@@@@@@@@@@@@%#*+*
  #**=--*#***+                            .+@@@@@@@@@@q<3a@@@@@@@@@#**#%
  *+==+##**+:   -+*%@**+.  +#######+-+%=-*@@@@@@@@@@@@@@@@@@@@@@@@@#*#%@
  *++*##**+..=#@@@@@@@#-.-@@@@@@@@@%%@*-#@@@@@@@@@@@@@@@@@@@@@@@@@#*#@@#
  P          H                O              B            O            S
```

## AI Disclaimer

Claude Code was used, among the Gemini and Codex free tiers, when building
this project.

[^1]: [Table 2. GeForce RTX 3080 vs GeForce RTX 2080 / 2080 Super; P.14](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf)
[^2]: build: c629da565 (10219)