### Running Phobos on vast.ai

A rented card runs the Linux release as is: no build, no CUDA toolkit. The
package needs glibc 2.28, libstdc++ and the driver's `libcuda.so.1`, which the
NVIDIA container runtime mounts into every vast.ai container (with
`libnvidia-ml.so.1`, which the expert cache reads).

Every step below is billed by the minute, so prepare everything on the host
first and tear the instance down the moment the run is over.

### Setup

```powershell
$env:PATH="C:\Users\joaeb\AppData\Local\Python\pythoncore-3.14-64\Scripts;$env:PATH"
vastai show user        # credit
vastai show instances   # must be empty before and after a session
```

### Choosing an offer

The release's kernels are PTX ISA 9.0, which needs a CUDA 13 driver (R580 or
newer); a host on a 535 or 565 driver refuses to load them, so filter on
`cuda_max_good>=13.0`.

| Card     | Chip   | In the shipped cache | v0.1.1 smoke test |
|----------|--------|----------------------|-------------------|
| RTX 3090 | sm_86  | yes                  | passes            |
| RTX 4090 | sm_89  | yes                  | passes            |
| RTX 5090 | sm_120 | yes                  | fails             |

On the RTX 5090, v0.1.1 runs prompt processing and then stops at the first
decode attention with `raising a kernel's dynamic shared memory ceiling:
CUDA_ERROR_INVALID_VALUE`. Do not rent one to rediscover it. Branch
`fix-5090-attn-splits` (v0.1.1 plus the fix, same compiler fingerprint)
passes there.

To try a CI build against a release's cache, ship only the changed binaries
and pull the release on the box. The base image has no `xz`
(`apt-get install -y xz-utils`), and a tarball packed on Windows drops the
executable bit, so `chmod +x` after unpacking.

Traffic is billed per GB on top of `$/hr`, and the advertised `inet_down` is
the host's claim, not what reaches HuggingFace. Filter on both prices and sort
by the total:

```powershell
vastai search offers 'gpu_name=RTX_3090 num_gpus=1 verified=true rentable=true direct_port_count>=1 cuda_max_good>=13.0 disk_space>=30 inet_down>=500 inet_down_cost<=0.02 inet_up_cost<=0.02' -o 'dph_total'
```

In PowerShell 5.1, `ConvertFrom-Json` hands the parsed array down a pipe as
one object, so a `Where-Object` filter on it silently passes everything.
Unroll it first: `$offers = @(vastai search offers ... --raw | ConvertFrom-Json | ForEach-Object { $_ })`.

`dph_total` includes the disk. For the per-GB prices, the driver and where
the box is, `--raw | ConvertFrom-Json | Select-Object id,dph_total,inet_down_cost,inet_up_cost,inet_down,driver_version,geolocation`.

Skip hosts in mainland China (`geolocation` ending in `CN`): the first rental
was one, and HuggingFace reset its connections every few megabytes. Hosts may run
containers under gVisor (`extra_env.CONTAINER_RUNTIME` in
`vastai show instance ID --raw`); Phobos runs under it unchanged.

### Renting

A plain CUDA base image with no entrypoint of its own; the package brings
everything else. Phobos needs only glibc 2.28, but the llama.cpp Linux builds
the benchmark uses need 2.38, so rent Ubuntu 24.04: on 22.04 `llama-bench`
fails to load. 20 GB of disk holds the image, the package (650 MB extracted)
and a couple of small models.

```powershell
vastai create instance OFFER_ID --image nvidia/cuda:12.8.1-base-ubuntu24.04 --disk 20 --ssh --direct --label phobos-smoke --raw
vastai show instance INSTANCE_ID --raw   # poll until actual_status is "running", about a minute
                                         # intended_status "stopped" means the container died: destroy, next offer
vastai ssh-url INSTANCE_ID               # ssh://root@HOST:PORT
```

### Checking the link before anything else

The first thing on a fresh box is a speed test against the sources it will
pull from. One RTX 3090 host advertised 3100 Mbps and delivered 0.5 MB/s from
HuggingFace with repeated connection resets. If a 25-second sample is far
below 10 MB/s, destroy the instance and take another offer: an hour of a
cheap card is cheaper than waiting.

```sh
for u in https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf \
         https://github.com/joa/phobos-lang/releases/download/v0.1.1/phobos-v0.1.1-windows-x64.zip; do
  curl -sSL -r 0-99999999 -o /dev/null --max-time 25 -w "%{speed_download} B/s  $u\n" "$u"
done
```

Uploading a model from the workstation over `scp` is no way around it; the
home uplink is slower still.

### Smoke test

`scripts/vastai_smoke.sh` downloads the release and a model, runs
`phobos-bench` in the shape `scripts/record_kernels.py` recorded the shipped
cache with, a one-shot `phobos-cli` prompt, a `llama-bench`-shaped run, and
lists the kernels the box had to compile. Its log lands in `/root/smoke.log`.

```powershell
scp -P PORT scripts/vastai_smoke.sh root@HOST:/root/
ssh -p PORT root@HOST "touch ~/.no_auto_tmux; bash /root/vastai_smoke.sh v0.1.1"
scp -P PORT root@HOST:/root/smoke.log results/
```

Run the binaries from the package directory and never copy them elsewhere:
the shipped `kernel-cache/` is found beside the executable, and a copy in
`/usr/local/bin` compiles every kernel cold. A long HTTPS download is resumed
with `curl -C -` in a shell loop into a `.part` file, renamed once whole;
check its sha256 against the local copy before trusting a run. Never add
`--retry` to that curl: before each retry it truncates the file back to the
size it had at start, so on a flaky link the download never finishes.

### Cache misses

The shipped cache is read-only; anything the box compiles goes to
`~/.phobos/kernel-cache/sm_XX/`, so the files there are exactly the misses
(`phobos-cache list` shows only that directory, not the shipped one). An
empty directory means every kernel loaded from the release.

Expect a few misses on any card other than the one the manifest was recorded
on. v0.1.1 compiled six on an RTX 3090 and six on an RTX 4090, each a different
set of three `attention_persist` and three `fused`: the persistent kernels
whose grid is sized to the card's multiprocessor count, so their source
differs per card model and a manifest replay for another chip cannot produce
them. They cost the first load about 40 s instead of 0.5 s and nothing after. If the list is
long, or a single compile runs for minutes, the cache wants rewarming: destroy
the instance, warm on the workstation with `phobos-cache warm` (no GPU
needed), and come back with the new package.

### Benchmark against llama.cpp

`scripts/vastai_bench.sh` runs `scripts/bench.py` on the box: Phobos from a
release package (`--phobos-bench`, so the binary stays beside its
`kernel-cache/`) against a pinned llama.cpp Linux CUDA build, interleaved, with
the card checked for contention. Not yet run on a rented card.

- **Phobos:** the v0.1.1 package with the RTX 5090 fix laid over it. The
  overlay is `target/dist/v0.1.1-fix/overlay.tar.xz` (`phobos-cli` and
  `phobos-bench`, mode 755) with its `overlay.sha256`. It keeps v0.1.1's
  compiler fingerprint, so the shipped cache applies. The CI run it came from
  is 37769033303.
- **llama.cpp:** `b11497`, the `ubuntu-cuda-12.8-x64` build plus its `cudart`
  libraries in the same directory (both binaries carry `RUNPATH $ORIGIN`).
  The build BENCHMARKS.md used, 10636, has no Linux CUDA asset. CUDA 12.8 sits
  well inside the R580 and R595 drivers these hosts run.
- **Models:** HuggingFace `REPO/FILE`, fetched on the box. bench.py's default
  three come to 4.7 GB:
  `ggml-org/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf`,
  `Abiray/MiniCPM5-1B-GGUF/minicpm5-1b-Q8_0.gguf`,
  `unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf`.
  `unsloth/GLM-4.6V-Flash-GGUF/GLM-4.6V-Flash-Q4_K_M.gguf` and
  `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_M.gguf` add 12.9 GB. Rent
  `--disk 40` for those.

The fetch runs while the overlay uploads:

```powershell
scp -P PORT scripts/vastai_bench.sh scripts/bench.py root@HOST:/root/
ssh -p PORT root@HOST "nohup bash /root/vastai_bench.sh fetch ggml-org/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf Abiray/MiniCPM5-1B-GGUF/minicpm5-1b-Q8_0.gguf unsloth/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf > /root/fetch.log 2>&1 < /dev/null &"
scp -P PORT target/dist/v0.1.1-fix/overlay.tar.xz target/dist/v0.1.1-fix/overlay.sha256 root@HOST:/root/
ssh -p PORT root@HOST "until [ -f /root/bench/fetch.done ]; do sleep 5; done; bash /root/vastai_bench.sh run"
scp -P PORT -r root@HOST:/root/results results/vastai/bench-CARD
```

`run` passes anything after it to bench.py, for instance `-R 3` or `--force`.
Before `--force`, look at the idle readings at the top of `bench.log`: the
first RTX 5090 host reported 100% utilization with 2 MiB in use and no
processes. That is the reporting, not a tenant, and it is the one case
`--force` is for. `run` loads each model once before bench.py starts,
so the card's persistent kernels compile outside the timed rounds and a model
that does not load stops the run within a minute.

### Profiling without ncu

`ncu` fails on these hosts with `ERR_NVGPUCTRPERM`: the container gets no
`SYS_ADMIN`, and asking for it through `--env` is ignored. The tools in
`scripts/vastai/` build with the box's `gcc` (the two `.c` programs also need
`cuda.h`, copied from a local toolkit) and stand in for it:

- `cuprof.c`, an `LD_PRELOAD` shim: with `CUPROF=1` it replays each pass
  graph as single launches timed with events, and prints GPU time per kernel.
- `cuattr.c`, an `LD_PRELOAD` shim: registers and local memory the driver
  reports for every kernel it loads.
- `kq.c`: one real `q4k_qdot_i8_matvec` from a kernel-cache file, launched in
  isolation through the `push_descriptor` ABI across grid sizes. Time growing
  linearly with blocks below the SM count means the blocks serialize on
  something GPU-wide; that is how the 32-bit prefetch address was found.
- `tlb.c`: read bandwidth for an access shape, over `cuMemAlloc` and VMM
  memory.

### Tearing down

```powershell
vastai destroy instance INSTANCE_ID -y   # without -y it prompts, and a piped "y" does not reach it
vastai show instances   # confirm it is gone
```

`destroy`, not `stop`: a stopped instance still bills its disk.
