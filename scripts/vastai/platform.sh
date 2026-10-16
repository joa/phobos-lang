#!/usr/bin/env bash
# The platforms page's numbers for one rented card: Phobos built from REF
# against llama.cpp on Qwen3.5-4B Q4_K_M through bench.py, then phobos-kbench
# twice. Wants /root/build.sh and /root/bench.py beside it, and an image with
# cuBLAS (nvidia/cuda:*-devel-ubuntu24.04). Run it under nohup; each stage logs
# to /root/results and /root/results/STAGE says which one is running.
#
#     bash platform.sh REF [BENCH_ARGS]
set -uo pipefail
REF="${1:?usage: platform.sh REF [BENCH_ARGS]}"
shift
OUT=/root/results; W=/root/bench; LLAMA="${LLAMA:-b11681}"
mkdir -p "$OUT" "$W/models"
stage() { echo "$1 $(date +%T)" | tee -a "$OUT/STAGE"; }
fetch() {
  cd "$W"
  base=https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA
  curl -sSfL "$base/llama-$LLAMA-bin-ubuntu-cuda-12.8-x64.tar.gz" | tar -xz && mv "llama-$LLAMA" llama
  curl -sSfL "$base/cudart-llama-$LLAMA-bin-ubuntu-cuda-12.8-x64.tar.gz" | tar -xz --strip-components=1 -C llama
  f=Qwen3.5-4B-Q4_K_M.gguf
  until curl -sSfL -C - -o "models/$f.part" "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/$f"; do sleep 2; done
  mv "models/$f.part" "models/$f"; sha256sum "models/$f"
}
stage apt
apt-get update -qq && apt-get install -y -qq python3 curl ca-certificates libgomp1 openssl build-essential git pkg-config libclang-dev libzstd-dev zlib1g-dev > "$OUT/apt.log" 2>&1 || { stage APT-FAILED; exit 1; }
stage fetch+build
fetch > "$OUT/fetch.log" 2>&1 &
FETCH=$!
{
  # phobos-kbench links cuBLAS, which the devel image keeps here.
  export RUSTFLAGS="-L native=/usr/local/cuda/lib64"
  bash /root/build.sh "$REF" &&
  source "$HOME/.cargo/env" && cd /root/src &&
  LLVM=/opt/llvm-install && export LLVM_SYS_221_PREFIX="$LLVM" MLIR_SYS_220_PREFIX="$LLVM" MLIR_SYS_221_PREFIX="$LLVM" TABLEGEN_220_PREFIX="$LLVM" &&
  export LIBCLANG_PATH="$(dirname "$(ls /usr/lib/llvm-*/lib/libclang.so | sort -V | tail -1)")" CUDA_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu PATH="$LLVM/bin:$PATH" &&
  cargo build -q --release --locked -p phobos-kbench
} > "$OUT/build.log" 2>&1 || { stage BUILD-FAILED; exit 1; }
wait $FETCH || { stage FETCH-FAILED; exit 1; }
BIN=/root/src/target/release
M="$W/models/Qwen3.5-4B-Q4_K_M.gguf"
stage firstload
{ "$BIN/phobos-bench" --version; time "$BIN/phobos-bench" -m "$M" -p 16 -n 4 -r 1; } > "$OUT/firstload.log" 2>&1 || { stage FIRSTLOAD-FAILED; exit 1; }
for _ in $(seq 120); do w=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | cut -d. -f1); [ "$w" -lt 80 ] && break; sleep 1; done
stage bench
python3 -u /root/bench.py --phobos-bench "$BIN/phobos-bench" --llama-bench "$W/llama/llama-bench" -m "$M" \
  -p 128 512 -n 32 128 512 1024 2048 -r 3 -R 5 --csv "$OUT/bench.csv" --json "$OUT/bench.json" "$@" > "$OUT/bench.log" 2>&1 || stage BENCH-FAILED
nvidia-smi -q -d PERFORMANCE > "$OUT/perf-after-bench.txt"
for i in 1 2; do
  stage kbench$i
  (cd /root/src && "$BIN/phobos-kbench" --csv "$OUT/kbench$i.csv") > "$OUT/kbench$i.log" 2>&1 || stage KBENCH$i-FAILED
done
nvidia-smi -q -d PERFORMANCE > "$OUT/perf-after-kbench.txt"
stage DONE
