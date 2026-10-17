#!/usr/bin/env bash
# The platforms page's kernel rows for one rented card: phobos-kbench built
# from REF and run twice, the second reusing the first's tuned winners.
# Wants /root/build.sh beside it and an image with cuBLAS
# (nvidia/cuda:*-devel-ubuntu24.04). /root/src may be cloned from a git
# bundle first, for a ref that is not on GitHub. Run it under nohup; each
# stage logs to /root/results and /root/results/STAGE says which one runs.
#
#     bash kbench.sh REF
set -uo pipefail
REF="${1:?usage: kbench.sh REF}"
OUT=/root/results
mkdir -p "$OUT"
stage() { echo "$1 $(date +%T)" | tee -a "$OUT/STAGE"; }
stage build
{
  # phobos-kbench links cuBLAS, which the devel image keeps here.
  export RUSTFLAGS="-L native=/usr/local/cuda/lib64"
  bash /root/build.sh "$REF" &&
  source "$HOME/.cargo/env" && cd /root/src &&
  LLVM=/opt/llvm-install && export LLVM_SYS_221_PREFIX="$LLVM" MLIR_SYS_220_PREFIX="$LLVM" MLIR_SYS_221_PREFIX="$LLVM" TABLEGEN_220_PREFIX="$LLVM" &&
  export LIBCLANG_PATH="$(dirname "$(ls /usr/lib/llvm-*/lib/libclang.so | sort -V | tail -1)")" CUDA_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu PATH="$LLVM/bin:$PATH" &&
  cargo build -q --release --locked -p phobos-kbench
} > "$OUT/build.log" 2>&1 || { stage BUILD-FAILED; exit 1; }
BIN=/root/src/target/release
nvidia-smi -q -d PERFORMANCE > "$OUT/perf-before.txt"
for i in 1 2; do
  stage kbench$i
  (cd /root/src && "$BIN/phobos-kbench" --csv "$OUT/kbench$i.csv" --json "$OUT/kbench$i.json") > "$OUT/kbench$i.log" 2>&1 || stage KBENCH$i-FAILED
done
nvidia-smi -q -d PERFORMANCE > "$OUT/perf-after.txt"
stage DONE
