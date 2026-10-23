#!/usr/bin/env bash
# The Benchmarks page's agent replay on one rented card: Phobos built from REF
# and llama.cpp each serve Qwen3.6-35B-A3B-UD-Q4_K_M while scripts/agent_bench.py
# replays the recorded requests in /root/traces, at a context of CTX tokens.
# llama.cpp fits its expert split to the card itself. Wants /root/build.sh
# beside it and the devel image, with disk for the 22 GB model. Run it under
# nohup; /root/results/STAGE says which stage runs.
#
#     bash agent.sh REF [CTX] [AGENT_BENCH_ARGS]
set -uo pipefail
REF="${1:?usage: agent.sh REF [CTX] [AGENT_BENCH_ARGS]}"
CTX="${2:-32768}"
shift; [ $# -gt 0 ] && shift
OUT=/root/results; W=/root/bench; LLAMA="${LLAMA:-b11681}"
M=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
mkdir -p "$OUT" "$W/models"
stage() { echo "$1 $(date +%T)" | tee -a "$OUT/STAGE"; }
fetch() {
  cd "$W"
  base=https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA
  curl -sSfL "$base/llama-$LLAMA-bin-ubuntu-cuda-12.8-x64.tar.gz" | tar -xz && mv "llama-$LLAMA" llama
  curl -sSfL "$base/cudart-llama-$LLAMA-bin-ubuntu-cuda-12.8-x64.tar.gz" | tar -xz --strip-components=1 -C llama
  until curl -sSfL -C - -o "models/$M.part" "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/$M"; do sleep 2; done
  mv "models/$M.part" "models/$M"; sha256sum "models/$M"
}
stage apt
apt-get update -qq && apt-get install -y -qq python3 curl ca-certificates libgomp1 build-essential git pkg-config libclang-dev libzstd-dev zlib1g-dev > "$OUT/apt.log" 2>&1 || { stage APT-FAILED; exit 1; }
stage fetch+build
fetch > "$OUT/fetch.log" 2>&1 &
FETCH=$!
bash /root/build.sh "$REF" > "$OUT/build.log" 2>&1 || { stage BUILD-FAILED; exit 1; }
wait $FETCH || { stage FETCH-FAILED; exit 1; }
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm,memory.total --format=csv > "$OUT/card.txt"
free -g > "$OUT/host.txt"; nproc >> "$OUT/host.txt"
stage agent
cd /root/src && python3 -u scripts/agent_bench.py --no-build --replay /root/traces/*.requests.jsonl -r 1 --ctx "$CTX" \
  -m "$W/models/$M" --llama-server "$W/llama/llama-server" --max-idle-power 200 --out "$OUT/agent" "$@" \
  > "$OUT/agent.log" 2>&1 || stage AGENT-FAILED
nvidia-smi -q -d PERFORMANCE > "$OUT/perf-after.txt"
stage DONE
