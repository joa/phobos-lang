#!/usr/bin/env bash
# Phobos smoke test on a rented vast.ai box. Logs everything to ~/smoke.log.
#     bash vastai_smoke.sh [TAG]
set -uo pipefail
TAG="${1:-v0.1.1}"
WORK=/root/phobos
LOG=/root/smoke.log
MODEL_URL=https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== $* ($(date -u +%H:%M:%S))"; }

step "system"
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used --format=csv
grep PRETTY /etc/os-release; ldd --version | head -1; nproc; free -g | head -2; df -h /root | tail -1
ls /usr/lib/x86_64-linux-gnu/ | grep -E 'libcuda.so|libnvidia-ml.so' || echo "no libcuda/libnvidia-ml in /usr/lib/x86_64-linux-gnu"

step "tools"
command -v curl >/dev/null || { apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null; }

step "download"
mkdir -p "$WORK" && cd "$WORK"
[ -d "phobos-$TAG-linux-x64" ] || curl -sSfL "https://github.com/joa/phobos-lang/releases/download/$TAG/phobos-$TAG-linux-x64.tar.gz" | tar -xz
# Hosts drop long HTTPS transfers; resume until the file is whole. No
# --retry: curl rewinds the file to its starting size before each retry.
if [ ! -f Qwen3.5-0.8B-Q8_0.gguf ]; then
    until curl -sSfL -C - -o model.part "$MODEL_URL"; do sleep 2; done
    mv model.part Qwen3.5-0.8B-Q8_0.gguf
fi
ls -la "$WORK" "$WORK/phobos-$TAG-linux-x64"
BIN="$WORK/phobos-$TAG-linux-x64"
MODEL="$WORK/Qwen3.5-0.8B-Q8_0.gguf"

# Run from the package directory: the shipped cache is found beside the binary.
step "version and cache"
"$BIN/phobos-cache" --version
"$BIN/phobos-cache" list 2>&1 | head -20
rm -rf /root/.phobos/kernel-cache

# The shape scripts/record_kernels.py recorded, so every kernel should hit.
step "phobos-bench, manifest shape"
"$BIN/phobos-bench" -m "$MODEL" -p 7,100,512,600 -n 16 -d 0,2048 -r 1 --no-warmup
echo "bench exit $?"

step "phobos-cli oneshot"
"$BIN/phobos-cli" --gguf "$MODEL" -n 96 --temp 0 "What is the capital of France? Answer in one sentence."
echo "cli exit $?"

step "phobos-bench, llama-bench shape"
"$BIN/phobos-bench" -m "$MODEL" -p 128,512 -n 128 -r 3
echo "bench exit $?"

# Any file here is a kernel the shipped cache lacked and this box compiled.
step "cache misses"
find /root/.phobos/kernel-cache -type f 2>/dev/null | sed 's|.*/||' | sort | tee /root/misses.txt
echo "misses: $(wc -l < /root/misses.txt)"
step "done"
