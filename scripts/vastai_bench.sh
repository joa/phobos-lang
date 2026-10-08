#!/usr/bin/env bash
# Phobos against llama.cpp on a rented vast.ai box, through bench.py.
#
#     bash vastai_bench.sh fetch REPO/FILE...   # release, llama.cpp, models
#     bash vastai_bench.sh run [BENCH_ARGS]     # once fetch is done
#
# fetch takes HuggingFace models as REPO/FILE and is meant to run under nohup
# while the host uploads. run expects /root/bench.py, and applies
# /root/overlay.tar.xz (checked against /root/overlay.sha256) over the release
# when it is there, renaming the package so the report names the build that
# ran. Results land in /root/results.
set -euo pipefail
TAG="${TAG:-v0.1.1}"
LLAMA="${LLAMA:-b11497}"
LLAMA_CUDA="${LLAMA_CUDA:-12.8}"
WORK=/root/bench
MODELS="$WORK/models"
OUT=/root/results
LLAMA_DIR="$WORK/llama-$LLAMA-cuda$LLAMA_CUDA"
# The overlaid package's directory, which names the build in the report.
FIXED="${FIXED:-phobos-$TAG-fix-linux-x64}"

get_model() {
    local file="${1##*/}"
    [ -f "$MODELS/$file" ] && return
    # No --retry: curl rewinds the file to its starting size before each one.
    until curl -sSfL -C - -o "$MODELS/$file.part" "https://huggingface.co/${1%/*}/resolve/main/$file"; do
        sleep 2
    done
    mv "$MODELS/$file.part" "$MODELS/$file"
}

fetch() {
    [ $# -gt 0 ] || { echo "fetch needs at least one REPO/FILE" >&2; exit 2; }
    mkdir -p "$MODELS"
    cd "$WORK"
    if ! command -v python3 >/dev/null || ! command -v xz >/dev/null || ! command -v curl >/dev/null; then
        apt-get update -qq && apt-get install -y -qq python3 xz-utils curl ca-certificates >/dev/null
    fi
    # llama.cpp links OpenMP and OpenSSL, which the CUDA base images lack.
    if ! ldconfig -p | grep -q libgomp.so.1 || ! ldconfig -p | grep -q libssl.so.3; then
        apt-get update -qq && apt-get install -y -qq libgomp1 openssl >/dev/null
    fi
    local gh=https://github.com
    [ -d "phobos-$TAG-linux-x64" ] || [ -d "$FIXED" ] ||
        curl -sSfL "$gh/joa/phobos-lang/releases/download/$TAG/phobos-$TAG-linux-x64.tar.gz" | tar -xz
    if [ ! -d "$LLAMA_DIR" ]; then
        local base="$gh/ggml-org/llama.cpp/releases/download/$LLAMA"
        curl -sSfL "$base/llama-$LLAMA-bin-ubuntu-cuda-$LLAMA_CUDA-x64.tar.gz" | tar -xz
        mv "llama-$LLAMA" "$LLAMA_DIR"
        # llama-bench and libggml-cuda.so carry RUNPATH $ORIGIN, so the CUDA
        # runtime goes beside them.
        curl -sSfL "$base/cudart-llama-$LLAMA-bin-ubuntu-cuda-$LLAMA_CUDA-x64.tar.gz" |
            tar -xz --strip-components=1 -C "$LLAMA_DIR"
    fi
    local pids=()
    for model in "$@"; do
        get_model "$model" &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    touch "$WORK/fetch.done"
}

run() {
    [ -f "$WORK/fetch.done" ] || { echo "fetch has not finished" >&2; exit 1; }
    cd "$WORK"
    local pkg="phobos-$TAG-linux-x64"
    if [ -d "$FIXED" ]; then
        pkg="$FIXED"
    elif [ -f /root/overlay.tar.xz ]; then
        tar -xJf /root/overlay.tar.xz -C "$pkg"
        (cd "$pkg" && sha256sum -c /root/overlay.sha256)
        mv "$pkg" "$FIXED"
        pkg="$FIXED"
    fi
    mkdir -p "$OUT"
    {
        "$WORK/$pkg/phobos-bench" --version
        # The llama.cpp Linux builds assume Ubuntu 24.04's glibc; on an older
        # image they fail to load, which bench.py would only report as a
        # skipped engine.
        if ldd "$LLAMA_DIR/llama-bench" "$LLAMA_DIR"/libggml-cuda.so 2>&1 | grep "not found"; then
            echo "llama.cpp $LLAMA cannot load here; rent a 24.04 image" >&2
            exit 1
        fi
        nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm --format=csv
        # gVisor hosts have read 100% utilization on an idle card; keep the
        # readings beside the result in case bench.py refuses on them.
        for _ in 1 2 3; do
            nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
            sleep 1
        done
        # Each model's first load compiles the card's persistent kernels.
        # Pay that here rather than inside a timed round, and fail fast on a
        # model that does not load.
        for model in "$MODELS"/*.gguf; do
            echo "first load: ${model##*/}"
            if ! out="$("$WORK/$pkg/phobos-bench" -m "$model" -p 16 -n 4 -r 1 2>&1)"; then
                echo "$out" | tail -5
                exit 1
            fi
            echo "$out" | grep "^model"
        done
        # The loads leave the card drawing well over idle power for a while,
        # which bench.py's contention check reads as another tenant.
        for _ in $(seq 90); do
            watts="$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | cut -d. -f1)"
            [ "$watts" -lt 80 ] && break
            sleep 1
        done
        echo "settled at ${watts} W"
        # Unbuffered, or the log shows nothing until bench.py exits.
        python3 -u /root/bench.py \
            --phobos-bench "$WORK/$pkg/phobos-bench" \
            --llama-bench "$LLAMA_DIR/llama-bench" \
            -m "$MODELS"/*.gguf \
            --csv "$OUT/bench.csv" --json "$OUT/bench.json" "$@"
    } 2>&1 | tee "$OUT/bench.log"
}

case "${1:-}" in
    fetch) shift; fetch "$@" ;;
    run) shift; run "$@" ;;
    *) sed -n '2,11p' "$0"; exit 2 ;;
esac
