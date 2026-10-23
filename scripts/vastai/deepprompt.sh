#!/usr/bin/env bash
# Where a deep prompt's time goes on one rented card: Phobos built from REF
# and llama.cpp on Qwen3.5-4B Q4_K_M at pp512 and pp32768, Phobos once per
# flash attention shape (PHOBOS_ATTN_FLASH_SHAPE), then one 32k pass of each
# engine under nsys, its kernels summed. Wants /root/build.sh beside it and the
# devel image. Run it under nohup; /root/results/STAGE says which stage runs.
#
#     bash deepprompt.sh REF [SHAPE...]
set -uo pipefail
REF="${1:?usage: deepprompt.sh REF [SHAPE...]}"
shift
SHAPES=(default streamed16 resident16 resident32 resident16p resident32p)
[ $# -gt 0 ] && SHAPES=("$@")
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
apt-get update -qq && apt-get install -y -qq python3 curl ca-certificates libgomp1 build-essential git pkg-config libclang-dev libzstd-dev zlib1g-dev cuda-nsight-systems-12-8 > "$OUT/apt.log" 2>&1 || { stage APT-FAILED; exit 1; }
stage fetch+build
fetch > "$OUT/fetch.log" 2>&1 &
FETCH=$!
bash /root/build.sh "$REF" > "$OUT/build.log" 2>&1 || { stage BUILD-FAILED; exit 1; }
wait $FETCH || { stage FETCH-FAILED; exit 1; }
BIN=/root/src/target/release
M="$W/models/Qwen3.5-4B-Q4_K_M.gguf"
NSYS=$(ls /opt/nvidia/nsight-systems/*/bin/nsys /usr/local/cuda/bin/nsys 2>/dev/null | head -1)
"$BIN/phobos-bench" --version > "$OUT/version.txt"
nvidia-smi --query-gpu=name,driver_version,power.limit,clocks.max.sm --format=csv > "$OUT/card.txt"
for shape in "${SHAPES[@]}"; do
  stage "shape-$shape"
  if [ "$shape" = default ]; then unset PHOBOS_ATTN_FLASH_SHAPE; else export PHOBOS_ATTN_FLASH_SHAPE=$shape; fi
  # A first pass compiles this shape's kernels outside the timed one.
  "$BIN/phobos-bench" -m "$M" -p 32768 -n 0 -r 1 > /dev/null 2>&1
  "$BIN/phobos-bench" -m "$M" -p 512,32768 -n 0 -r 3 > "$OUT/phobos-$shape.log" 2>&1 || stage "SHAPE-$shape-FAILED"
done
unset PHOBOS_ATTN_FLASH_SHAPE
stage llama
"$W/llama/llama-bench" -m "$M" -p 512,32768 -n 0 -r 3 > "$OUT/llama.log" 2>&1 || stage LLAMA-FAILED
stage nsys
if [ -n "$NSYS" ]; then
  "$NSYS" profile -t cuda --cuda-graph-trace=node -o "$OUT/phobos-32k" -f true "$BIN/phobos-bench" -m "$M" -p 32768 -n 0 -r 1 > "$OUT/nsys-phobos.log" 2>&1
  "$NSYS" stats -q --report cuda_gpu_kern_sum --format csv -o "$OUT/phobos-32k" "$OUT/phobos-32k.nsys-rep" >> "$OUT/nsys-phobos.log" 2>&1
  "$NSYS" profile -t cuda --cuda-graph-trace=node -o "$OUT/llama-32k" -f true "$W/llama/llama-bench" -m "$M" -p 32768 -n 0 -r 1 > "$OUT/nsys-llama.log" 2>&1
  "$NSYS" stats -q --report cuda_gpu_kern_sum --format csv -o "$OUT/llama-32k" "$OUT/llama-32k.nsys-rep" >> "$OUT/nsys-llama.log" 2>&1
else
  stage NO-NSYS
fi
stage DONE
