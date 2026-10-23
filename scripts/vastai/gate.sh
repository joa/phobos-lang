#!/usr/bin/env bash
# The first thing on a rented card, before anything is spent: idle power and
# utilization, the driver's lifetime power-capping and thermal-slowdown
# counters, and the link to the two sources the benchmark pulls from. A counter
# of days, a high idle draw or a link far below 10 MB/s means destroy the
# instance and take another offer.
#
#     bash gate.sh
touch ~/.no_auto_tmux
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap,power.limit --format=csv
for _ in 1 2 3; do nvidia-smi --query-gpu=power.draw,utilization.gpu,clocks.sm,memory.used --format=csv,noheader; sleep 1; done
nvidia-smi -q -d PERFORMANCE | grep -iE "power cap|thermal slowdown"
for u in https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf \
         https://github.com/ggml-org/llama.cpp/releases/download/b11681/llama-b11681-bin-ubuntu-cuda-12.8-x64.tar.gz; do
  curl -sSL -r 0-99999999 -o /dev/null --max-time 20 -w "%{speed_download} B/s  ${u##*/}\n" "$u"
done
nproc; free -g | head -2; df -h /root | tail -1
