#!/usr/bin/env bash
# Builds phobos-bench and phobos-cli from a git ref on a rented Ubuntu box, so
# a tuning loop does not wait on CI. The first run fetches the toolchain the
# release builds with and Rust; later runs rebuild incrementally.
#
#     bash build.sh [REF]      # default main; a branch, tag or commit
#
# EXAMPLES="backend_check model_check" also builds those phobos-gguf examples,
# into /root/src/target/release/examples.
#
# The binaries land in /root/src/target/release. They compile kernels on first
# use into ~/.phobos/kernel-cache.
set -euo pipefail
REF="${1:-main}"
LLVM_VERSION=22.1.7
RUST_VERSION=1.96.0
SRC=/root/src
LLVM=/opt/llvm-install

if ! command -v cc >/dev/null || ! command -v git >/dev/null || [ -z "$(ls /usr/lib/llvm-*/lib/libclang.so 2>/dev/null)" ]; then
    apt-get update -qq
    apt-get install -y -qq build-essential git curl ca-certificates pkg-config libclang-dev libzstd-dev zlib1g-dev >/dev/null
fi
[ -d "$LLVM" ] || curl -sSfL "https://github.com/joa/phobos-lang/releases/download/toolchain-llvm-$LLVM_VERSION/llvm-$LLVM_VERSION-linux-x64.tar.gz" |
    tar -xz -C /opt
if ! command -v cargo >/dev/null; then
    [ -f "$HOME/.cargo/env" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain "$RUST_VERSION"
    source "$HOME/.cargo/env"
fi

[ -d "$SRC/.git" ] || git clone -q https://github.com/joa/phobos-lang.git "$SRC"
cd "$SRC"
git fetch -q origin
git checkout -q --detach "origin/$REF" 2>/dev/null || git checkout -q --detach "$REF"
echo "building $(git log --oneline -1)"

export LLVM_SYS_221_PREFIX="$LLVM" MLIR_SYS_220_PREFIX="$LLVM" MLIR_SYS_221_PREFIX="$LLVM" TABLEGEN_220_PREFIX="$LLVM"
LIBCLANG_PATH="$(dirname "$(ls /usr/lib/llvm-*/lib/libclang.so | sort -V | tail -1)")"
export LIBCLANG_PATH
# The driver's own libcuda, which the NVIDIA runtime mounts into the container.
export CUDA_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export PATH="$LLVM/bin:$PATH"
cargo build -q --release --locked -p phobos-bench -p phobos-cli --features phobos-bench/cuda,phobos-cli/cuda
for example in ${EXAMPLES:-}; do
    cargo build -q --release --locked -p phobos-gguf --features cuda --example "$example"
done
target/release/phobos-bench --version
