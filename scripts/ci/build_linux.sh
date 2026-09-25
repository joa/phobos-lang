#!/usr/bin/env bash
# Builds the release binaries inside quay.io/pypa/manylinux_2_28_x86_64
# against the toolchain from build_llvm.sh, checks they need nothing newer
# than glibc 2.28, and stages them with the compiler fingerprint.
#
#     scripts/ci/build_linux.sh LLVM_PREFIX OUT_DIR
#
# The CUDA driver library is linked against its stub; a user's driver
# provides the real one.

set -euo pipefail

LLVM="$(realpath "${1:?usage: build_linux.sh LLVM_PREFIX OUT_DIR}")"
OUT="$(realpath -m "${2:?usage: build_linux.sh LLVM_PREFIX OUT_DIR}")"
RUST_VERSION="${RUST_VERSION:-1.96.0}"
CUDA_PACKAGE="${CUDA_PACKAGE:-cuda-driver-devel-13-3}"

toolset="$(ls -d /opt/rh/gcc-toolset-* | sort -V | tail -1)"
source "$toolset/enable"

# bindgen wants a libclang for the MLIR and TableGen headers; any recent one
# parses them, so the distribution's does.
dnf install -y -q clang-devel
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y -q "$CUDA_PACKAGE"
cuda="$(ls -d /usr/local/cuda-* | sort -V | tail -1)"

curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain "$RUST_VERSION"
source "$HOME/.cargo/env"

export LLVM_SYS_221_PREFIX="$LLVM" MLIR_SYS_220_PREFIX="$LLVM" MLIR_SYS_221_PREFIX="$LLVM" TABLEGEN_220_PREFIX="$LLVM"
export LIBCLANG_PATH=/usr/lib64
export CUDA_LIBRARY_PATH="$cuda/lib64/stubs"
export PATH="$LLVM/bin:$PATH"

bins=(phobos-cli phobos-bench phobos-cache phobos-compile)
cargo build --release --locked \
    -p phobos-cli -p phobos-bench -p phobos-cache -p phobos-compile \
    --features phobos-cli/cuda,phobos-bench/cuda

mkdir -p "$OUT"
for bin in "${bins[@]}"; do
    cp "target/release/$bin" "$OUT/"
    # The newest symbol versions a binary asks for are its floor.
    newest_glibc="$(objdump -T "$OUT/$bin" | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1)"
    newest_glibcxx="$(objdump -T "$OUT/$bin" | grep -oE 'GLIBCXX_[0-9.]+' | sort -uV | tail -1 || true)"
    echo "$bin: $newest_glibc ${newest_glibcxx:-no GLIBCXX}"
    if [ "$(printf '%s\nGLIBC_2.28\n' "$newest_glibc" | sort -V | tail -1)" != "GLIBC_2.28" ]; then
        echo "$bin needs $newest_glibc, past the 2.28 floor" >&2
        exit 1
    fi
done
"$OUT/phobos-cache" --version > "$OUT/fingerprint.txt"
cat "$OUT/fingerprint.txt"
