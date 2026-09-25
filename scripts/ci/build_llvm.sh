#!/usr/bin/env bash
# Builds the LLVM and MLIR a Linux release links against, inside
# quay.io/pypa/manylinux_2_28_x86_64, so everything built on it runs on
# glibc 2.28. Static libraries only, NVPTX and X86 only, and none of the
# optional system libraries, so llvm-config --system-libs asks for nothing a
# user's machine might lack.
#
#     scripts/ci/build_llvm.sh OUT_DIR
#
# Writes OUT_DIR/llvm-$LLVM_VERSION-linux-x64.tar.gz holding llvm-install/.

set -euo pipefail

LLVM_VERSION="${LLVM_VERSION:-22.1.7}"
OUT="$(realpath "${1:?usage: build_llvm.sh OUT_DIR}")"
WORK="${WORK:-/tmp/llvm-work}"

# The newest gcc-toolset the image carries; its libstdc++ links the parts
# newer than the system's statically, which is what keeps the floor at 2.28.
toolset="$(ls -d /opt/rh/gcc-toolset-* | sort -V | tail -1)"
source "$toolset/enable"
gcc --version | head -1
command -v ninja >/dev/null || pipx install ninja

mkdir -p "$WORK" "$OUT"
cd "$WORK"
if [ ! -d llvm-project ]; then
    git clone --depth 1 --branch "llvmorg-$LLVM_VERSION" https://github.com/llvm/llvm-project.git
fi

cmake -S llvm-project/llvm -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$WORK/llvm-install" \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DMLIR_INCLUDE_TESTS=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF \
    -DLLVM_PARALLEL_LINK_JOBS=2
cmake --build build --target install

tar -C "$WORK" -czf "$OUT/llvm-$LLVM_VERSION-linux-x64.tar.gz" llvm-install
ls -l "$OUT"
