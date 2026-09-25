### Building LLVM+MLIR on Windows

MSVC is installed and available. 

Make sure you build **x64** LLVM if Rust is targeting **x64**.

Execute these commands in a `Developer Command Prompt for VS 2022` or similar.

**These instructions are for LLVM 22.x and will install LLVM to C:\llvm-install**

```powershell
git clone  --depth 1 --branch llvmorg-22.1.7 https://github.com/llvm/llvm-project.git

cd llvm-project

cmake -S llvm -B build `
  -DCMAKE_INSTALL_PREFIX="C:\llvm-install" `
  -DLLVM_ENABLE_PROJECTS="mlir;clang" `
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" `
  -DCMAKE_BUILD_TYPE=Release `
  -Thost=x64 `
  -DLLVM_ENABLE_ASSERTIONS=ON

cmake --build build --config Release --target install
```

### Environment Variables

Set the following environment variables

```powershell
$env:LLVM_SYS_221_PREFIX="C:\llvm-install"
$env:MLIR_SYS_220_PREFIX="C:\llvm-install"
$env:MLIR_SYS_221_PREFIX="C:\llvm-install"
$env:TABLEGEN_220_PREFIX="C:\llvm-install"
$env:PATH = "C:\llvm-install\bin;" + $env:PATH
```

`libclang.dll` is found through `llvm-config` on the path, so `LIBCLANG_PATH` is only
needed to point bindgen at a different one.