//! Fingerprints the compiler that turns a kernel source into PTX, so the disk
//! cache can key on it.
//!
//! It has to change whenever the generated PTX could and stay put otherwise,
//! so it hashes what reaches the generated code: every `.rs` under the crates
//! that lower a kernel, and the toolchain versions from `Cargo.lock`. Not the
//! calling executable, which would give each binary its own cold compile.

use std::{
    fs,
    path::{Path, PathBuf},
};

/// Crates whose source can change what a kernel compiles to.
const CODEGEN_CRATES: [&str; 4] = ["phobos-lang", "phobos-mlir", "phobos-base", "phobos-kernels"];

/// Dependencies whose version can change what a kernel compiles to, even with
/// our own source untouched.
const TOOLCHAIN: [&str; 4] = ["melior", "mlir-sys", "llvm-sys", "inkwell"];

fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .to_path_buf();

    let mut parts: Vec<String> = Vec::new();
    for crate_name in CODEGEN_CRATES {
        let src = root.join(crate_name).join("src");
        println!("cargo:rerun-if-changed={}", src.display());
        let mut files = Vec::new();
        collect_rs(&src, &mut files);
        files.sort();
        for file in files {
            parts.push(file.strip_prefix(&root).unwrap_or(&file).display().to_string());
            parts.push(fs::read_to_string(&file).unwrap_or_default());
        }
    }

    let lock = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    parts.extend(toolchain_versions(&lock));

    // A build's own fingerprint, folded from the parts above. Not a hash of
    // the whole workspace: a change to the server or the CLI leaves it alone.
    let digest = fold(&parts);
    println!("cargo:rustc-env=PHOBOS_COMPILER_FINGERPRINT={digest}");
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The `name version` line for each toolchain crate, in lock-file order.
fn toolchain_versions(lock: &Path) -> Vec<String> {
    let Ok(text) = fs::read_to_string(lock) else { return Vec::new() };
    let mut out = Vec::new();
    let mut name: Option<&str> = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("name = \"") {
            name = rest.strip_suffix('"');
        } else if let Some(rest) = line.strip_prefix("version = \"") {
            if let (Some(n), Some(v)) = (name, rest.strip_suffix('"'))
                && TOOLCHAIN.contains(&n)
            {
                out.push(format!("{n} {v}"));
            }
            name = None;
        }
    }
    out
}

/// FNV-1a over the parts: not worth a hash crate in a build script, and the
/// input is our own source.
fn fold(parts: &[String]) -> String {
    let mut h: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    for part in parts {
        for byte in part.as_bytes() {
            h ^= u128::from(*byte);
            h = h.wrapping_mul(PRIME);
        }
        h ^= 0xff;
        h = h.wrapping_mul(PRIME);
    }
    format!("{h:032x}")
}
