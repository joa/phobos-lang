//! Persists compiled PTX to disk, keyed by everything that can change what a
//! kernel's source compiles to, so a cache hit survives a process restart and
//! is shared by every binary built from the same compiler.
//!
//! Set `PHOBOS_KERNEL_CACHE_DIR` to override the default
//! (`~/.phobos/kernel-cache`), or to an empty string to disable caching.
//! Skipped outright under `print_phases`, since a hit has nothing to print.
//!
//! `PHOBOS_KERNEL_CACHE_EPOCH` replaces the compiler fingerprint with its own
//! value. Editing one intrinsic otherwise invalidates every entry, which is
//! right for correctness and wrong for a benchmarking session that changed a
//! single kernel: pin the epoch, then evict just the kernels that moved with
//! `cargo run -p phobos-kernels --example cache -- evict <name>`. Entries are
//! named `<kernel>-<hash>` so that eviction is a glob rather than a guess.

use std::sync::OnceLock;

use crate::util::kernel_cache_dir as cache_dir;

use phobos_base::context::{Context, GpuConfig};
use sha2::{Digest, Sha256};

/// Identifies the compiler that produces the PTX: the codegen crates' source
/// and the MLIR/LLVM versions they call, folded at build time by `build.rs`.
/// Any change to either -- committed or not, a dependency bump, an LLVM
/// upgrade -- changes it, where a git commit hash would miss all three.
///
/// Deliberately not a hash of the running binary: two examples compiling
/// identical kernels would then share no entries and each pay a cold
/// compile.
fn build_fingerprint() -> &'static str {
    static FINGERPRINT: OnceLock<String> = OnceLock::new();
    FINGERPRINT.get_or_init(|| {
        std::env::var("PHOBOS_KERNEL_CACHE_EPOCH")
            .unwrap_or_else(|_| env!("PHOBOS_COMPILER_FINGERPRINT").to_string())
    })
}

/// The kernel's own name, so an entry can be found without its hash. The
/// hash is what makes the file unique, so a miss here is harmless.
fn kernel_name(source: &str) -> String {
    let name = source
        .split_once("kernel ")
        .and_then(|(_, rest)| rest.split_once('('))
        .map(|(name, _)| name.trim())
        .filter(|name| !name.is_empty())
        .unwrap_or("kernel");
    name.chars()
        .take(64)
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Cache key for a `(ctx, texts)` compile: every `Context` field that reaches
/// codegen, folded with the source text(s) and the build fingerprint.
fn hash(ctx: &Context, texts: &[&str]) -> String {
    let GpuConfig::Nvidia(nv) = &ctx.gpu_config;
    let mut overrides: Vec<_> = ctx.shape_overrides.iter().collect();
    overrides.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut hasher = Sha256::new();
    hasher.update(build_fingerprint().as_bytes());
    for part in [nv.chip(), nv.features(), nv.target_triple()] {
        hasher.update(part.as_bytes());
        hasher.update(b"\0");
    }
    hasher.update(ctx.index_bitwidth.to_le_bytes());
    for (name, value) in overrides {
        hasher.update(name.as_bytes());
        hasher.update(value.to_le_bytes());
    }
    for text in texts {
        hasher.update(text.as_bytes());
        hasher.update(b"\0");
    }
    format!("{}-{}", kernel_name(texts[0]), hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn write(ctx: &Context, texts: &[&str], bytes: &[u8]) {
    let Some(dir) = cache_dir() else { return };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    // Written under a per-process name and renamed into place so a
    // concurrent reader never sees a half-written entry.
    let tmp = dir.join(format!(".tmp-{}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, dir.join(hash(ctx, texts)));
    }
}

/// A cached `(ptx, shared)` pair for one kernel, or `None` on any cache miss,
/// corrupt entry, or disabled cache.
pub(crate) fn load(ctx: &Context, source: &str) -> Option<(String, Vec<(String, usize)>)> {
    if ctx.print_phases {
        return None;
    }
    let bytes = std::fs::read(cache_dir()?.join(hash(ctx, &[source]))).ok()?;
    decode_one(&bytes)
}

/// Persists one kernel's compiled output. Best-effort: a write failure never
/// fails a compile that already succeeded.
pub(crate) fn store(ctx: &Context, source: &str, ptx: &str, shared: &[(String, usize)]) {
    if ctx.print_phases {
        return;
    }
    write(ctx, &[source], &encode_one(ptx, shared));
}

/// [`load`] for [`crate::Variants`]'s aligned/general pair, which compile and
/// cache together; a hit also skips the `@pipeline` cross-variant check,
/// since that only has something to check right after a fresh compile.
pub(crate) fn load_pair(ctx: &Context, aligned_src: &str, general_src: &str) -> Option<(String, String)> {
    if ctx.print_phases {
        return None;
    }
    let bytes = std::fs::read(cache_dir()?.join(hash(ctx, &[aligned_src, general_src]))).ok()?;
    decode_pair(&bytes)
}

/// [`store`] for the aligned/general pair.
pub(crate) fn store_pair(
    ctx: &Context,
    aligned_src: &str,
    general_src: &str,
    aligned_ptx: &str,
    general_ptx: &str,
) {
    if ctx.print_phases {
        return;
    }
    write(
        ctx,
        &[aligned_src, general_src],
        &encode_pair(aligned_ptx, general_ptx),
    );
}

/// `shared.len() : u32`, then each `(name.len() : u32, name, bytes : u64)`,
/// then `ptx.len() : u64, ptx`.
fn encode_one(ptx: &str, shared: &[(String, usize)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend((shared.len() as u32).to_le_bytes());
    for (name, bytes) in shared {
        out.extend((name.len() as u32).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((*bytes as u64).to_le_bytes());
    }
    out.extend((ptx.len() as u64).to_le_bytes());
    out.extend(ptx.as_bytes());
    out
}

fn decode_one(bytes: &[u8]) -> Option<(String, Vec<(String, usize)>)> {
    let mut r = Reader::new(bytes);
    let shared_len = r.u32()? as usize;
    let mut shared = Vec::with_capacity(shared_len);
    for _ in 0..shared_len {
        let name_len = r.u32()? as usize;
        let name = r.utf8(name_len)?;
        let value = r.u64()? as usize;
        shared.push((name, value));
    }
    let ptx_len = r.u64()? as usize;
    Some((r.utf8(ptx_len)?, shared))
}

/// `a.len() : u64, a, b.len() : u64, b`.
fn encode_pair(a: &str, b: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for s in [a, b] {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    out
}

fn decode_pair(bytes: &[u8]) -> Option<(String, String)> {
    let mut r = Reader::new(bytes);
    let a_len = r.u64()? as usize;
    let a = r.utf8(a_len)?;
    let b_len = r.u64()? as usize;
    Some((a, r.utf8(b_len)?))
}

/// A cursor over a cache entry's bytes, since [`decode_one`] and
/// [`decode_pair`] both walk a run of length-prefixed fields.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, at: 0 }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let slice = self.bytes.get(self.at..self.at + n)?;
        self.at += n;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }

    fn utf8(&mut self, n: usize) -> Option<String> {
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_entry_round_trips() {
        let shared = vec![("delta_scan".to_string(), 42_000usize), ("rms_norm_q".to_string(), 0)];
        let (ptx, got) = decode_one(&encode_one("// ptx text\n", &shared)).unwrap();
        assert_eq!(ptx, "// ptx text\n");
        assert_eq!(got, shared);
    }

    #[test]
    fn a_pair_entry_round_trips() {
        let (a, b) = decode_pair(&encode_pair("aligned ptx", "general ptx")).unwrap();
        assert_eq!(a, "aligned ptx");
        assert_eq!(b, "general ptx");
    }

    #[test]
    fn truncated_bytes_decode_to_nothing_rather_than_panicking() {
        let full = encode_one("kernel body", &[("k".to_string(), 8)]);
        assert!(decode_one(&full[..full.len() - 1]).is_none());
    }

    #[test]
    fn the_hash_changes_with_every_field_that_reaches_codegen() {
        let base = Context::default();
        let mut wider_index = base.clone();
        wider_index.index_bitwidth = 64;
        let mut different_shape = base.clone();
        different_shape.shape_overrides.insert("TN".to_string(), 64);

        let hashes = [
            hash(&base, &["kernel k() {}"]),
            hash(&base, &["kernel k() { let x = 1 }"]),
            hash(&wider_index, &["kernel k() {}"]),
            hash(&different_shape, &["kernel k() {}"]),
        ];
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j], "{i} vs {j}");
            }
        }
    }

    #[test]
    fn shape_override_order_does_not_change_the_hash() {
        let mut a = Context::default();
        a.shape_overrides.insert("TN".to_string(), 64);
        a.shape_overrides.insert("TM".to_string(), 32);
        let mut b = Context::default();
        b.shape_overrides.insert("TM".to_string(), 32);
        b.shape_overrides.insert("TN".to_string(), 64);
        assert_eq!(hash(&a, &["kernel k() {}"]), hash(&b, &["kernel k() {}"]));
    }
}
