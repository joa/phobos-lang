//! What a run asked the compiler for, recorded so the same kernels can be
//! compiled again later for any chip without a GPU or a model.
//!
//! `PHOBOS_KERNEL_MANIFEST` names a directory, and every compile request,
//! hit or miss, lands in it as one file named `<kernel>-<key>`. The key
//! covers everything a request carries except the chip, so several runs, and
//! several models, can share one directory and a kernel two of them ask for
//! is written once.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};
use sha2::{Digest, Sha256};

use crate::cache::{self, Reader};
use crate::lower;

const MAGIC: &[u8; 4] = b"phm1";

fn manifest_dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        std::env::var_os("PHOBOS_KERNEL_MANIFEST")
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from)
    })
    .as_deref()
}

/// One compile request: a single source, or the aligned and general texts of
/// a pair, with the context fields that reach codegen apart from the chip.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Request {
    pub name: String,
    pub index_bitwidth: u32,
    pub overrides: Vec<(String, i64)>,
    pub texts: Vec<String>,
}

/// Appends a request to the manifest directory when one is set. Best-effort
/// like the cache, but a failure is logged: a manifest with a hole in it
/// warms a cache that still compiles on first launch.
pub(crate) fn record(ctx: &Context, texts: &[&str]) {
    if let Some(dir) = manifest_dir() {
        record_in(dir, ctx, texts);
    }
}

fn record_in(dir: &Path, ctx: &Context, texts: &[&str]) {
    let request = Request::of(ctx, texts);
    let path = dir.join(request.file_name());
    if path.exists() {
        return;
    }
    if let Err(e) = cache::write_atomic(&path, &request.encode()) {
        phobos_base::phinfo!("kernel manifest: writing {}: {e}", path.display());
    }
}

/// Every request in a manifest directory with the file it came from, in
/// file-name order.
pub fn read(dir: &Path) -> Result<Vec<(PathBuf, Request)>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading the manifest {}", dir.display()))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && !path.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.'))
        })
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|path| {
            let request = read_one(&path)?;
            Ok((path, request))
        })
        .collect()
}

/// The request one manifest file holds.
pub fn read_one(path: &Path) -> Result<Request> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Request::decode(&bytes).with_context(|| format!("{} is not a manifest entry", path.display()))
}

impl Request {
    fn of(ctx: &Context, texts: &[&str]) -> Request {
        let mut overrides: Vec<_> = ctx
            .shape_overrides
            .iter()
            .map(|(name, value)| (name.clone(), *value))
            .collect();
        overrides.sort_unstable();
        Request {
            name: cache::kernel_name(texts[0]),
            index_bitwidth: ctx.index_bitwidth,
            overrides,
            texts: texts.iter().map(|t| t.to_string()).collect(),
        }
    }

    /// `<kernel>-<key>`, the key a hash of the encoded request.
    fn file_name(&self) -> String {
        let digest = Sha256::digest(self.encode());
        let key: String = digest[..16].iter().map(|b| format!("{b:02x}")).collect();
        format!("{}-{key}", self.name)
    }

    /// The request's context compiled for `chip`, at the default PTX version.
    pub fn context(&self, chip: &str) -> Context {
        Context {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
            shape_overrides: self.overrides.iter().cloned().collect(),
            index_bitwidth: self.index_bitwidth,
            ..Context::default()
        }
    }

    fn texts(&self) -> Vec<&str> {
        self.texts.iter().map(String::as_str).collect()
    }

    /// Where this request's entry for `chip` lives under `root`.
    pub fn entry(&self, root: &Path, chip: &str) -> PathBuf {
        cache::entry(root, &self.context(chip), &self.texts())
    }

    /// Whether `root` already holds this request's entry for `chip`.
    pub fn is_cached(&self, root: &Path, chip: &str) -> bool {
        self.entry(root, chip).is_file()
    }

    /// Lowers the request for `chip`, without storing it.
    pub fn compile(&self, chip: &str) -> Result<Compiled> {
        let ctx = self.context(chip);
        let texts = self.texts();
        let at = Instant::now();
        let (ptx, bytes) = match texts.as_slice() {
            [source] => {
                let ((ptx, shared), _) = lower::single(&ctx, source, &self.name)?;
                let bytes = cache::encode_one(&ptx, &shared);
                (vec![ptx], bytes)
            }
            [aligned, general] => {
                let (a, g) = lower::pair(&ctx, aligned, general, &self.name)?;
                let bytes = cache::encode_pair(&a, &g);
                (vec![a, g], bytes)
            }
            _ => bail!("{}: a request holds one text or two, not {}", self.name, texts.len()),
        };
        Ok(Compiled { ctx, texts: self.texts.clone(), ptx, bytes, took: at.elapsed() })
    }

    /// `magic, name, index_bitwidth : u32, overrides.len() : u32`, each
    /// `(name, value : i64)`, then `texts.len() : u32` and each text; strings
    /// are `len : u64, utf8`.
    fn encode(&self) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        let string = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        };
        string(&mut out, &self.name);
        out.extend(self.index_bitwidth.to_le_bytes());
        out.extend((self.overrides.len() as u32).to_le_bytes());
        for (name, value) in &self.overrides {
            string(&mut out, name);
            out.extend(value.to_le_bytes());
        }
        out.extend((self.texts.len() as u32).to_le_bytes());
        for text in &self.texts {
            string(&mut out, text);
        }
        out
    }

    fn decode(bytes: &[u8]) -> Option<Request> {
        let mut r = Reader::new(bytes);
        if r.take(MAGIC.len())? != MAGIC {
            return None;
        }
        let string = |r: &mut Reader| {
            let len = r.u64()? as usize;
            r.utf8(len)
        };
        let name = string(&mut r)?;
        let index_bitwidth = r.u32()?;
        let overrides = (0..r.u32()?)
            .map(|_| Some((string(&mut r)?, r.u64()? as i64)))
            .collect::<Option<Vec<_>>>()?;
        let texts = (0..r.u32()?)
            .map(|_| string(&mut r))
            .collect::<Option<Vec<_>>>()?;
        Some(Request { name, index_bitwidth, overrides, texts })
    }
}

/// A request lowered for one chip, ready to be checked and stored.
pub struct Compiled {
    ctx: Context,
    texts: Vec<String>,
    bytes: Vec<u8>,
    /// The PTX, one text per source: the aligned variant first for a pair.
    pub ptx: Vec<String>,
    pub took: Duration,
}

impl Compiled {
    /// Stores the entry under `root`, where a run on that chip reads it.
    pub fn store(&self, root: &Path) -> Result<()> {
        let texts: Vec<&str> = self.texts.iter().map(String::as_str).collect();
        let path = cache::entry(root, &self.ctx, &texts);
        cache::write_atomic(&path, &self.bytes).with_context(|| format!("writing {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(chip: &str) -> Context {
        let mut ctx = Context {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
            ..Context::default()
        };
        ctx.shape_overrides.insert("TN".to_string(), 64);
        ctx.shape_overrides.insert("TM".to_string(), -32);
        ctx
    }

    #[test]
    fn a_request_round_trips() {
        let request = Request::of(&ctx("sm_75"), &["kernel k() {}", "kernel k() { }"]);
        assert_eq!(Request::decode(&request.encode()), Some(request));
    }

    #[test]
    fn the_file_name_ignores_the_chip() {
        let texts = ["kernel k() {}"];
        let sm75 = Request::of(&ctx("sm_75"), &texts);
        let sm90 = Request::of(&ctx("sm_90"), &texts);
        assert_eq!(sm75.file_name(), sm90.file_name());
        assert!(sm75.file_name().starts_with("k-"));
    }

    #[test]
    fn the_file_name_moves_with_everything_else() {
        let base = Request::of(&ctx("sm_75"), &["kernel k() {}"]);
        let mut wider = base.clone();
        wider.index_bitwidth = 64;
        let mut shaped = base.clone();
        shaped.overrides[0].1 = 16;
        let mut pair = base.clone();
        pair.texts.push("kernel k() {}".to_string());
        for other in [&wider, &shaped, &pair] {
            assert_ne!(base.file_name(), other.file_name());
        }
    }

    #[test]
    fn a_context_comes_back_for_the_asked_chip() {
        let request = Request::of(&ctx("sm_75"), &["kernel k() {}"]);
        let back = request.context("sm_86");
        let GpuConfig::Nvidia(nv) = &back.gpu_config;
        assert_eq!(nv.chip(), "sm_86");
        assert_eq!(back.shape_overrides, ctx("sm_75").shape_overrides);
    }

    #[test]
    fn warming_writes_each_chips_entry_where_a_run_reads_it() {
        let root = std::env::temp_dir().join(format!("phobos-warm-{}", std::process::id()));
        let mut ctx = Context::default();
        for (name, value) in crate::matmul::shapes() {
            ctx.shape_overrides.insert(name.to_string(), value as i64);
        }
        let aligned = crate::matmul::TEMPLATE.replace("{ALIGNED}", "@aligned(M = TILE_M, N = TILE_N, K = TILE_K)");
        let general = crate::matmul::TEMPLATE.replace("{ALIGNED}\n", "");
        let single = Request::of(&ctx, &[&general]);
        let pair = Request::of(&ctx, &[&aligned, &general]);

        for chip in phobos_base::context::SUPPORTED_CHIPS {
            for request in [&single, &pair] {
                assert!(!request.is_cached(&root, chip));
                request.compile(chip).unwrap().store(&root).unwrap();
                assert!(request.is_cached(&root, chip));
            }
            let entry = cache::entry(&root, &single.context(chip), &single.texts());
            assert_eq!(entry.parent().unwrap().file_name().unwrap(), chip);
            let bytes = std::fs::read(entry).unwrap();
            let (ptx, _) = cache::decode_one(&bytes).unwrap();
            assert!(ptx.contains(&format!(".target {chip}")), "{chip}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn what_is_recorded_reads_back_once() {
        let dir = std::env::temp_dir().join(format!("phobos-manifest-{}", std::process::id()));
        let (single, pair) = (["kernel a() {}"], ["kernel b() {}", "kernel b() { }"]);
        for texts in [&single[..], &pair[..], &single[..]] {
            record_in(&dir, &ctx("sm_75"), texts);
        }
        record_in(&dir, &ctx("sm_86"), &single);
        let got: Vec<_> = read(&dir).unwrap().into_iter().map(|(_, request)| request).collect();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got, [Request::of(&ctx("sm_75"), &single), Request::of(&ctx("sm_75"), &pair)]);
    }

    #[test]
    fn a_foreign_file_decodes_to_nothing() {
        assert_eq!(Request::decode(b"not a manifest"), None);
        let full = Request::of(&ctx("sm_75"), &["kernel k() {}"]).encode();
        assert_eq!(Request::decode(&full[..full.len() - 1]), None);
    }
}
