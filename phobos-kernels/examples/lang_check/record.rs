// With `--record DIR`, every probe launch is written to DIR as it ran: the
// source, the PTX and its launch contracts, the slots as this launcher
// packed them, and each tensor's bytes before and after. Another launcher
// replays it on the same PTX and inputs and compares the bytes, as
// `python/tests/gpu/replay.py` does.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context as _, Result};
use phobos_base::context::Context;
use phobos_lang::contract::{Part, Role};

static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
static NEXT: AtomicUsize = AtomicUsize::new(0);

/// Reads `--record DIR` from the command line, once.
pub(crate) fn init() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut dir = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--record" => dir = Some(PathBuf::from(args.next().context("--record needs a directory")?)),
            other => anyhow::bail!("unknown argument {other}; the only one is --record DIR"),
        }
    }
    if let Some(dir) = &dir {
        std::fs::create_dir_all(dir)?;
    }
    DIR.set(dir).ok();
    Ok(())
}

/// A launch being recorded: where, and each tensor's address and size.
pub(crate) struct Recording {
    at: PathBuf,
    tensors: Vec<(String, u64, usize)>,
}

fn elem_bytes(elem: &str) -> usize {
    match elem {
        "i8" | "u8" | "bool" => 1,
        "f16" | "bf16" | "u16" => 2,
        "f64" | "i64" => 8,
        _ => 4,
    }
}

fn download(ptr: u64, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; bytes];
    // SAFETY: `ptr` is a live device allocation of at least `bytes` bytes,
    // which the launch's own tensor descriptor names.
    phobos_kernels::cuda_ok(
        unsafe { cust::sys::cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) },
        "reading a recorded tensor",
    )?;
    Ok(out)
}

/// Records the launch about to run, when recording.
pub(crate) fn before(ctx: &Context, src: &str, name: &str, grid: [u32; 3], threads: u32, slots: &[u64]) -> Result<Option<Recording>> {
    let Some(Some(dir)) = DIR.get() else { return Ok(None) };
    let at = dir.join(format!("{:03}_{name}", NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&at)?;
    let file = phobos_lang::diag::SourceFile::new("", src);
    let contracts = phobos_lang::contracts(ctx, &file)?;
    let ptx = match phobos_kernels::cache::load(ctx, src) {
        Some((ptx, _)) => ptx,
        None => phobos_lang::compile_shared(ctx, src)?.0,
    };
    let contract = contracts.iter().find(|c| c.kernel == name).context("no contract for the launched kernel")?;
    let mut tensors = Vec::new();
    for (i, p) in contract.params.iter().enumerate() {
        if p.shape.is_none() || p.role != Role::Declared {
            continue;
        }
        let sizes = contract.slots.iter().zip(slots).filter(|(s, _)| s.param == i && matches!(s.part, Part::Size(_)));
        let count: usize = sizes.map(|(s, &v)| (v & if s.bytes == 4 { 0xffff_ffff } else { u64::MAX }) as usize).product();
        tensors.push((p.name.clone(), slots[p.slot], count * elem_bytes(&p.elem)));
    }
    let descriptor = serde_json::json!({
        "compiler_fingerprint": phobos_kernels::COMPILER_FINGERPRINT,
        "cache_key": phobos_kernels::cache::key(ctx, src),
        "kernels": contracts.iter().map(|c| c.to_json()).collect::<Vec<_>>(),
    });
    let launch = serde_json::json!({
        "kernel": name,
        "grid": grid,
        "block": [threads, 1, 1],
        "slots": slots,
        "tensors": tensors.iter().map(|(n, _, b)| serde_json::json!({"name": n, "bytes": b})).collect::<Vec<_>>(),
    });
    std::fs::write(at.join("kernel.ph"), src)?;
    std::fs::write(at.join("kernel.ptx"), ptx)?;
    std::fs::write(at.join("kernel.json"), serde_json::to_string_pretty(&descriptor)?)?;
    std::fs::write(at.join("launch.json"), serde_json::to_string_pretty(&launch)?)?;
    for (name, ptr, bytes) in &tensors {
        std::fs::write(at.join(format!("{name}.in")), download(*ptr, *bytes)?)?;
    }
    Ok(Some(Recording { at, tensors }))
}

impl Recording {
    /// Records each tensor as the launch left it.
    pub(crate) fn after(self) -> Result<()> {
        for (name, ptr, bytes) in &self.tensors {
            std::fs::write(self.at.join(format!("{name}.out")), download(*ptr, *bytes)?)?;
        }
        Ok(())
    }
}
