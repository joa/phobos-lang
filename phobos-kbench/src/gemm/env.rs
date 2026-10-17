// What a GEMM result was measured on: card, driver, libraries, compiler and
// commit. Every JSON report carries it, and the winner cache keys on it.

use cust::device::{Device, DeviceAttribute};
use std::process::Command;

pub(crate) struct Env {
    pub json: serde_json::Value,
    /// The part of the environment a tuned winner depends on.
    pub key: String,
}

/// A command's trimmed stdout, or None when it is missing or fails.
fn output(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The card's live state from nvidia-smi: clocks, power, temperature,
/// utilization and memory in use, which is how contention shows.
pub(crate) fn gpu_state() -> serde_json::Value {
    const FIELDS: &str = "clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,pstate";
    let line = output(
        "nvidia-smi",
        &["-i", "0", &format!("--query-gpu={FIELDS}"), "--format=csv,noheader"],
    );
    match line {
        Some(line) => FIELDS
            .split(',')
            .zip(line.split(',').map(str::trim))
            .map(|(k, v)| (k.to_string(), serde_json::Value::from(v)))
            .collect::<serde_json::Map<_, _>>()
            .into(),
        None => serde_json::Value::Null,
    }
}

pub(crate) fn capture(cublas_version: i32) -> anyhow::Result<Env> {
    let dev = Device::get_device(0)?;
    let name = dev.name()?;
    let cc = dev.get_attribute(DeviceAttribute::ComputeCapabilityMajor)? * 10
        + dev.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?;
    let sms = dev.get_attribute(DeviceAttribute::MultiprocessorCount)?;
    let clock_khz = dev.get_attribute(DeviceAttribute::ClockRate)?;
    let cuda_driver = cust::CudaApiVersion::get()?;
    let driver = output("nvidia-smi", &["-i", "0", "--query-gpu=driver_version", "--format=csv,noheader"]);
    let commit = output("git", &["rev-parse", "HEAD"]);
    let dirty = output("git", &["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty());
    let llvm = output("llvm-config", &["--version"]);
    let fingerprint = phobos_kernels::COMPILER_FINGERPRINT;
    let cuda = format!("{}.{}", cuda_driver.major(), cuda_driver.minor());

    let json = serde_json::json!({
        "gpu": name,
        "sm": format!("sm_{cc}"),
        "sms": sms,
        "max_clock_mhz": clock_khz / 1000,
        "driver": driver,
        "cuda_driver_api": cuda,
        "cublas": cublas_version,
        "llvm": llvm,
        "compiler_fingerprint": fingerprint,
        "commit": commit,
        "tracked_changes": dirty,
        "os": format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        "args": std::env::args().skip(1).collect::<Vec<_>>(),
        "gpu_state_at_start": gpu_state(),
    });
    let key = format!(
        "{name}|sm_{cc}|driver {}|compiler {fingerprint}",
        driver.as_deref().unwrap_or(&cuda)
    );
    Ok(Env { json, key })
}
