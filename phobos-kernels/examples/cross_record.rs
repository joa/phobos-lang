// Launches each kernel of `examples/` once from Rust, as the runtime packs a
// launch (the memref ABI at the index width `needs_wide_index` picks, the
// CTA size `@launch` gives), on deterministic inputs, and records each
// launch as `lang_check --record` does, for `python/tests/gpu/replay.py` to
// replay on the same PTX and bytes:
//
//   cargo run --release -p phobos-kernels --features cuda --example cross_record -- --record DIR

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context as _, Result};
use cust::memory::DeviceBuffer;
use cust::prelude::*;
use phobos_kernels::compile::context_for;
use phobos_kernels::{compile_in, cuda_ok};
use phobos_lang::ast::{Dim, Type};

#[path = "lang_check/record.rs"]
mod record;

/// A symbol's value: a run-time extent or an `@autotune` constant.
type Lookup<'a> = &'a dyn Fn(&str) -> i64;

/// One launch: the source, its kernel, the run-time extents its symbols
/// take, and its grid from those and the `@autotune` constants.
struct Job {
    file: &'static str,
    kernel: &'static str,
    dims: &'static [(&'static str, i64)],
    grid: fn(Lookup<'_>) -> [u32; 3],
}

const SQUARE: &[(&str, i64)] = &[("M", 256), ("N", 256), ("K", 256)];
const ATTENTION: &[(&str, i64)] = &[("Nq", 256), ("Nk", 256)];

fn tiles(v: Lookup<'_>) -> [u32; 3] {
    [(v("M") / v("TILE_M")) as u32, (v("N") / v("TILE_N")) as u32, 1]
}

fn rows(v: Lookup<'_>) -> [u32; 3] {
    [(v("Nq") / v("BR")) as u32, 1, 1]
}

const JOBS: &[Job] = &[
    Job { file: "saxpy_fp32.ph", kernel: "saxpy", dims: &[("N", 4096)], grid: |v| [(v("N") / v("BLOCK")) as u32, 1, 1] },
    Job { file: "gemm_fp32.ph", kernel: "gemm", dims: SQUARE, grid: tiles },
    Job { file: "gemm_fp32_unaligned.ph", kernel: "gemm", dims: &[("M", 200), ("N", 136), ("K", 100)], grid: |v| {
        [(v("M") as u64).div_ceil(v("TILE_M") as u64) as u32, (v("N") as u64).div_ceil(v("TILE_N") as u64) as u32, 1]
    } },
    Job { file: "gemm_fp16.ph", kernel: "gemm", dims: SQUARE, grid: tiles },
    Job { file: "gemm_fp16_fp32acc.ph", kernel: "gemm", dims: SQUARE, grid: tiles },
    Job { file: "gemm_fp16tc_fp32acc.ph", kernel: "gemm", dims: SQUARE, grid: tiles },
    Job { file: "gemm_fp32_splitk.ph", kernel: "gemm_split", dims: &[("M", 256), ("N", 256), ("K", 256), ("SM", 512)], grid: |v| {
        [(v("M") / v("TILE_M")) as u32, (v("N") / v("TILE_N")) as u32, 2]
    } },
    Job { file: "gemm_fp32_splitk.ph", kernel: "gemm_reduce", dims: &[("M", 256), ("N", 256), ("SM", 512)], grid: |v| {
        [(v("M") / v("RM")) as u32, (v("N") / v("RN")) as u32, 1]
    } },
    Job { file: "matmul_cluster_fp32.ph", kernel: "matmul", dims: SQUARE, grid: tiles },
    Job { file: "flash_attention_fp32.ph", kernel: "flash_attention", dims: ATTENTION, grid: rows },
    Job { file: "flash_attention_fp16.ph", kernel: "flash_attention", dims: ATTENTION, grid: rows },
    Job { file: "moba_efficient.ph", kernel: "moba_attention", dims: ATTENTION, grid: rows },
    Job { file: "kda_fp32.ph", kernel: "kda", dims: &[("N", 256)], grid: |_| [1, 1, 1] },
];

/// Deterministic values in [-1, 1).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn main() -> Result<()> {
    record::init()?;
    let _ctx = cust::quick_init()?;
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples");
    for (j, job) in JOBS.iter().enumerate() {
        let src = std::fs::read_to_string(format!("{root}/{}", job.file))?.replace('\r', "");
        let mut ctx = context_for(&[]);
        let kernels = phobos_lang::parse(&src)?;
        if phobos_lang::needs_wide_index(&ctx, &kernels) {
            ctx.index_bitwidth = 64;
        }
        let kernel = kernels.iter().find(|k| k.name == job.kernel).context("no such kernel")?;
        let env: BTreeMap<String, i64> = kernel.shape_env(&HashMap::new());
        let dims: HashMap<&str, i64> = job.dims.iter().copied().collect();
        let value = |name: &str| dims.get(name).copied().or_else(|| env.get(name).copied()).unwrap_or_else(|| panic!("{}: no value for {name}", job.file));
        let threads = kernel.cta_threads().map_err(anyhow::Error::msg)? as u32;

        let mut buffers: Vec<DeviceBuffer<u8>> = Vec::new();
        let mut slots: Vec<u64> = Vec::new();
        let wide = ctx.index_bitwidth == 64;
        let index = |v: i64| if wide { v as u64 } else { v as i32 as u32 as u64 };
        for (p, param) in kernel.params.iter().enumerate() {
            match &param.ty {
                Type::Tensor(elem, shape) => {
                    let extents: Vec<i64> = shape
                        .iter()
                        .map(|d| match d {
                            Dim::Int(n) => *n,
                            Dim::Sym(s) => value(s),
                        })
                        .collect();
                    let count: i64 = extents.iter().product();
                    let values = noise(count as usize, (j * 31 + p) as u64 + 1);
                    let bytes: Vec<u8> = match elem {
                        phobos_lang::ast::Scalar::F16 => {
                            values.iter().flat_map(|&v| phobos_base::half::f32_to_f16(v).to_le_bytes()).collect()
                        }
                        phobos_lang::ast::Scalar::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        other => anyhow::bail!("{}: no inputs for a {other:?} tensor", job.file),
                    };
                    let buf = DeviceBuffer::from_slice(&bytes)?;
                    let ptr = buf.as_device_ptr().as_raw();
                    slots.extend([ptr, ptr, index(0)]);
                    slots.extend(extents.iter().map(|&e| index(e)));
                    slots.extend(phobos_kernels::abi::row_major_strides(&extents).into_iter().map(index));
                    buffers.push(buf);
                }
                Type::Scalar(phobos_lang::ast::Scalar::F32) => {
                    let v = match param.name.as_str() {
                        "alpha" => 1.5f32,
                        "beta" => 0.5,
                        "scale" => 0.125,
                        "inv_chunk" => 1.0 / 16.0,
                        _ => 1.0,
                    };
                    slots.push(u64::from(v.to_bits()));
                }
                other => anyhow::bail!("{}: no value for a {other:?} parameter", job.file),
            }
        }

        let grid = (job.grid)(&value);
        let module = compile_in(&ctx, &src, job.kernel)?.0;
        let func = module.get_function(job.kernel)?.to_raw();
        let recording = record::before(&ctx, &src, job.kernel, grid, threads, &slots)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let mut argv: Vec<*mut std::ffi::c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut std::ffi::c_void).collect();
        // SAFETY: argv points into slots, which outlives the launch, in the
        // memref ABI the kernel was compiled for; every buffer is alive.
        cuda_ok(
            unsafe {
                cust::sys::cuLaunchKernel(
                    func,
                    grid[0],
                    grid[1],
                    grid[2],
                    threads,
                    1,
                    1,
                    0,
                    stream.as_inner(),
                    argv.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            job.kernel,
        )?;
        stream.synchronize()?;
        if let Some(recording) = recording {
            recording.after()?;
        }
        println!("{} {} grid {grid:?} block {threads}", job.file, job.kernel);
    }
    Ok(())
}
