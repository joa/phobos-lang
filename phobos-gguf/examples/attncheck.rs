// Each kernel of the matmul attention path against a host reference:
//
//   cargo run --release -p phobos-gguf --features cuda --example attncheck
//
// `backend_check` compares the whole call, which says only that the answer is
// wrong. This checks the transpose, the scores, the softmax and the mix one at
// a time, which says which one.

use std::ffi::c_void;

use anyhow::{Context, Result, bail};
use cust::function::Function;
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

use phobos_kernels::abi::{self, KernelArg};

// Only four kernels of it are wanted here, so most of the backend is dead.
use phobos_gguf::backend::device;

/// One head's shape: queries, keys already cached, head dimension, and the
/// width of a cached position.
const ROWS: usize = 128;
const START: usize = 0;
const DIM: usize = 256;
const KV_HEADS: usize = 4;

fn compile(source: &str) -> Result<Module> {
    phobos_kernels::compile(source, &[], "compiling")
}

/// One launch of a kernel over tensor operands given as (pointer, extents).
///
/// # Safety
/// Every operand must outlive the launch and match the kernel's expected shape.
unsafe fn launch(
    stream: &Stream,
    func: &Function,
    operands: &[(u64, [i64; 2])],
    grid: (u32, u32, u32),
) -> Result<()> {
    let mut args: Vec<KernelArg> = Vec::new();
    for (ptr, dims) in operands {
        abi::push_tensor_descriptor(&mut args, *ptr, dims);
    }
    let mut slots: Vec<u64> = args.iter().map(|a| a.slot()).collect();
    let raw: Vec<*mut c_void> = slots
        .iter_mut()
        .map(|s| s as *mut u64 as *mut c_void)
        .collect();
    unsafe {
        stream.launch(func, grid, (256, 1, 1), 0, &raw)?;
    }
    Ok(())
}

fn report(what: &str, got: &[f32], want: &[f32]) -> bool {
    let mut worst = 0.0f32;
    let mut at = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        if err > worst {
            (worst, at) = (err, i);
        }
    }
    let ok = worst < 1e-3;
    println!(
        "{} {what:<12} worst {worst:.3e} at {at}",
        if ok { "ok  " } else { "FAIL" }
    );
    ok
}

fn main() -> Result<()> {
    let _ctx = cust::quick_init().context("initializing CUDA")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let module = compile(&device::attn_gemm_src(DIM, device::ATTN_GEMM_TILE))?;

    let nk = START + ROWS;
    let kw = KV_HEADS * DIM;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / 8388608.0 - 1.0
    };

    // One query head against key head 1, so the column offset is exercised.
    let head = 1;
    let q: Vec<f32> = (0..ROWS * DIM).map(|_| next()).collect();
    let keys: Vec<f32> = (0..nk * kw).map(|_| next()).collect();
    let values: Vec<f32> = (0..nk * kw).map(|_| next()).collect();

    let q_dev = DeviceBuffer::from_slice(&q)?;
    let k_dev = DeviceBuffer::from_slice(&keys)?;
    let kt_dev = DeviceBuffer::<f32>::zeroed(DIM * nk)?;
    let mut vh_dev = DeviceBuffer::<f32>::zeroed(nk * DIM)?;
    let s_dev = DeviceBuffer::<f32>::zeroed(ROWS * nk)?;
    let l_dev = DeviceBuffer::<f32>::zeroed(ROWS)?;
    let o_dev = DeviceBuffer::<f32>::zeroed(ROWS * DIM)?;
    let at = head * DIM;

    let mut all_ok = true;

    // The transpose: KT[d, j] is key j's element d of this head.
    let kt_fn = module.get_function("attn_kt")?;
    unsafe {
        launch(
            &stream,
            &kt_fn,
            &[
                (
                    k_dev.as_device_ptr().as_raw() + (at * 4) as u64,
                    [nk as i64, kw as i64],
                ),
                (kt_dev.as_device_ptr().as_raw(), [DIM as i64, nk as i64]),
            ],
            (nk.div_ceil(8) as u32, 1, 1),
        )?;
    }
    stream.synchronize()?;
    let mut got = vec![0.0f32; DIM * nk];
    kt_dev.copy_to(&mut got)?;
    let mut want = vec![0.0f32; DIM * nk];
    for j in 0..nk {
        for d in 0..DIM {
            want[d * nk + j] = keys[j * kw + at + d];
        }
    }
    all_ok &= report("attn_kt", &got, &want);

    // The scores, scaled but not masked.
    let scale = (DIM as f32).sqrt().recip();
    let score_fn = module.get_function("attn_scores")?;
    unsafe {
        launch(
            &stream,
            &score_fn,
            &[
                (q_dev.as_device_ptr().as_raw(), [ROWS as i64, DIM as i64]),
                (kt_dev.as_device_ptr().as_raw(), [DIM as i64, nk as i64]),
                (s_dev.as_device_ptr().as_raw(), [ROWS as i64, nk as i64]),
            ],
            (
                (ROWS / device::ATTN_GEMM_TILE) as u32,
                nk.div_ceil(device::ATTN_GEMM_TILE) as u32,
                1,
            ),
        )?;
    }
    stream.synchronize()?;
    let mut scores = vec![0.0f32; ROWS * nk];
    s_dev.copy_to(&mut scores)?;
    let mut want = vec![0.0f32; ROWS * nk];
    for i in 0..ROWS {
        for j in 0..nk {
            let dot: f32 = (0..DIM)
                .map(|d| q[i * DIM + d] * keys[j * kw + at + d])
                .sum();
            want[i * nk + j] = dot * scale;
        }
    }
    // Only the causal half is written; the rest is whatever was there.
    let keep = |v: &[f32]| -> Vec<f32> {
        let mut out = Vec::new();
        for i in 0..ROWS {
            let last = START + i;
            out.extend_from_slice(&v[i * nk..i * nk + last + 1]);
        }
        out
    };
    all_ok &= report("attn_scores", &keep(&scores), &keep(&want));

    // The softmax, in place, leaving the row sums for the mix.
    let soft_fn = module.get_function("attn_softmax")?;
    unsafe {
        launch(
            &stream,
            &soft_fn,
            &[
                (s_dev.as_device_ptr().as_raw(), [ROWS as i64, nk as i64]),
                (l_dev.as_device_ptr().as_raw(), [ROWS as i64, 1]),
            ],
            ((ROWS / device::ATTN_SOFT_TILE) as u32, 1, 1),
        )?;
    }
    stream.synchronize()?;
    let mut probs = vec![0.0f32; ROWS * nk];
    s_dev.copy_to(&mut probs)?;
    let mut sums = vec![0.0f32; ROWS];
    l_dev.copy_to(&mut sums)?;
    let mut want_p = vec![0.0f32; ROWS * nk];
    let mut want_l = vec![0.0f32; ROWS];
    for i in 0..ROWS {
        let last = START + i;
        let row = &want[i * nk..(i + 1) * nk];
        // The kernel's maximum takes in the masked entries of its own diagonal
        // tile, which only shifts the exponentials down, so the reference takes
        // the same range: everything up to the end of that tile.
        let seen = (i / device::ATTN_SOFT_TILE + 1) * device::ATTN_SOFT_TILE + START;
        let m = row[..seen.min(nk)].iter().copied().fold(f32::MIN, f32::max);
        for j in 0..=last {
            want_p[i * nk + j] = (row[j] - m).exp();
            want_l[i] += want_p[i * nk + j];
        }
    }
    all_ok &= report("attn_softmax", &keep(&probs), &keep(&want_p));
    all_ok &= report("attn_sums", &sums, &want_l);

    // The mix, normalizing by those sums.
    let mut want_v = vec![0.0f32; nk * DIM];
    for j in 0..nk {
        want_v[j * DIM..(j + 1) * DIM].copy_from_slice(&values[j * kw + at..j * kw + at + DIM]);
    }
    vh_dev.copy_from(&want_v)?;
    let mix_fn = module.get_function("attn_mix")?;
    unsafe {
        launch(
            &stream,
            &mix_fn,
            &[
                (s_dev.as_device_ptr().as_raw(), [ROWS as i64, nk as i64]),
                (vh_dev.as_device_ptr().as_raw(), [nk as i64, DIM as i64]),
                (l_dev.as_device_ptr().as_raw(), [ROWS as i64, 1]),
                (o_dev.as_device_ptr().as_raw(), [ROWS as i64, DIM as i64]),
            ],
            (
                (ROWS / device::ATTN_SOFT_TILE) as u32,
                (DIM / device::ATTN_GEMM_TILE) as u32,
                1,
            ),
        )?;
    }
    stream.synchronize()?;
    let mut out = vec![0.0f32; ROWS * DIM];
    o_dev.copy_to(&mut out)?;
    let mut want_o = vec![0.0f32; ROWS * DIM];
    for i in 0..ROWS {
        for d in 0..DIM {
            let mut acc = 0.0;
            for j in 0..=START + i {
                acc += want_p[i * nk + j] * want_v[j * DIM + d];
            }
            want_o[i * DIM + d] = acc / want_l[i];
        }
    }
    all_ok &= report("attn_mix", &out, &want_o);

    if !all_ok {
        bail!("a kernel disagrees with the host reference");
    }
    println!("\nevery kernel agrees");
    Ok(())
}
