// The MoE router's top-k on the device against the host's `route`.

use anyhow::Result;
use cust::prelude::*;
use phobos_gguf::backend::{device::moe_topk_src, route};
use phobos_kernels::{compile, cuda_ok, push_descriptor};

/// Experts per router row and picks per row, as the 35B-A3B routes.
const EXPERTS: usize = 256;
const USED: usize = 8;

/// Router rows: random logits, one with planted ties across the top, and
/// one where every logit is equal.
fn rows() -> Vec<Vec<f32>> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
    };
    let random: Vec<f32> = (0..EXPERTS).map(|_| next()).collect();
    let mut tied: Vec<f32> = (0..EXPERTS).map(|_| next()).collect();
    for k in [3, 40, 41, 200, 255] {
        tied[k] = 9.0;
    }
    vec![random, tied, vec![0.5; EXPERTS]]
}

/// Whether the device's picks are the host's, in order, with the same
/// weights.
pub fn check_topk(failures: &std::cell::Cell<u32>) -> Result<()> {
    let rows = rows();
    let flat: Vec<f32> = rows.concat();
    let iota: Vec<f32> = (0..EXPERTS).map(|k| k as f32).collect();
    let logits = DeviceBuffer::from_slice(&flat)?;
    let io = DeviceBuffer::from_slice(&iota)?;
    let topk = DeviceBuffer::from_slice(&vec![0i32; rows.len() * USED])?;
    let w = DeviceBuffer::from_slice(&vec![0f32; rows.len() * USED])?;
    let n = rows.len() as i64;
    let mut slots = Vec::new();
    push_descriptor(&mut slots, logits.as_device_ptr().as_raw(), [n, EXPERTS as i64])?;
    push_descriptor(&mut slots, io.as_device_ptr().as_raw(), [1, EXPERTS as i64])?;
    push_descriptor(&mut slots, topk.as_device_ptr().as_raw(), [n, USED as i64])?;
    push_descriptor(&mut slots, w.as_device_ptr().as_raw(), [n, USED as i64])?;

    let module = compile(&moe_topk_src(), &[], "moe_topk")?;
    let func = module.get_function("moe_topk")?.to_raw();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let mut argv: Vec<*mut std::ffi::c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut _).collect();
    // SAFETY: argv points into slots, which outlives the launch, in the
    // exploded-memref ABI the kernel was compiled for.
    cuda_ok(
        unsafe {
            cust::sys::cuLaunchKernel(
                func,
                rows.len() as u32,
                1,
                1,
                32,
                1,
                1,
                0,
                stream.as_inner(),
                argv.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        "moe_topk",
    )?;
    stream.synchronize()?;
    let (mut ids, mut ws) = (vec![0i32; rows.len() * USED], vec![0f32; rows.len() * USED]);
    topk.copy_to(&mut ids)?;
    w.copy_to(&mut ws)?;

    for (r, logits) in rows.iter().enumerate() {
        let want = route(logits, USED);
        let got_ids = &ids[r * USED..(r + 1) * USED];
        let same = want.iter().zip(got_ids).all(|(&(k, _), &g)| k as i32 == g);
        // `total_cmp` ranks NaN above every number: a NaN error is the worst one,
        // where `f32::max` would drop it.
        let err = want
            .iter()
            .zip(&ws[r * USED..(r + 1) * USED])
            .map(|(&(_, p), &g)| (p - g).abs())
            .max_by(f32::total_cmp)
            .unwrap_or(0.0);
        let ok = same && err < 1e-5;
        if !ok {
            failures.set(failures.get() + 1);
        }
        println!(
            "{} moe_topk row {r}                 ids {} weight err {err:.3e}",
            if ok { "ok  " } else { "FAIL" },
            if same { "match".to_string() } else { format!("{got_ids:?} want {:?}", want.iter().map(|p| p.0).collect::<Vec<_>>()) },
        );
    }
    Ok(())
}
