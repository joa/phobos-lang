// The mixture-of-experts slot matvecs on the device, each K-quant against a
// host reference over the same int8 activations: one program a used expert,
// each reading the slot the table names.

use std::cell::Cell;
use std::ffi::c_void;

use anyhow::Result;
use cust::prelude::*;
use phobos_gguf::backend::device::{moe_gateup_src, moe_qdot_src};
use phobos_gguf::quant::grouped::group_rows;
use phobos_gguf::quant::{Packed, Quant, quantize_row};
use phobos_kernels::{compile, cuda_ok, push_descriptor};

use super::raw::random_raw_block;

/// Experts a token routes to, the kernels' `U`.
const USED: usize = 8;
/// Cache slots in the slab, and the table that picks one a program, with
/// repeats as a real routing has none of but a slab reuse would.
const SLOTS: usize = 5;
const TABLE: [i32; USED] = [3, 0, 4, 4, 1, 2, 0, 3];
/// Rows an expert holds and its input width.
const NE: usize = 128;
const K: usize = 512;

/// A slab of `SLOTS` experts in the device's grouped layout, and the same
/// weight decoded on the host, a row of `K` per output.
struct Slab {
    bytes: DeviceBuffer<i8>,
    d: DeviceBuffer<u16>,
    rb: usize,
    dense: Vec<f32>,
}

fn slab(quant: Quant, next: &mut dyn FnMut() -> f32) -> Result<Slab> {
    let (block_bytes, d_off, dmin_off) = match quant {
        Quant::Q4_K => (144, 0, Some(2)),
        Quant::Q5_K => (176, 0, Some(2)),
        _ => (210, 208, None),
    };
    let (rows, nb) = (SLOTS * NE, K / 256);
    let file: Vec<u8> = (0..rows * nb).flat_map(|_| random_raw_block(&mut *next, block_bytes, d_off, dmin_off)).collect();
    let packed = Packed::new(quant, file, K, rows)?;
    let dev = quant.device_block().1;
    let bytes = group_rows(&packed.device_blocks(), rows, nb, dev);
    let d = group_rows(&packed.raw_scales()?.d, rows, nb, 1);
    let mut dense = vec![0.0; rows * K];
    for (r, out) in dense.chunks_exact_mut(K).enumerate() {
        packed.row_into(r, out)?;
    }
    Ok(Slab { bytes: DeviceBuffer::from_slice(&bytes)?, d: DeviceBuffer::from_slice(&d)?, rb: nb * dev, dense })
}

/// Launches `name` from `src` on the `(NE / 64, USED)` grid the backend uses.
fn launch(src: &str, name: &str, slots: &mut [u64]) -> Result<()> {
    let module = compile(src, &[], "moe slot check")?;
    let func = module.get_function(name)?.to_raw();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let mut argv: Vec<*mut c_void> = slots.iter_mut().map(|s| s as *mut u64 as *mut _).collect();
    // SAFETY: argv points into slots, which outlives the launch, in the
    // exploded-memref ABI the kernel was compiled for.
    cuda_ok(
        unsafe {
            cust::sys::cuLaunchKernel(
                func,
                (NE / 64) as u32,
                USED as u32,
                1,
                256,
                1,
                1,
                0,
                stream.as_inner(),
                argv.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        name,
    )?;
    stream.synchronize()?;
    Ok(())
}

/// Output `c` of the expert program `j` reads, as a row of `K`.
fn row(dense: &[f32], j: usize, c: usize) -> &[f32] {
    &dense[(TABLE[j] as usize * NE + c) * K..][..K]
}

/// The largest difference over the reference's RMS.
fn relative(got: &[f32], want: &[f32]) -> f32 {
    let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max) / rms
}

fn report(failures: &Cell<u32>, label: &str, err: f32) {
    let ok = err < 1e-4;
    if !ok {
        failures.set(failures.get() + 1);
    }
    println!("{} {label:<32} worst relative error {err:.3e}", if ok { "ok  " } else { "FAIL" });
}

pub fn check_moe_slots(failures: &Cell<u32>, next: &mut dyn FnMut() -> f32) -> Result<()> {
    // Eight activation rows, quantized as the backend does, and their
    // dequantized values the reference reads.
    let x: Vec<f32> = (0..USED * K).map(|_| next()).collect();
    let mut qs = vec![0i8; USED * K];
    let mut scales = vec![0f32; USED * K / 32];
    for ((xb, qb), s) in x.chunks_exact(32).zip(qs.chunks_exact_mut(32)).zip(&mut scales) {
        *s = quantize_row(xb, qb);
    }
    let act: Vec<f32> = qs.iter().enumerate().map(|(i, &q)| f32::from(q) * scales[i / 32]).collect();
    let dot = |w: &[f32], a: &[f32]| w.iter().zip(a).map(|(w, a)| w * a).sum::<f32>();

    let aq = DeviceBuffer::from_slice(&qs)?;
    let asc = DeviceBuffer::from_slice(&scales)?;
    let table = DeviceBuffer::from_slice(&TABLE)?;
    for (quant, name) in [(Quant::Q4_K, "q4k"), (Quant::Q5_K, "q5k"), (Quant::Q6_K, "q6k")] {
        let (g, u) = (slab(quant, next)?, slab(quant, next)?);
        let ns = (SLOTS * NE) as i64;
        let operands = |slots: &mut Vec<u64>, s: &Slab| {
            push_descriptor(slots, s.bytes.as_device_ptr().as_raw(), [ns, s.rb as i64]);
            push_descriptor(slots, s.d.as_device_ptr().as_raw(), [ns, (K / 256) as i64]);
        };
        let head = |slots: &mut Vec<u64>, rows: usize| {
            push_descriptor(slots, aq.as_device_ptr().as_raw(), [rows as i64, K as i64]);
            push_descriptor(slots, asc.as_device_ptr().as_raw(), [rows as i64, (K / 32) as i64]);
            push_descriptor(slots, table.as_device_ptr().as_raw(), [1, USED as i64]);
        };

        // One shared activation row (gate and up), then a row a program (down).
        for (act_row, rows) in [("0", 1usize), ("j", USED)] {
            let out = DeviceBuffer::from_slice(&vec![0f32; USED * NE])?;
            let mut slots = Vec::new();
            head(&mut slots, rows);
            operands(&mut slots, &g);
            push_descriptor(&mut slots, out.as_device_ptr().as_raw(), [USED as i64, NE as i64]);
            launch(&moe_qdot_src(name, NE, act_row), &format!("{name}_moe_qdot"), &mut slots)?;
            let mut got = vec![0f32; USED * NE];
            out.copy_to(&mut got)?;
            let want: Vec<f32> = (0..USED * NE)
                .map(|i| {
                    let (j, c) = (i / NE, i % NE);
                    let ar = if rows == 1 { 0 } else { j };
                    dot(row(&g.dense, j, c), &act[ar * K..][..K])
                })
                .collect();
            report(failures, &format!("{name}_moe_qdot act_row {act_row}"), relative(&got, &want));
        }

        let out = DeviceBuffer::from_slice(&vec![0f32; USED * NE])?;
        let mut slots = Vec::new();
        head(&mut slots, 1);
        operands(&mut slots, &g);
        operands(&mut slots, &u);
        push_descriptor(&mut slots, out.as_device_ptr().as_raw(), [USED as i64, NE as i64]);
        launch(&moe_gateup_src(name, NE), &format!("{name}_moe_gateup"), &mut slots)?;
        let mut got = vec![0f32; USED * NE];
        out.copy_to(&mut got)?;
        let want: Vec<f32> = (0..USED * NE)
            .map(|i| {
                let (j, c) = (i / NE, i % NE);
                let gv = dot(row(&g.dense, j, c), &act[..K]);
                gv / (1.0 + (-gv).exp()) * dot(row(&u.dense, j, c), &act[..K])
            })
            .collect();
        report(failures, &format!("{name}_moe_gateup"), relative(&got, &want));
    }
    Ok(())
}
