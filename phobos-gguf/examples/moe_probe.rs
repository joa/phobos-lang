// The expert-streaming probes, standalone on the card:
//
//   cargo run --release --features cuda -p phobos-gguf --example moe_probe -- [MIRROR_GIB]
//
// In order: whether the driver maps host memory into the device's address
// space at all; whether it grants a pinned, device-mapped allocation the
// size of the model's experts (20 GiB by default, the argument overrides);
// how fast the K-quant decode matvec reads weights straight out of that
// mapping over PCIe against the same kernel on device memory; how fast the
// copy engine moves expert-sized chunks from pinned and from pageable host
// memory with one, four and sixteen in flight; and whether copies on a
// second stream overlap kernels on the first, or serialize behind them.
//
// Every number is printed as measured, once, on this card and driver; none
// is a spec-sheet figure. Nothing here touches a model file.

use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use cust::event::{Event, EventFlags};
use cust::prelude::*;
use cust::sys;
use phobos_kernels::{compile, cuda_ok, push_descriptor};

/// One expert's gate and up matrices, Q4_K, `[512, 2048]` each.
const EXPERT_ROWS: usize = 2 * 512;
const K: usize = 2048;
const NB: usize = K / 256;
const RB: usize = NB * 144;
/// Experts a decode block reads.
const USED: usize = 8;
/// Bytes of one expert's three matrices, the copy engine's chunk.
const EXPERT_BYTES: usize = 1_900_544;

const Q4K_SRC: &str = "@launch(256, 4)
@autotune(TN in [64])
@aligned(N = TN)
kernel q4k_qdot_i8_matvec(AQ: tensor<i8>[M, K], AS: tensor<f32>[M, KB],
                          QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                          C: tensor<f32>[M, N]) {
  let pn = program_id(0)
  C[0 :+ 1, pn * TN :+ TN] = q4k_qdot_i8_t(AQ[0 :+ 1, :], AS[0 :+ 1, :],
                                           QB[pn * TN :+ TN, :], D[pn * TN :+ TN, :])
}
";

fn main() -> Result<()> {
    let mirror_gib: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(20);
    let _ctx = cust::quick_init()?;
    let device = cust::device::Device::get_device(0)?;
    println!("{}", device.name()?);
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let copy_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    probe_attributes(&device)?;
    let mapped = probe_mirror(mirror_gib)?;
    probe_zero_copy(&stream, mapped.as_ref())?;
    probe_dma(&stream, &copy_stream)?;
    probe_overlap(&stream, &copy_stream)?;
    Ok(())
}

fn attribute(device: &cust::device::Device, attr: sys::CUdevice_attribute) -> Result<i32> {
    let mut value = 0i32;
    cuda_ok(unsafe { sys::cuDeviceGetAttribute(&mut value, attr, device.as_raw()) }, "device attribute")?;
    Ok(value)
}

fn probe_attributes(device: &cust::device::Device) -> Result<()> {
    let unified = attribute(device, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)?;
    let can_map = attribute(device, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY)?;
    let mut flags = 0u32;
    cuda_ok(unsafe { sys::cuCtxGetFlags(&mut flags) }, "context flags")?;
    println!(
        "unified addressing {unified}, can map host memory {can_map}, context flags {flags:#x} (map-host bit {})",
        flags & sys::CUctx_flags::CU_CTX_MAP_HOST as u32 != 0
    );
    Ok(())
}

/// A pinned, device-mapped host allocation: the host pointer, the device
/// pointer that reaches it, and its size.
struct Mapped {
    host: *mut c_void,
    device: sys::CUdeviceptr,
    bytes: usize,
}

fn host_alloc_mapped(bytes: usize) -> Result<Mapped> {
    let mut host: *mut c_void = std::ptr::null_mut();
    cuda_ok(
        unsafe { sys::cuMemHostAlloc(&mut host, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) },
        "cuMemHostAlloc(DEVICEMAP)",
    )?;
    let mut device: sys::CUdeviceptr = 0;
    cuda_ok(unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, host, 0) }, "device pointer of pinned memory")?;
    Ok(Mapped { host, device, bytes })
}

fn probe_mirror(gib: usize) -> Result<Option<Mapped>> {
    let bytes = gib << 30;
    let started = Instant::now();
    match host_alloc_mapped(bytes) {
        Ok(mapped) => {
            let alloc = started.elapsed();
            // Touch every page, as building the mirror would.
            let started = Instant::now();
            let slice = unsafe { std::slice::from_raw_parts_mut(mapped.host.cast::<u8>(), bytes) };
            for page in slice.chunks_mut(4096) {
                page[0] = 1;
            }
            println!(
                "pinned mirror: cuMemHostAlloc(DEVICEMAP) of {gib} GiB ok in {:.2} s, device pointer {:#x}, pages touched in {:.2} s",
                alloc.as_secs_f64(),
                mapped.device,
                started.elapsed().as_secs_f64()
            );
            // Given back: the zero-copy probe wants a small mapping of its
            // own, and holding 20 GiB pinned for the rest of the run says
            // nothing more.
            cuda_ok(unsafe { sys::cuMemFreeHost(mapped.host) }, "free pinned")?;
        }
        Err(e) => {
            println!("pinned mirror: cuMemHostAlloc(DEVICEMAP) of {gib} GiB REFUSED after {:.2} s: {e}", started.elapsed().as_secs_f64());
            let started = Instant::now();
            let mut plain = vec![0u8; bytes];
            match cuda_ok(
                unsafe { sys::cuMemHostRegister_v2(plain.as_mut_ptr().cast(), bytes, sys::CU_MEMHOSTREGISTER_DEVICEMAP) },
                "cuMemHostRegister(DEVICEMAP)",
            ) {
                Ok(()) => {
                    println!("pinned mirror: cuMemHostRegister(DEVICEMAP) of {gib} GiB ok in {:.2} s", started.elapsed().as_secs_f64());
                    cuda_ok(unsafe { sys::cuMemHostUnregister(plain.as_mut_ptr().cast()) }, "unregister")?;
                }
                Err(e) => println!("pinned mirror: cuMemHostRegister(DEVICEMAP) of {gib} GiB REFUSED after {:.2} s: {e}", started.elapsed().as_secs_f64()),
            }
        }
    }
    // The small mapping the zero-copy probe reads: eight experts' gate and up.
    let small = USED * EXPERT_ROWS * RB;
    Ok(host_alloc_mapped(small).map_err(|e| println!("small mapping refused: {e}")).ok())
}

/// The decode matvec over eight experts' worth of rows, from `qb` (a device
/// pointer, wherever it points), with its operands built once so that
/// timing it queues nothing but launches.
struct Matvec {
    function: sys::CUfunction,
    n: usize,
    slots: Vec<u64>,
    _keep: [DeviceBuffer<u8>; 4],
}

impl Matvec {
    fn new(function: sys::CUfunction, qb: u64, n: usize) -> Result<Matvec> {
        let aq: Vec<u8> = (0..K).map(|i| ((i % 17) as i32 - 8) as u8).collect();
        let asc: Vec<u8> = vec![0.01f32; K / 32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let d: Vec<u8> = vec![0x3400u16; n * NB].iter().flat_map(|v| v.to_le_bytes()).collect();
        let c = vec![0u8; n * 4];
        let keep = [
            DeviceBuffer::from_slice(&aq)?,
            DeviceBuffer::from_slice(&asc)?,
            DeviceBuffer::from_slice(&d)?,
            DeviceBuffer::from_slice(&c)?,
        ];
        let mut slots = Vec::new();
        push_descriptor(&mut slots, keep[0].as_device_ptr().as_raw(), [1, K as i64]);
        push_descriptor(&mut slots, keep[1].as_device_ptr().as_raw(), [1, (K / 32) as i64]);
        push_descriptor(&mut slots, qb, [n as i64, RB as i64]);
        push_descriptor(&mut slots, keep[2].as_device_ptr().as_raw(), [n as i64, NB as i64]);
        push_descriptor(&mut slots, keep[3].as_device_ptr().as_raw(), [1, n as i64]);
        Ok(Matvec { function, n, slots, _keep: keep })
    }

    fn launch(&mut self, stream: &Stream) -> Result<()> {
        let mut params: Vec<*mut c_void> = self.slots.iter_mut().map(|s| (s as *mut u64).cast()).collect();
        cuda_ok(
            unsafe {
                sys::cuLaunchKernel(
                    self.function,
                    (self.n / 64) as u32,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    stream.as_inner(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "matvec launch",
        )
    }

    /// Milliseconds a launch over `reps` launches, by events on `stream`.
    fn time(&mut self, stream: &Stream, reps: usize) -> Result<f64> {
        self.launch(stream)?;
        stream.synchronize()?;
        let (begin, end) = (Event::new(EventFlags::DEFAULT)?, Event::new(EventFlags::DEFAULT)?);
        begin.record(stream)?;
        for _ in 0..reps {
            self.launch(stream)?;
        }
        end.record(stream)?;
        end.synchronize()?;
        Ok(f64::from(end.elapsed_time_f32(&begin)?) / reps as f64)
    }
}

fn time_matvec(stream: &Stream, function: sys::CUfunction, qb: u64, n: usize, reps: usize) -> Result<f64> {
    Matvec::new(function, qb, n)?.time(stream, reps)
}

fn probe_zero_copy(stream: &Stream, mapped: Option<&Mapped>) -> Result<()> {
    let n = USED * EXPERT_ROWS;
    let bytes: Vec<i8> = (0..n * RB).map(|i| (i.wrapping_mul(2654435761) >> 13) as i8).collect();
    let module = compile(Q4K_SRC, &[("TN", 64)], "q4k probe")?;
    let function = module.get_function("q4k_qdot_i8_matvec")?.to_raw();

    let resident = DeviceBuffer::from_slice(&bytes)?;
    let ms = time_matvec(stream, function, resident.as_device_ptr().as_raw(), n, 50)?;
    println!(
        "zero-copy: {} rows of {} bytes ({:.1} MB), {} CTAs; from device memory {ms:.3} ms, {:.1} GB/s",
        n,
        RB,
        (n * RB) as f64 / 1e6,
        n / 64,
        (n * RB) as f64 / (ms / 1e3) / 1e9
    );
    let Some(mapped) = mapped else {
        println!("zero-copy: no mapping to read from");
        return Ok(());
    };
    if mapped.bytes < n * RB {
        bail!("the mapping is smaller than the weight");
    }
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<u8>(), mapped.host.cast::<u8>(), n * RB) };
    let ms = time_matvec(stream, function, mapped.device, n, 20)?;
    println!(
        "zero-copy: the same kernel reading the pinned host mapping over PCIe {ms:.3} ms, {:.2} GB/s",
        (n * RB) as f64 / (ms / 1e3) / 1e9
    );
    Ok(())
}

/// `chunks` copies of one expert each from `src` (host) into `dst` (device),
/// spread round-robin over `streams`, as GB/s once all have landed.
fn dma_rate(streams: &[&Stream], src: *const c_void, dst: sys::CUdeviceptr, chunks: usize) -> Result<f64> {
    for s in streams {
        s.synchronize()?;
    }
    let started = Instant::now();
    for i in 0..chunks {
        let stream = streams[i % streams.len()];
        let at = (i * EXPERT_BYTES) as u64;
        cuda_ok(
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(dst + at, src.cast::<u8>().wrapping_add(i * EXPERT_BYTES).cast(), EXPERT_BYTES, stream.as_inner())
            },
            "async copy",
        )?;
    }
    for s in streams {
        s.synchronize()?;
    }
    Ok((chunks * EXPERT_BYTES) as f64 / started.elapsed().as_secs_f64() / 1e9)
}

fn probe_dma(stream: &Stream, copy_stream: &Stream) -> Result<()> {
    let chunks = 64;
    let bytes = chunks * EXPERT_BYTES;
    let dst = DeviceBuffer::from_slice(&vec![0u8; bytes])?;
    let pinned = host_alloc_mapped(bytes).context("pinned source for the DMA probe")?;
    let pageable = vec![1u8; bytes];
    let more: Vec<Stream> = (0..14).map(|_| Stream::new(StreamFlags::NON_BLOCKING, None)).collect::<Result<_, _>>()?;
    let mut all: Vec<&Stream> = vec![stream, copy_stream];
    all.extend(more.iter());
    // One large copy first: the link's peak with no per-copy overhead.
    for _ in 0..2 {
        stream.synchronize()?;
        let started = Instant::now();
        cuda_ok(
            unsafe { sys::cuMemcpyHtoDAsync_v2(dst.as_device_ptr().as_raw(), pinned.host, bytes, stream.as_inner()) },
            "large copy",
        )?;
        stream.synchronize()?;
        println!(
            "dma: one {:.0} MB copy from pinned memory {:.2} GB/s",
            bytes as f64 / 1e6,
            bytes as f64 / started.elapsed().as_secs_f64() / 1e9
        );
    }
    println!("dma: {chunks} chunks of {:.2} MB, pinned then pageable source, by streams in flight:", EXPERT_BYTES as f64 / 1e6);
    for in_flight in [1usize, 4, 16] {
        let streams = &all[..in_flight];
        let _ = dma_rate(streams, pinned.host, dst.as_device_ptr().as_raw(), chunks)?;
        let pinned_rate = dma_rate(streams, pinned.host, dst.as_device_ptr().as_raw(), chunks)?;
        let _ = dma_rate(streams, pageable.as_ptr().cast(), dst.as_device_ptr().as_raw(), chunks)?;
        let pageable_rate = dma_rate(streams, pageable.as_ptr().cast(), dst.as_device_ptr().as_raw(), chunks)?;
        println!("dma: {in_flight:>2} in flight: pinned {pinned_rate:>6.2} GB/s, pageable {pageable_rate:>6.2} GB/s");
    }
    cuda_ok(unsafe { sys::cuMemFreeHost(pinned.host) }, "free pinned")?;
    Ok(())
}

fn probe_overlap(stream: &Stream, copy_stream: &Stream) -> Result<()> {
    let n = USED * EXPERT_ROWS;
    let bytes: Vec<i8> = (0..n * RB).map(|i| (i.wrapping_mul(2654435761) >> 13) as i8).collect();
    let module = compile(Q4K_SRC, &[("TN", 64)], "q4k probe")?;
    let function = module.get_function("q4k_qdot_i8_matvec")?.to_raw();
    let resident = DeviceBuffer::from_slice(&bytes)?;
    let chunks = 64;
    let dst = DeviceBuffer::from_slice(&vec![0u8; chunks * EXPERT_BYTES])?;
    let pinned = host_alloc_mapped(chunks * EXPERT_BYTES).context("pinned source for the overlap probe")?;

    // Enough launches to run about as long as the copies do.
    let mut matvec = Matvec::new(function, resident.as_device_ptr().as_raw(), n)?;
    let ms_one = matvec.time(stream, 20)?;
    let copy_rate = dma_rate(&[copy_stream], pinned.host, dst.as_device_ptr().as_raw(), chunks)?;
    let copy_ms = (chunks * EXPERT_BYTES) as f64 / copy_rate / 1e6;
    let reps = ((copy_ms / ms_one).ceil() as usize).max(1);
    let kernels_ms = matvec.time(stream, reps)? * reps as f64;

    // Both at once: the copies queued on their stream, the kernels on
    // theirs, nothing else submitted, one wall clock over the pair.
    stream.synchronize()?;
    copy_stream.synchronize()?;
    let started = Instant::now();
    for i in 0..chunks {
        let at = (i * EXPERT_BYTES) as u64;
        cuda_ok(
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(dst.as_device_ptr().as_raw() + at, pinned.host.cast::<u8>().wrapping_add(i * EXPERT_BYTES).cast(), EXPERT_BYTES, copy_stream.as_inner())
            },
            "async copy",
        )?;
    }
    for _ in 0..reps {
        matvec.launch(stream)?;
    }
    stream.synchronize()?;
    copy_stream.synchronize()?;
    let both_ms = started.elapsed().as_secs_f64() * 1e3;
    println!(
        "overlap: {chunks} copies alone {copy_ms:.1} ms, {reps} kernels alone {kernels_ms:.1} ms, both together {both_ms:.1} ms (serial would be {:.1}, perfect overlap {:.1})",
        copy_ms + kernels_ms,
        copy_ms.max(kernels_ms)
    );
    cuda_ok(unsafe { sys::cuMemFreeHost(pinned.host) }, "free pinned")?;
    Ok(())
}
