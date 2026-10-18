// One problem's device buffers, and the launches the GEMM benchmark times.
// Nothing here synchronizes the stream: the caller decides what a timing
// covers.

use super::contract::{Baseline, Elem, Kind};
use super::verify::Inputs;
use crate::cublas::{self, CuBlas};
use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;

/// Rounds f32 to f16 elementwise, for the cuBLAS side of a kind that reads
/// f32 and multiplies in f16.
const CODE_CONVERT: &str = "\
@autotune(BLOCK in [4096, 4096])
kernel convert(x: tensor<f32>[N], y: tensor<f16>[N]) {
  let base = ctx.block.x * BLOCK
  y[base :+ BLOCK] = x[base :+ BLOCK]
}
";

/// The f32 to f16 conversion kernel, compiled once.
pub(crate) struct Converter {
    module: Module,
    block: u32,
    tile: u32,
    wide: bool,
}

impl Converter {
    pub fn new() -> anyhow::Result<Self> {
        let kernels = phobos_lang::parse(CODE_CONVERT)?;
        let block = kernels[0].cta_threads().map_err(anyhow::Error::msg)? as u32;
        let ptx = crate::autotune::compile(CODE_CONVERT, &[("BLOCK".into(), 4096)])?;
        Ok(Converter {
            module: Module::from_ptx(ptx.as_str(), &[])?,
            block,
            tile: 4096,
            wide: crate::autotune::wide_index(CODE_CONVERT),
        })
    }

    fn launch(&self, stream: &Stream, x: u64, y: u64, n: usize) -> anyhow::Result<()> {
        let func = self.module.get_function("convert")?;
        let (grid, n) = ((n as u32).div_ceil(self.tile), n as i32);
        let block = self.block;
        let n64 = n as i64;
        unsafe {
            if self.wide {
                launch!(func<<<grid, block, 0, stream>>>(
                    x, x, 0i64, n64, 1i64,
                    y, y, 0i64, n64, 1i64
                ))?;
            } else {
                launch!(func<<<grid, block, 0, stream>>>(
                    x, x, 0i32, n, 1i32,
                    y, y, 0i32, n, 1i32
                ))?;
            }
        }
        Ok(())
    }
}

/// A [m, k], B [k, n] and C [m, n] in the kind's storage types, a pristine C
/// every run restores from, one output C per implementation, and f16 copies
/// of A and B when the baseline needs them converted.
pub(crate) struct Problem {
    pub kind: Kind,
    pub dims: (usize, usize, usize),
    a: DeviceBuffer<u8>,
    b: DeviceBuffer<u8>,
    c_init: DeviceBuffer<u8>,
    pub c_phobos: DeviceBuffer<u8>,
    pub c_blas: DeviceBuffer<u8>,
    half: Option<(DeviceBuffer<u8>, DeviceBuffer<u8>)>,
    /// A split-K kind's partials, `[S * M, N]` f32 for up to `max_split`.
    work: Option<DeviceBuffer<u8>>,
    pub max_split: u32,
}

/// The most bytes a split-K workspace takes; larger splits are skipped.
const WORK_BYTES: usize = 256 << 20;

fn to_bytes(xs: &[f32], elem: Elem) -> Vec<u8> {
    match elem {
        Elem::F32 => xs.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Elem::F16 => xs
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
            .collect(),
    }
}

fn from_bytes(bytes: &[u8], elem: Elem) -> Vec<f32> {
    match elem {
        Elem::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect(),
        Elem::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
    }
}

impl Problem {
    pub fn new(kind: Kind, dims: (usize, usize, usize)) -> anyhow::Result<Self> {
        let (m, n, k) = dims;
        let zeroed = |elems: usize, elem: Elem| DeviceBuffer::<u8>::zeroed(elems * elem.bytes());
        let half = if kind.baseline_converts() {
            Some((zeroed(m * k, Elem::F16)?, zeroed(k * n, Elem::F16)?))
        } else {
            None
        };
        let split_k = kind.splits().len() > 1;
        let max_split = if split_k { (WORK_BYTES / (m * n * 4)).clamp(1, 16) as u32 } else { 1 };
        let work = if split_k {
            Some(DeviceBuffer::<u8>::zeroed(max_split as usize * m * n * 4)?)
        } else {
            None
        };
        Ok(Problem {
            kind,
            dims,
            work,
            max_split,
            a: zeroed(m * k, kind.ab())?,
            b: zeroed(k * n, kind.ab())?,
            c_init: zeroed(m * n, kind.c())?,
            c_phobos: zeroed(m * n, kind.c())?,
            c_blas: zeroed(m * n, kind.c())?,
            half,
        })
    }

    pub fn upload(&mut self, inp: &Inputs) -> anyhow::Result<()> {
        self.a.copy_from(to_bytes(&inp.a, self.kind.ab()).as_slice())?;
        self.b.copy_from(to_bytes(&inp.b, self.kind.ab()).as_slice())?;
        self.c_init.copy_from(to_bytes(&inp.c, self.kind.c()).as_slice())?;
        Ok(())
    }

    /// Overwrites the pristine C, e.g. with NaN for the beta = 0 check.
    pub fn set_c_init(&mut self, c: &[f32]) -> anyhow::Result<()> {
        self.c_init.copy_from(to_bytes(c, self.kind.c()).as_slice())?;
        Ok(())
    }

    /// Enqueues dst := pristine C.
    pub fn restore(&self, stream: &Stream, dst: &DeviceBuffer<u8>) -> anyhow::Result<()> {
        let r = unsafe {
            cust::sys::cuMemcpyDtoDAsync_v2(
                dst.as_device_ptr().as_raw(),
                self.c_init.as_device_ptr().as_raw(),
                self.c_init.len(),
                stream.as_inner(),
            )
        };
        anyhow::ensure!(r == cust::sys::CUresult::CUDA_SUCCESS, "cuMemcpyDtoDAsync failed: {r:?}");
        Ok(())
    }

    /// Synchronizes the stream and reads `c` back as f32.
    pub fn read(&self, stream: &Stream, c: &DeviceBuffer<u8>) -> anyhow::Result<Vec<f32>> {
        stream.synchronize()?;
        let mut bytes = vec![0u8; c.len()];
        c.copy_to(&mut bytes)?;
        Ok(from_bytes(&bytes, self.kind.c()))
    }

    /// Enqueues the Phobos kernel into `c_phobos`.
    pub fn launch_phobos(
        &self,
        stream: &Stream,
        func: &cust::function::Function,
        grid: (u32, u32),
        block: u32,
        wide: bool,
        (alpha, beta): (f32, f32),
    ) -> anyhow::Result<()> {
        let (m, n, k) = self.dims;
        crate::harness::launch_gemm(
            func,
            grid,
            block,
            stream,
            self.a.as_device_ptr().as_raw(),
            self.b.as_device_ptr().as_raw(),
            self.c_phobos.as_device_ptr().as_raw(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            beta,
            wide,
        )
    }

    /// Enqueues a split-K kind's two kernels into `c_phobos`: `split` writes
    /// `s` partials over a third grid axis, `reduce` sums them as
    /// `alpha * sum + beta * C` on 32x32 tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_split(
        &self,
        stream: &Stream,
        (split, reduce): (&cust::function::Function, &cust::function::Function),
        (gx, gy, s): (u32, u32, u32),
        block: u32,
        (alpha, beta): (f32, f32),
        wide: bool,
    ) -> anyhow::Result<()> {
        let (m, n, k) = self.dims;
        anyhow::ensure!(s <= self.max_split, "{s} splits exceed the workspace");
        let work = self.work.as_ref().ok_or_else(|| anyhow::anyhow!("this kind has no split-K workspace"))?;
        let (a, b, p, c) = (
            self.a.as_device_ptr().as_raw(),
            self.b.as_device_ptr().as_raw(),
            work.as_device_ptr().as_raw(),
            self.c_phobos.as_device_ptr().as_raw(),
        );
        let reduce_grid = ((m / 32) as u32, (n / 32) as u32);
        unsafe {
            if wide {
                let (m, n, k, sm) = (m as i64, n as i64, k as i64, s as i64 * m as i64);
                launch!(split<<<(gx, gy, s), block, 0, stream>>>(
                    a, a, 0i64, m, k, k, 1i64,
                    b, b, 0i64, k, n, n, 1i64,
                    p, p, 0i64, sm, n, n, 1i64
                ))?;
                launch!(reduce<<<reduce_grid, 256, 0, stream>>>(
                    p, p, 0i64, sm, n, n, 1i64,
                    c, c, 0i64, m, n, n, 1i64,
                    alpha,
                    beta
                ))?;
            } else {
                let (m, n, k, sm) = (m as i32, n as i32, k as i32, s as i32 * m as i32);
                launch!(split<<<(gx, gy, s), block, 0, stream>>>(
                    a, a, 0i32, m, k, k, 1i32,
                    b, b, 0i32, k, n, n, 1i32,
                    p, p, 0i32, sm, n, n, 1i32
                ))?;
                launch!(reduce<<<reduce_grid, 256, 0, stream>>>(
                    p, p, 0i32, sm, n, n, 1i32,
                    c, c, 0i32, m, n, n, 1i32,
                    alpha,
                    beta
                ))?;
            }
        }
        Ok(())
    }

    /// Enqueues the f32 to f16 conversion of A and B the converting baseline
    /// reads.
    pub fn convert(&self, stream: &Stream, conv: &Converter) -> anyhow::Result<()> {
        let (m, n, k) = self.dims;
        let (a16, b16) = self.half.as_ref().ok_or_else(|| anyhow::anyhow!("this kind converts nothing"))?;
        conv.launch(stream, self.a.as_device_ptr().as_raw(), a16.as_device_ptr().as_raw(), m * k)?;
        conv.launch(stream, self.b.as_device_ptr().as_raw(), b16.as_device_ptr().as_raw(), k * n)
    }

    /// Enqueues the cuBLAS baseline into `c_blas`. A converting baseline
    /// reads the f16 copies [`Problem::convert`] last wrote.
    pub fn launch_blas(&self, blas: &CuBlas, (alpha, beta): (f32, f32)) -> anyhow::Result<()> {
        let (m, n, k) = self.dims;
        let dims = (m as i32, n as i32, k as i32);
        let (a, b) = match &self.half {
            Some((a16, b16)) => (a16.as_device_ptr().as_raw(), b16.as_device_ptr().as_raw()),
            None => (self.a.as_device_ptr().as_raw(), self.b.as_device_ptr().as_raw()),
        };
        let c = self.c_blas.as_device_ptr().as_raw();
        match self.kind.baseline() {
            Baseline::Sgemm => blas.matmul(dims.0, dims.1, dims.2, a, b, c, alpha, beta),
            Baseline::GemmExF16F32 => blas.gemm_ex(
                dims,
                (a, b, c),
                (cublas::CUDA_R_16F, cublas::CUDA_R_32F, cublas::CUBLAS_COMPUTE_32F),
                alpha,
                beta,
            ),
            Baseline::GemmExF16F16 => blas.gemm_ex(
                dims,
                (a, b, c),
                (cublas::CUDA_R_16F, cublas::CUDA_R_16F, cublas::CUBLAS_COMPUTE_16F),
                alpha,
                beta,
            ),
        }
    }
}
