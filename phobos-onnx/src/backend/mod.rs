use anyhow::Result;

use crate::shape::Dims;

pub mod host;

#[cfg(feature = "cuda")]
pub mod chain;
#[cfg(feature = "cuda")]
pub mod device;

pub use host::{HostBackend, host_layer_norm};

#[cfg(feature = "cuda")]
pub use device::GpuBackend;

/// Row-major data plus its shape.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub dims: Dims,
    pub data: Data,
}

#[derive(Clone, Debug)]
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl Tensor {
    pub fn f32(dims: Dims, data: Vec<f32>) -> Tensor {
        Tensor {
            dims,
            data: Data::F32(data),
        }
    }
    pub fn i64(dims: Dims, data: Vec<i64>) -> Tensor {
        Tensor {
            dims,
            data: Data::I64(data),
        }
    }
    /// Values as f32, casting i64, for callers reading results.
    pub fn to_f32(&self) -> Vec<f32> {
        self.as_f32()
    }
    /// Values as f32, casting i64, for arithmetic.
    fn as_f32(&self) -> Vec<f32> {
        match &self.data {
            Data::F32(v) => v.clone(),
            Data::I64(v) => v.iter().map(|&x| x as f32).collect(),
        }
    }
    fn as_i64(&self) -> Vec<i64> {
        match &self.data {
            Data::I64(v) => v.clone(),
            Data::F32(v) => v.iter().map(|&x| x as i64).collect(),
        }
    }
}

/// The ops worth offloading to Phobos GPU kernels: the FLOP-heavy `Gemm`
/// projections and `LayerNormalization`. Everything else stays on the host, and
/// `layer_norm` defaults to a host implementation so a matmul-only backend
/// still works.
pub trait MatmulBackend {
    /// `C[m,n] = A[m,k] @ B[k,n]`, both row-major.
    fn matmul(&self, a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Result<Vec<f32>>;

    /// [`MatmulBackend::matmul`] where `b_key`, when set, names a constant right
    /// operand so the backend can keep it device-resident across calls. The
    /// default ignores it; only weight-caching backends override this.
    fn matmul_cached(
        &self,
        a: &[f32],
        m: usize,
        k: usize,
        b: &[f32],
        n: usize,
        _b_key: Option<&str>,
    ) -> Result<Vec<f32>> {
        self.matmul(a, m, k, b, n)
    }

    /// LayerNorm over the last axis of an `[rows, w]` view:
    /// `y = (x - mean) / sqrt(var + eps) * scale + bias`, both of length w.
    fn layer_norm(
        &self,
        x: &[f32],
        rows: usize,
        w: usize,
        scale: &[f32],
        bias: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        host_layer_norm(x, rows, w, scale, bias, eps)
    }
}
