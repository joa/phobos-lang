use std::ffi::c_void;

use anyhow::{Result, anyhow};

type Handle = *mut c_void;

/// cublasOperation_t: no transpose.
const CUBLAS_OP_N: i32 = 0;

/// cudaDataType values, from library_types.h.
pub const CUDA_R_32F: i32 = 0;
pub const CUDA_R_16F: i32 = 2;

/// cublasComputeType_t values, from cublas_api.h.
pub const CUBLAS_COMPUTE_16F: i32 = 64;
pub const CUBLAS_COMPUTE_32F: i32 = 68;

/// cublasGemmAlgo_t: let the heuristic pick, tensor cores allowed.
const CUBLAS_GEMM_DEFAULT: i32 = -1;

#[allow(non_snake_case)]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut Handle) -> i32;
    fn cublasDestroy_v2(handle: Handle) -> i32;
    fn cublasSetStream_v2(handle: Handle, stream: *mut c_void) -> i32;
    fn cublasGetVersion_v2(handle: Handle, version: *mut i32) -> i32;
    /// C = alpha * op(A) * op(B) + beta * C with explicit storage and
    /// compute types (column-major).
    fn cublasGemmEx(
        handle: Handle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        a_type: i32,
        lda: i32,
        b: *const c_void,
        b_type: i32,
        ldb: i32,
        beta: *const c_void,
        c: *mut c_void,
        c_type: i32,
        ldc: i32,
        compute: i32,
        algo: i32,
    ) -> i32;
    /// y = alpha * x + y (single precision).
    fn cublasSaxpy_v2(
        handle: Handle,
        n: i32,
        alpha: *const f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    ) -> i32;
    /// C = alpha * op(A) * op(B) + beta * C (column-major).
    fn cublasSgemm_v2(
        handle: Handle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> i32;
}

fn check(status: i32, what: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(anyhow!("{what} failed: cublasStatus_t = {status}"))
    }
}

pub struct CuBlas {
    handle: Handle,
}

impl CuBlas {
    pub fn new(stream: &cust::stream::Stream) -> Result<Self> {
        let mut handle: Handle = std::ptr::null_mut();
        check(unsafe { cublasCreate_v2(&mut handle) }, "cublasCreate")?;
        check(
            unsafe { cublasSetStream_v2(handle, stream.as_inner() as *mut c_void) },
            "cublasSetStream",
        )?;
        Ok(CuBlas { handle })
    }

    /// The library's version, e.g. 130400 for 13.4.0.
    pub fn version(&self) -> Result<i32> {
        let mut v = 0;
        check(unsafe { cublasGetVersion_v2(self.handle, &mut v) }, "cublasGetVersion")?;
        Ok(v)
    }

    /// In-place y = alpha * x + y over contiguous vectors. The pointers are
    /// raw device addresses (`DevicePointer::as_raw()`).
    pub fn saxpy(&self, n: i32, alpha: f32, x: u64, y: u64) -> Result<()> {
        check(
            unsafe {
                cublasSaxpy_v2(
                    self.handle,
                    n,
                    &alpha,
                    x as usize as *const f32,
                    1,
                    y as usize as *mut f32,
                    1,
                )
            },
            "cublasSaxpy",
        )
    }
}

impl CuBlas {
    /// Row-major c = alpha * a * b + beta * c (a: m x k, b: k x n, c: m x n).
    /// cuBLAS is column-major, and a row-major buffer is its column-major
    /// transpose. So this computes C^T = alpha*B^T*A^T + beta*C^T, with no
    /// transpose or copy.
    #[allow(clippy::too_many_arguments)] // mirrors the BLAS gemm signature
    pub fn matmul(
        &self,
        m: i32,
        n: i32,
        k: i32,
        a: u64,
        b: u64,
        c: u64,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        check(
            unsafe {
                cublasSgemm_v2(
                    self.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha,
                    b as usize as *const f32,
                    n,
                    a as usize as *const f32,
                    k,
                    &beta,
                    c as usize as *mut f32,
                    n,
                )
            },
            "cublasSgemm",
        )
    }
}

impl CuBlas {
    /// Row-major c = alpha * a * b + beta * c through cublasGemmEx, with A and
    /// B stored as `ab_type`, C as `c_type`, accumulated as `compute`. The
    /// scalars are f32 for CUBLAS_COMPUTE_32F and f16 for CUBLAS_COMPUTE_16F,
    /// as cuBLAS requires.
    #[allow(clippy::too_many_arguments)] // mirrors the BLAS gemm signature
    pub fn gemm_ex(
        &self,
        (m, n, k): (i32, i32, i32),
        (a, b, c): (u64, u64, u64),
        (ab_type, c_type, compute): (i32, i32, i32),
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let (alpha16, beta16) = (
            half::f16::from_f32(alpha).to_bits(),
            half::f16::from_f32(beta).to_bits(),
        );
        let (alpha_p, beta_p): (*const c_void, *const c_void) = if compute == CUBLAS_COMPUTE_16F {
            (&alpha16 as *const u16 as _, &beta16 as *const u16 as _)
        } else {
            (&alpha as *const f32 as _, &beta as *const f32 as _)
        };
        // Row-major as column-major transposed, as in [`CuBlas::matmul`].
        check(
            unsafe {
                cublasGemmEx(
                    self.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    alpha_p,
                    b as usize as *const c_void,
                    ab_type,
                    n,
                    a as usize as *const c_void,
                    ab_type,
                    k,
                    beta_p,
                    c as usize as *mut c_void,
                    c_type,
                    n,
                    compute,
                    CUBLAS_GEMM_DEFAULT,
                )
            },
            "cublasGemmEx",
        )
    }
}

impl Drop for CuBlas {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.handle) };
    }
}
