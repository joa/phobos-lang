use std::ffi::c_void;

use anyhow::{Result, anyhow};

type Handle = *mut c_void;

/// cublasOperation_t: no transpose.
const CUBLAS_OP_N: i32 = 0;

#[allow(non_snake_case)]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut Handle) -> i32;
    fn cublasDestroy_v2(handle: Handle) -> i32;
    fn cublasSetStream_v2(handle: Handle, stream: *mut c_void) -> i32;
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
    /// Half-precision GEMM (f16 operands, f16 accumulation); the __half
    /// arguments are passed as their u16 bit patterns.
    fn cublasHgemm(
        handle: Handle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const u16,
        a: *const u16,
        lda: i32,
        b: *const u16,
        ldb: i32,
        beta: *const u16,
        c: *mut u16,
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

    /// In-place y = alpha * x + y over contiguous vectors (unit stride). The
    /// pointers are raw device addresses (DevicePointer::as_raw()).
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
    /// cuBLAS is column-major, so this computes the column-major identity
    /// C^T = alpha*B^T*A^T + beta*C^T; a row-major buffer is its column-major
    /// transpose, so nothing is actually transposed or copied.
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

    /// Row-major f16 c = alpha * a * b + beta * c via cublasHgemm: the
    /// half-precision analog of [`CuBlas::matmul`], with f16 operands and f16
    /// accumulation. The pointers are raw device addresses to u16/f16 bit
    /// patterns; alpha/beta are given in f32 and rounded to f16. The same
    /// row-major-is-column-major-transpose trick applies (compute C^T = B^T*A^T).
    #[allow(clippy::too_many_arguments)] // mirrors the BLAS gemm signature
    pub fn matmul_fp16(
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
        let alpha = half::f16::from_f32(alpha).to_bits();
        let beta = half::f16::from_f32(beta).to_bits();
        check(
            unsafe {
                cublasHgemm(
                    self.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha,
                    b as usize as *const u16,
                    n,
                    a as usize as *const u16,
                    k,
                    &beta,
                    c as usize as *mut u16,
                    n,
                )
            },
            "cublasHgemm",
        )
    }
}

impl Drop for CuBlas {
    fn drop(&mut self) {
        unsafe { cublasDestroy_v2(self.handle) };
    }
}
