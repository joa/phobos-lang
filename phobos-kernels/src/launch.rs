use anyhow::{Result, bail};

#[inline]
pub fn cuda_ok(status: cust::sys::CUresult, what: &str) -> Result<()> {
    match status {
        cust::sys::CUresult::CUDA_SUCCESS => Ok(()),
        other => bail!("{what}: {other:?}"),
    }
}

#[inline]
pub fn push_descriptor(slots: &mut Vec<u64>, ptr: u64, dims: [i64; 2]) {
    let word = |v: i64| v as i32 as u32 as u64;
    slots.extend_from_slice(&[ptr, ptr, 0, word(dims[0]), word(dims[1]), word(dims[1]), 1]);
}

pub const STATIC_SHARED_LIMIT: usize = 48 * 1024;

pub const CTA_THREADS: u32 = 256;

pub const WARP_THREADS: usize = 32;
