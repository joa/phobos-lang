use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use anyhow::{Result, bail, ensure};

use crate::isa::Region;

pub fn file_path(uri: &str) -> Result<PathBuf> {
    let rest = match uri.strip_prefix("file://") {
        Some(r) => r,
        None => bail!("unsupported storage URI '{uri}' (expected file://)"),
    };
    let rest = if rest.starts_with('/') && rest.as_bytes().get(2) == Some(&b':') {
        &rest[1..]
    } else {
        rest
    };
    Ok(PathBuf::from(rest))
}

fn byte_spans(tensor_shape: &[u64], region: &Region) -> Result<Vec<(u64, usize)>> {
    match region.shape.len() {
        1 => {
            let o = region.offset[0];
            let n = region.shape[0] as usize;
            Ok(vec![(o * 4, n)])
        }
        2 => {
            let (r0, c0) = (region.offset[0], region.offset[1]);
            let (sr, sc) = (region.shape[0] as usize, region.shape[1] as usize);
            ensure!(
                tensor_shape.len() == 2,
                "rank-2 region needs a rank-2 tensor"
            );
            let pitch = tensor_shape[1];
            Ok((0..sr)
                .map(|r| {
                    let base = (r0 + r as u64) * pitch + c0;
                    (base * 4, sc)
                })
                .collect())
        }
        r => bail!("file:// storage supports rank <= 2, got rank {r}"),
    }
}

pub fn load_f32(uri: &str, tensor_shape: &[u64], region: &Region) -> Result<Vec<f32>> {
    let mut f = OpenOptions::new().read(true).open(file_path(uri)?)?;
    let total: usize = region.shape.iter().product::<u64>() as usize;
    let mut out = vec![0f32; total];
    let mut written = 0usize;
    for (byte_off, elems) in byte_spans(tensor_shape, region)? {
        f.seek(SeekFrom::Start(byte_off))?;
        let dst = unsafe_cast_mut(&mut out[written..written + elems]);
        f.read_exact(dst)?;
        written += elems;
    }
    Ok(out)
}

pub fn store_f32(uri: &str, tensor_shape: &[u64], region: &Region, data: &[f32]) -> Result<()> {
    let total: usize = region.shape.iter().product::<u64>() as usize;
    ensure!(data.len() == total, "store data/region length mismatch");
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false) // region writes preserve the rest of the tensor
        .open(file_path(uri)?)?;
    let mut read = 0usize;
    for (byte_off, elems) in byte_spans(tensor_shape, region)? {
        f.seek(SeekFrom::Start(byte_off))?;
        let src = unsafe_cast(&data[read..read + elems]);
        f.write_all(src)?;
        read += elems;
    }
    Ok(())
}

pub fn write_tensor_f32(uri: &str, data: &[f32]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path(uri)?)?;
    f.write_all(unsafe_cast(data))?;
    Ok(())
}

pub fn read_tensor_f32(uri: &str, len: usize) -> Result<Vec<f32>> {
    let mut f = OpenOptions::new().read(true).open(file_path(uri)?)?;
    let mut out = vec![0f32; len];
    f.read_exact(unsafe_cast_mut(&mut out))?;
    Ok(out)
}

pub fn f32_to_le_bytes(data: &[f32]) -> Vec<u8> {
    unsafe_cast(data).to_vec()
}

pub fn le_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(
        bytes.len().is_multiple_of(4),
        "tile byte length {} not a multiple of 4",
        bytes.len()
    );
    let mut out = vec![0f32; bytes.len() / 4];
    unsafe_cast_mut(&mut out).copy_from_slice(bytes);
    Ok(out)
}

fn unsafe_cast(s: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn unsafe_cast_mut(s: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}
