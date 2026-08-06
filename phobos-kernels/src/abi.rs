#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelArg {
    Ptr(u64),
    /// A memref offset, size or stride. The kernels compile at a 32-bit index.
    I32(i32),
    F32(f32),
}

impl KernelArg {
    pub fn slot(self) -> u64 {
        match self {
            KernelArg::Ptr(p) => p,
            KernelArg::I32(v) => v as u32 as u64,
            KernelArg::F32(v) => v.to_bits() as u64,
        }
    }
}

pub fn push_tensor_descriptor(out: &mut Vec<KernelArg>, ptr: u64, dims: &[i64]) {
    out.push(KernelArg::Ptr(ptr)); // allocated pointer
    out.push(KernelArg::Ptr(ptr)); // aligned pointer
    out.push(KernelArg::I32(0)); // offset
    for &d in dims {
        out.push(KernelArg::I32(d as i32));
    }
    for s in row_major_strides(dims) {
        out.push(KernelArg::I32(s as i32));
    }
}

pub fn row_major_strides(dims: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; dims.len()];
    let mut acc = 1i64;
    for i in (0..dims.len()).rev() {
        strides[i] = acc;
        acc *= dims[i];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(row_major_strides(&[4]), vec![1]);
        assert_eq!(row_major_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn rank1_descriptor_matches_saxpy_abi() {
        // saxpy passes: ptr, ptr, 0i32, n, 1i32
        let mut args = Vec::new();
        push_tensor_descriptor(&mut args, 0xdead_beef, &[8]);
        assert_eq!(
            args,
            vec![
                KernelArg::Ptr(0xdead_beef),
                KernelArg::Ptr(0xdead_beef),
                KernelArg::I32(0),
                KernelArg::I32(8),
                KernelArg::I32(1),
            ]
        );
    }

    #[test]
    fn rank2_descriptor_matches_gemm_abi() {
        // gemm passes A[M,K] as: ptr, ptr, 0i32, M, K, K, 1i32
        let mut args = Vec::new();
        push_tensor_descriptor(&mut args, 0x1000, &[6, 4]);
        assert_eq!(
            args,
            vec![
                KernelArg::Ptr(0x1000),
                KernelArg::Ptr(0x1000),
                KernelArg::I32(0),
                KernelArg::I32(6), // size M
                KernelArg::I32(4), // size K
                KernelArg::I32(4), // stride M = K
                KernelArg::I32(1), // stride K
            ]
        );
    }

    #[test]
    fn scalar_slots_are_little_endian() {
        assert_eq!(KernelArg::I32(-1).slot(), 0xffff_ffff);
        assert_eq!(KernelArg::F32(1.0).slot(), 1.0f32.to_bits() as u64);
        assert_eq!(KernelArg::Ptr(0x1234_5678_9abc).slot(), 0x1234_5678_9abc);
    }
}
