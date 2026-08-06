use anyhow::{Result, bail};

use crate::abi::row_major_strides;
use crate::shape::Dims;

fn numel(dims: &[i64]) -> usize {
    dims.iter().product::<i64>() as usize
}

/// Resolve a Reshape target shape against the input dims, handling the two
/// ONNX conventions: a `0` copies the input extent at that position, and a
/// single `-1` is inferred from the total element count.
pub fn reshape_dims(input: &[i64], target: &[i64]) -> Result<Dims> {
    let total = numel(input);
    let mut out: Dims = Vec::with_capacity(target.len());
    let mut infer_axis = None;
    let mut known = 1i64;
    for (i, &t) in target.iter().enumerate() {
        match t {
            0 => {
                let d = *input
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("Reshape 0 at axis {i} has no input axis"))?;
                out.push(d);
                known *= d;
            }
            -1 => {
                if infer_axis.is_some() {
                    bail!("Reshape has more than one -1");
                }
                infer_axis = Some(i);
                out.push(-1);
            }
            n if n > 0 => {
                out.push(n);
                known *= n;
            }
            n => bail!("Reshape has a negative dim {n}"),
        }
    }
    if let Some(ax) = infer_axis {
        if known == 0 || !total.is_multiple_of(known as usize) {
            bail!("Reshape cannot infer axis: {total} not divisible by {known}");
        }
        out[ax] = (total / known as usize) as i64;
    } else if numel(&out) != total {
        bail!("Reshape changes element count: {} vs {total}", numel(&out));
    }
    Ok(out)
}

/// Permute axes. `perm[k]` is the input axis that becomes output axis `k`.
pub fn transpose(data: &[f32], dims: &[i64], perm: &[usize]) -> Result<(Vec<f32>, Dims)> {
    let rank = dims.len();
    if perm.len() != rank {
        bail!("Transpose perm has {} axes, tensor has {rank}", perm.len());
    }
    let mut seen = vec![false; rank];
    for &p in perm {
        if p >= rank || std::mem::replace(&mut seen[p], true) {
            bail!("Transpose perm {perm:?} is not a permutation of 0..{rank}");
        }
    }
    let out_dims: Dims = perm.iter().map(|&p| dims[p]).collect();
    let in_strides = row_major_strides(dims);
    let out_strides = row_major_strides(&out_dims);
    let total = numel(dims);
    let mut out = vec![0.0f32; total];
    for (lin, slot) in out.iter_mut().enumerate() {
        let mut rem = lin;
        let mut src = 0usize;
        for k in 0..rank {
            let c = rem / out_strides[k] as usize;
            rem %= out_strides[k] as usize;
            src += c * in_strides[perm[k]] as usize;
        }
        *slot = data[src];
    }
    Ok((out, out_dims))
}

/// Gather rows of `data` along `axis`, negative indices wrapping. The output
/// shape is `data[..axis] ++ indices_dims ++ data[axis+1..]`.
pub fn gather(
    data: &[f32],
    data_dims: &[i64],
    indices: &[i64],
    indices_dims: &[i64],
    axis: usize,
) -> Result<(Vec<f32>, Dims)> {
    if axis >= data_dims.len() {
        bail!("Gather axis {axis} out of range for {data_dims:?}");
    }
    let axis_len = data_dims[axis];
    let outer: usize = numel(&data_dims[..axis]);
    let inner: usize = numel(&data_dims[axis + 1..]);
    // The product of the index dims, one for a scalar index.
    let ni = indices.len();

    let resolved: Vec<usize> = indices
        .iter()
        .map(|&i| {
            let i = if i < 0 { i + axis_len } else { i };
            if i < 0 || i >= axis_len {
                bail!("Gather index {i} out of range [0,{axis_len})");
            }
            Ok(i as usize)
        })
        .collect::<Result<_>>()?;

    let mut out = vec![0.0f32; outer * ni * inner];
    for o in 0..outer {
        for (k, &idx) in resolved.iter().enumerate() {
            let src = (o * axis_len as usize + idx) * inner;
            let dst = (o * ni + k) * inner;
            out[dst..dst + inner].copy_from_slice(&data[src..src + inner]);
        }
    }

    let mut out_dims: Dims = data_dims[..axis].to_vec();
    out_dims.extend_from_slice(indices_dims);
    out_dims.extend_from_slice(&data_dims[axis + 1..]);
    Ok((out, out_dims))
}

/// Concatenate tensors of identical shape except along `axis`.
pub fn concat(inputs: &[(&[f32], &[i64])], axis: usize) -> Result<(Vec<f32>, Dims)> {
    let (_, first) = inputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("Concat needs inputs"))?;
    let rank = first.len();
    if axis >= rank {
        bail!("Concat axis {axis} out of range for rank {rank}");
    }
    let outer: usize = numel(&first[..axis]);
    let inner: usize = numel(&first[axis + 1..]);
    let mut axis_total = 0i64;
    for (_, dims) in inputs {
        if dims.len() != rank
            || dims[..axis] != first[..axis]
            || dims[axis + 1..] != first[axis + 1..]
        {
            bail!("Concat inputs disagree off axis {axis}: {dims:?} vs {first:?}");
        }
        axis_total += dims[axis];
    }

    let mut out_dims: Dims = first.to_vec();
    out_dims[axis] = axis_total;
    let mut out = vec![0.0f32; numel(&out_dims)];

    let out_axis = axis_total as usize;
    let mut axis_off = 0usize;
    for (data, dims) in inputs {
        let seg = dims[axis] as usize;
        for o in 0..outer {
            let src = o * seg * inner;
            let dst = (o * out_axis + axis_off) * inner;
            out[dst..dst + seg * inner].copy_from_slice(&data[src..src + seg * inner]);
        }
        axis_off += seg;
    }
    Ok((out, out_dims))
}

/// Split a tensor along `axis` into segments of the given sizes.
pub fn split(
    data: &[f32],
    dims: &[i64],
    axis: usize,
    sizes: &[i64],
) -> Result<Vec<(Vec<f32>, Dims)>> {
    if axis >= dims.len() {
        bail!("Split axis {axis} out of range for {dims:?}");
    }
    if sizes.iter().sum::<i64>() != dims[axis] {
        bail!(
            "Split sizes {sizes:?} do not sum to axis extent {}",
            dims[axis]
        );
    }
    let outer: usize = numel(&dims[..axis]);
    let inner: usize = numel(&dims[axis + 1..]);
    let axis_len = dims[axis] as usize;

    let mut outputs = Vec::with_capacity(sizes.len());
    let mut axis_off = 0usize;
    for &sz in sizes {
        let seg = sz as usize;
        let mut piece = vec![0.0f32; outer * seg * inner];
        for o in 0..outer {
            let src = (o * axis_len + axis_off) * inner;
            let dst = o * seg * inner;
            piece[dst..dst + seg * inner].copy_from_slice(&data[src..src + seg * inner]);
        }
        let mut pdims: Dims = dims.to_vec();
        pdims[axis] = sz;
        outputs.push((piece, pdims));
        axis_off += seg;
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_infers_and_copies() {
        assert_eq!(reshape_dims(&[2, 3, 4], &[6, 4]).unwrap(), vec![6, 4]);
        assert_eq!(reshape_dims(&[2, 3, 4], &[-1, 4]).unwrap(), vec![6, 4]);
        assert_eq!(reshape_dims(&[2, 3, 4], &[0, -1]).unwrap(), vec![2, 12]);
        assert!(reshape_dims(&[2, 3], &[5]).is_err()); // element count changes
    }

    #[test]
    fn transpose_2d_matches_matrix_transpose() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (out, dims) = transpose(&data, &[2, 3], &[1, 0]).unwrap();
        assert_eq!(dims, vec![3, 2]);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_3d_permutes_axes() {
        // shape [2,1,3], perm [1,0,2] -> [1,2,3], data order preserved per row.
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let (out, dims) = transpose(&data, &[2, 1, 3], &[1, 0, 2]).unwrap();
        assert_eq!(dims, vec![1, 2, 3]);
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn transpose_rejects_bad_perm() {
        assert!(transpose(&[1.0, 2.0], &[2], &[0, 1]).is_err());
        assert!(transpose(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[0, 0]).is_err());
    }

    #[test]
    fn gather_embeddings_selects_rows() {
        // data [4,2] vocab table, indices [3] -> [3,2]
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let (out, dims) = gather(&data, &[4, 2], &[2, 0, 3], &[3], 0).unwrap();
        assert_eq!(dims, vec![3, 2]);
        assert_eq!(out, vec![4.0, 5.0, 0.0, 1.0, 6.0, 7.0]);
    }

    #[test]
    fn gather_negative_index_wraps() {
        let data: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let (out, _) = gather(&data, &[4, 2], &[-1], &[1], 0).unwrap();
        assert_eq!(out, vec![6.0, 7.0]);
    }

    #[test]
    fn concat_and_split_round_trip() {
        // Concat two [2,2] along axis 1 -> [2,4], then split back.
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let (cat, dims) = concat(&[(&a, &[2, 2]), (&b, &[2, 2])], 1).unwrap();
        assert_eq!(dims, vec![2, 4]);
        assert_eq!(cat, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);

        let parts = split(&cat, &dims, 1, &[2, 2]).unwrap();
        assert_eq!(parts[0].0, a);
        assert_eq!(parts[1].0, b);
        assert_eq!(parts[0].1, vec![2, 2]);
    }

    #[test]
    fn split_qkv_along_last_axis() {
        // [1, 6] -> three [1, 2] (the QKV split shape).
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let parts = split(&data, &[1, 6], 1, &[2, 2, 2]).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].0, vec![0.0, 1.0]);
        assert_eq!(parts[1].0, vec![2.0, 3.0]);
        assert_eq!(parts[2].0, vec![4.0, 5.0]);
    }
}
