use std::cell::RefCell;

use anyhow::{Context, Result, bail, ensure};

use crate::backend::{Backend, Buf, Plane, Q8_BLOCK, QAct};
use crate::tensor::q8_0_blocks;
use crate::{GgmlType, Gguf, TensorInfo};

/// Output columns [`Linear::fuse`] rounds up to: the widest column tile the
/// batched projection has. 8224 becomes 8448, 2.7% more arithmetic against a
/// tile that is 23% faster.
const FUSE_ALIGN: usize = 256;

/// A projection matrix, transposed at load into the `[in, out]` row-major layout
/// [`Backend::matmul`] expects. GGUF stores linear weights the other way round:
/// the ggml extents are `[in, out]` with `in` fastest, which in memory is a
/// row-major `[out, in]` matrix.
pub(crate) struct Linear {
    weight: Weights,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
    key: String,
}

/// How a projection's weights are held between load and upload.
enum Weights {
    Dense(Vec<f32>),
    /// Q8_0 kept quantized: signed bytes plus one scale per block of `in_dim`.
    Q8 {
        qs: Vec<i8>,
        scales: Vec<f32>,
    },
}

impl Linear {
    pub(crate) fn load(gguf: &Gguf, name: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
        let info = gguf
            .tensor(name)
            .with_context(|| format!("missing tensor '{name}'"))?;
        check_dims(info, &[in_dim as u64, out_dim as u64])?;

        // The quantized bytes need no rearrangement: GGUF already stores them
        // [out, in] with `in` contiguous, which is what the four-way byte dot
        // product wants. Only the scales are transposed, so one block's scales
        // for a run of outputs sit together. Nothing is requantized.
        let weight = if info.ggml_type == GgmlType::Q8_0 && in_dim.is_multiple_of(Q8_BLOCK) {
            let numel = in_dim * out_dim;
            let (qs, src_scales) = q8_0_blocks(gguf.tensor_bytes(info)?, numel)?;
            let blocks = in_dim / Q8_BLOCK;
            let mut scales = vec![0.0f32; blocks * out_dim];

            for o in 0..out_dim {
                for b in 0..blocks {
                    scales[b * out_dim + o] = src_scales[o * blocks + b];
                }
            }

            Weights::Q8 { qs, scales }
        } else {
            let source = gguf.dequantize(name)?;
            let mut dense = vec![0.0f32; source.len()];

            for o in 0..out_dim {
                let row = &source[o * in_dim..(o + 1) * in_dim];
                for (i, &v) in row.iter().enumerate() {
                    dense[i * out_dim + o] = v;
                }
            }

            Weights::Dense(dense)
        };

        Ok(Linear {
            weight,
            in_dim,
            out_dim,
            key: name.to_string(),
        })
    }

    /// The same projection with its output channels permuted: output `j` of the
    /// result is output `order[j]` of this one. It goes up under a key of its
    /// own, and nothing is requantized, since a Q8_0 block covers 32 inputs of
    /// one output and a row moves as whole bytes and whole scales.
    ///
    /// The rotary layout wants this. ggml rotates either consecutive pairs or
    /// pairs half a head apart and the backend implements only the second, so
    /// an architecture using the first permutes its query and key weights.
    pub(crate) fn reorder_outputs(&self, suffix: &str, order: &[usize]) -> Result<Linear> {
        ensure!(
            order.len() == self.out_dim && order.iter().all(|&j| j < self.out_dim),
            "a reordering of '{}' must be a permutation of its {} outputs",
            self.key,
            self.out_dim
        );

        let (in_dim, out_dim) = (self.in_dim, self.out_dim);
        let weight = match &self.weight {
            Weights::Q8 { qs, scales } => {
                let blocks = in_dim / Q8_BLOCK;
                let mut moved = vec![0i8; qs.len()];
                let mut moved_scales = vec![0.0f32; scales.len()];
                for (j, &from) in order.iter().enumerate() {
                    moved[j * in_dim..(j + 1) * in_dim]
                        .copy_from_slice(&qs[from * in_dim..(from + 1) * in_dim]);
                    for b in 0..blocks {
                        moved_scales[b * out_dim + j] = scales[b * out_dim + from];
                    }
                }
                Weights::Q8 {
                    qs: moved,
                    scales: moved_scales,
                }
            }
            // Dense weights are held [in, out], so an output is a column.
            Weights::Dense(data) => {
                let mut moved = vec![0.0f32; data.len()];
                for i in 0..in_dim {
                    for (j, &from) in order.iter().enumerate() {
                        moved[i * out_dim + j] = data[i * out_dim + from];
                    }
                }
                Weights::Dense(moved)
            }
        };

        Ok(Linear {
            weight,
            in_dim,
            out_dim,
            key: format!("{}.{suffix}", self.key),
        })
    }

    /// Stacks weights that share an input into one projection.
    ///
    /// Attention's query, key and value read the same normalized row, as do the
    /// two halves of a SwiGLU and the four projections of a delta net. Run
    /// separately they are a launch each for a narrow slice of the output; a
    /// delta net's decay and beta are sixteen wide, two blocks of a forty-eight
    /// block card. Stacked they are one launch over a wide output.
    ///
    /// Stacking along the output axis appends the quantized bytes and
    /// requantizes nothing. The scales are held `[block, out]`, so those
    /// interleave rather than append.
    pub(crate) fn fuse(parts: &[&Linear]) -> Result<Linear> {
        let (first, rest) = parts.split_first().context("fusing needs a weight")?;
        let in_dim = first.in_dim;

        ensure!(
            rest.iter().all(|p| p.in_dim == in_dim),
            "fused projections must share their input width"
        );

        // A fused width that does not divide by the tile would send the whole
        // projection to the narrower kernel, so it is padded with zero columns.
        // The padding sits past every part's window, so nothing reads it.
        let out_dim = parts
            .iter()
            .map(|p| p.out_dim)
            .sum::<usize>()
            .next_multiple_of(FUSE_ALIGN);

        let key = parts
            .iter()
            .map(|p| p.key.as_str())
            .collect::<Vec<_>>()
            .join("+");

        let quantized = parts.iter().all(|p| matches!(p.weight, Weights::Q8 { .. }));
        let weight = if quantized {
            let blocks = in_dim / Q8_BLOCK;
            let mut qs = Vec::with_capacity(in_dim * out_dim);
            let mut scales = vec![0.0f32; blocks * out_dim];
            let mut at = 0;
            for part in parts {
                let Weights::Q8 { qs: pq, scales: ps } = &part.weight else {
                    unreachable!("checked above");
                };
                qs.extend_from_slice(pq);
                for b in 0..blocks {
                    let row = &ps[b * part.out_dim..(b + 1) * part.out_dim];
                    scales[b * out_dim + at..b * out_dim + at + part.out_dim].copy_from_slice(row);
                }
                at += part.out_dim;
            }
            qs.resize(in_dim * out_dim, 0);
            Weights::Q8 { qs, scales }
        } else {
            // Dense weights are held [in, out], so a row of the fused weight is
            // each part's row end to end.
            let mut dense = vec![0.0f32; in_dim * out_dim];
            for i in 0..in_dim {
                let mut at = 0;
                for part in parts {
                    let row = match &part.weight {
                        Weights::Dense(data) => &data[i * part.out_dim..(i + 1) * part.out_dim],
                        Weights::Q8 { .. } => {
                            bail!("cannot fuse a dense weight with a quantized one")
                        }
                    };
                    dense[i * out_dim + at..i * out_dim + at + part.out_dim].copy_from_slice(row);
                    at += part.out_dim;
                }
            }
            Weights::Dense(dense)
        };

        Ok(Linear {
            weight,
            in_dim,
            out_dim,
            key,
        })
    }

    /// Project `x[rows, in_dim]` into a freshly allocated `[rows, out_dim]`.
    pub(crate) fn forward(&self, backend: &dyn Backend, x: Buf, rows: usize) -> Result<Buf> {
        let out = backend.alloc(rows * self.out_dim)?;
        self.project_into(backend, x, rows, out)?;
        Ok(out)
    }

    /// Write output row `index` of this weight into `out`, dequantizing it.
    /// Only the embedding table uses this: row `token` of `token_embd.weight`
    /// is that token's embedding, and reading it out of the quantized bytes
    /// removes the f32 copy without changing the result.
    pub(crate) fn row_into(&self, index: usize, out: &mut [f32]) -> Result<()> {
        ensure!(
            index < self.out_dim && out.len() == self.in_dim,
            "row {index} of a [{}, {}] weight does not fit a {}-element slice",
            self.out_dim,
            self.in_dim,
            out.len()
        );
        match &self.weight {
            // Held [in, out], so a row is a strided gather down a column.
            Weights::Dense(data) => {
                for (j, v) in out.iter_mut().enumerate() {
                    *v = data[j * self.out_dim + index];
                }
            }
            Weights::Q8 { qs, scales } => {
                let row = &qs[index * self.in_dim..][..self.in_dim];
                for (j, (v, &q)) in out.iter_mut().zip(row).enumerate() {
                    *v = f32::from(q) * scales[(j / Q8_BLOCK) * self.out_dim + index];
                }
            }
        }
        Ok(())
    }

    /// Project into a destination the caller owns.
    pub(crate) fn project_into(
        &self,
        backend: &dyn Backend,
        x: Buf,
        rows: usize,
        out: Buf,
    ) -> Result<()> {
        self.project_shared(backend, x, None, rows, out)
    }

    /// Project and add into `dest`, the residual connection's epilogue. The
    /// dense fallback has no accumulating form, so it keeps the separate pass.
    pub(crate) fn add_into(
        &self,
        backend: &dyn Backend,
        x: Buf,
        rows: usize,
        dest: Buf,
    ) -> Result<()> {
        match &self.weight {
            Weights::Q8 { .. } => {
                let act = backend.quantize_act(x, rows, self.in_dim)?;
                self.add_into_act(backend, act, rows, dest)
            }
            Weights::Dense(_) => {
                let out = self.forward(backend, x, rows)?;
                backend.add_into(dest, out)?;
                backend.release(out);
                Ok(())
            }
        }
        .with_context(|| format!("residual matmul for '{}'", self.key))
    }

    /// [`Linear::add_into`] against an activation quantized already.
    pub(crate) fn add_into_act(
        &self,
        backend: &dyn Backend,
        act: QAct,
        rows: usize,
        dest: Buf,
    ) -> Result<()> {
        let Weights::Q8 { qs, scales } = &self.weight else {
            bail!("a dense weight has no accumulating form");
        };
        let w = backend.constant_q8(&self.key, qs, scales, self.in_dim, self.out_dim)?;
        backend
            .matmul_q8_add(act, rows, self.in_dim, w, self.out_dim, dest)
            .with_context(|| format!("residual matmul for '{}'", self.key))
    }

    /// [`Linear::forward`] against an activation quantized already. `act` must
    /// be `x` at this weight's input width; a dense weight ignores it.
    pub(crate) fn forward_act(
        &self,
        backend: &dyn Backend,
        x: Buf,
        act: QAct,
        rows: usize,
    ) -> Result<Buf> {
        let out = backend.alloc(rows * self.out_dim)?;
        self.project_shared(backend, x, Some(act), rows, out)?;
        Ok(out)
    }

    fn project_shared(
        &self,
        backend: &dyn Backend,
        x: Buf,
        act: Option<QAct>,
        rows: usize,
        out: Buf,
    ) -> Result<()> {
        match &self.weight {
            Weights::Dense(data) => {
                let w = backend.constant(&self.key, data)?;
                backend.matmul(x, rows, self.in_dim, w, self.out_dim, out)
            }
            Weights::Q8 { qs, scales } => {
                let w = backend.constant_q8(&self.key, qs, scales, self.in_dim, self.out_dim)?;
                match act {
                    Some(act) => {
                        backend.matmul_q8_act(act, rows, self.in_dim, w, self.out_dim, out)
                    }
                    None => backend.matmul_q8(x, rows, self.in_dim, w, self.out_dim, out),
                }
            }
        }
        .with_context(|| format!("matmul for '{}'", self.key))
    }
}

/// A named one-dimensional weight, uploaded once and referred to by handle.
pub(crate) struct Gain {
    pub(crate) data: Vec<f32>,
    pub(crate) key: String,
}

impl Gain {
    pub(crate) fn load(gguf: &Gguf, name: &str, len: usize) -> Result<Gain> {
        Ok(Gain {
            data: load_vector(gguf, name, len)?,
            key: name.to_string(),
        })
    }

    /// A constant with no tensor of its own: a rearrangement of one, or one
    /// variant's reading of it. The key must distinguish it from every other
    /// reading, since the backend caches the upload under it.
    pub(crate) fn derived(key: String, data: Vec<f32>) -> Gain {
        Gain { data, key }
    }

    pub(crate) fn buf(&self, backend: &dyn Backend) -> Result<Buf> {
        backend.constant(&self.key, &self.data)
    }
}

pub(crate) fn load_vector(gguf: &Gguf, name: &str, len: usize) -> Result<Vec<f32>> {
    let info = gguf
        .tensor(name)
        .with_context(|| format!("missing tensor '{name}'"))?;
    check_dims(info, &[len as u64])?;
    gguf.dequantize(name)
}

pub(crate) fn check_dims(info: &TensorInfo, expected: &[u64]) -> Result<()> {
    ensure!(
        info.dims == expected,
        "tensor '{}' has ggml extents {:?}, expected {expected:?}",
        info.name,
        info.dims
    );
    Ok(())
}

/// The SwiGLU feed-forward every architecture here ends a block with.
pub(crate) struct Ffn {
    /// Gate and up stacked: they read the same row, so one launch over a doubly
    /// wide output replaces two over half of it.
    gate_up: Linear,
    down: Linear,
}

impl Ffn {
    pub(crate) fn load(gguf: &Gguf, prefix: &str, d_model: usize, d_ff: usize) -> Result<Ffn> {
        let gate = Linear::load(gguf, &format!("{prefix}.ffn_gate.weight"), d_model, d_ff)?;
        let up = Linear::load(gguf, &format!("{prefix}.ffn_up.weight"), d_model, d_ff)?;
        Ok(Ffn {
            gate_up: Linear::fuse(&[&gate, &up])?,
            down: Linear::load(gguf, &format!("{prefix}.ffn_down.weight"), d_ff, d_model)?,
        })
    }

    /// SwiGLU: `down(silu(gate(x)) * up(x))`, added into `dest`. Nothing leaves
    /// the backend, so the two wide intermediates never reach the host.
    pub(crate) fn forward(
        &self,
        backend: &dyn Backend,
        x: Buf,
        act: QAct,
        rows: usize,
        dest: Buf,
    ) -> Result<()> {
        let width = self.down.in_dim;
        let both = self.gate_up.forward_act(backend, x, act, rows)?;
        // A separate destination: the host backend moves the destination out of
        // its slab while the sources stay borrowed, so it must not alias them.
        let joined = backend.alloc(rows * width)?;
        if rows == 1 {
            // One launch for the gate, the product, and the quantized copy.
            let act = backend.swiglu_q(both, 0, both, width, joined, width)?;
            backend.release(both);
            self.down.add_into_act(backend, act, rows, dest)?;
            backend.release(joined);
            return Ok(());
        }
        // Past one row the two halves interleave, so the SwiGLU reads them
        // where they lie rather than pulling them apart first.
        let stacked = |offset| Plane {
            buf: both,
            offset,
            pitch: 2 * width,
        };
        backend.swiglu_planes(stacked(0), stacked(width), joined, rows, width)?;
        backend.release(both);
        self.down.add_into(backend, joined, rows, dest)?;
        backend.release(joined);
        Ok(())
    }
}

/// An attention block's key and value caches, `[capacity, n_kv * head_dim]`
/// each. Backend-resident: at a few thousand positions the cache is the largest
/// thing in the block.
#[derive(Default)]
pub(crate) struct KvCache {
    keys: Option<Buf>,
    values: Option<Buf>,
    /// Positions the pair currently has room for.
    capacity: usize,
}

impl KvCache {
    /// The pair, grown to hold `total` positions. Doubling amortizes the copy
    /// to a constant per position; sizing to the context length up front would
    /// reserve gigabytes for a prompt of a dozen tokens.
    pub(crate) fn reserve(
        &mut self,
        backend: &dyn Backend,
        total: usize,
        width: usize,
    ) -> Result<(Buf, Buf)> {
        if self.capacity < total {
            let want = total.next_power_of_two().max(64);
            for slot in [&mut self.keys, &mut self.values] {
                let grown = backend.alloc(want * width)?;
                if let Some(old) = slot.replace(grown) {
                    backend.copy(old, 0, grown, 0, self.capacity * width)?;
                    backend.release(old);
                }
            }
            self.capacity = want;
        }
        let (keys, values) = (self.keys, self.values);
        Ok((
            keys.context("key cache was never allocated")?,
            values.context("value cache was never allocated")?,
        ))
    }

    /// Hands both caches back to the backend and leaves the pair as new.
    pub(crate) fn release(&mut self, backend: &dyn Backend) {
        for slot in [&mut self.keys, &mut self.values] {
            if let Some(buf) = slot.take() {
                backend.release(buf);
            }
        }
        self.capacity = 0;
    }
}

/// Rotary cosines and sines by absolute position, `[positions, rope_dim]`, a
/// row's cosines followed by its sines.
///
/// Built on the host because the language has no sine and the hardware's
/// approximate one loses accuracy across the range a position reaches.
/// Precomputing also makes the table a constant at decode.
pub(crate) struct RopeTable {
    rope_dim: usize,
    freq_base: f32,
    angles: RefCell<Vec<f32>>,
}

impl RopeTable {
    pub(crate) fn new(rope_dim: usize, freq_base: f32) -> RopeTable {
        RopeTable {
            rope_dim,
            freq_base,
            angles: RefCell::new(Vec::new()),
        }
    }

    /// The table, extended to cover `positions` and uploaded. The name carries
    /// its length, so a growth is a new constant rather than a mutation of one
    /// the backend has cached. Superseded copies stay resident, which doubling
    /// bounds at roughly one extra table.
    pub(crate) fn buf(&self, backend: &dyn Backend, positions: usize) -> Result<Buf> {
        let (rope_dim, half) = (self.rope_dim, self.rope_dim / 2);
        let mut table = self.angles.borrow_mut();
        let have = table.len() / rope_dim;
        if have < positions {
            let want = positions.next_power_of_two().max(512);
            table.resize(want * rope_dim, 0.0);
            for p in have..want {
                let row = &mut table[p * rope_dim..][..rope_dim];
                for i in 0..half {
                    let inv_freq = self.freq_base.powf(-(2.0 * i as f32) / rope_dim as f32);
                    let (sin, cos) = (p as f32 * inv_freq).sin_cos();
                    row[i] = cos;
                    row[half + i] = sin;
                }
            }
        }
        let key = format!("rope.{rope_dim}.{}.{}", self.freq_base, table.len());
        backend.constant(&key, &table)
    }
}
