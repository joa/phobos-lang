use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};

use crate::backend::{
    Backend, Buf, Fused, FusedAttnOut, FusedMix, FusedMlp, FusedProject, HBuf, Plane, ProjRun,
    QAct, QBuf, RawBuf,
};
use crate::quant::Packed;
use crate::{Gguf, TensorInfo};

/// Output columns [`Linear::fuse`] rounds up to: the widest column tile the
/// batched projection has. 8224 becomes 8448, 2.7% more arithmetic against a
/// tile that is 23% faster.
const FUSE_ALIGN: usize = 256;

/// What a model's constants will occupy on the backend once every one of them
/// is resident.
///
/// Entries are keyed the way the backend caches them, so a weight two places
/// reach for, a tied embedding and language-model head, counts once. Only
/// weights that actually go up belong here: a llama embedding table is read row
/// by row on the host and never uploaded, so the walk that fills this skips it.
#[derive(Default)]
pub(crate) struct Uploads {
    uploads: HashMap<String, Upload>,
}

struct Upload {
    bytes: usize,
    /// Held as f32 rather than left quantized, which is what makes a heavily
    /// quantized file want several times its own size on the device.
    dense: bool,
}

impl Uploads {
    fn add(&mut self, key: &str, bytes: usize, dense: bool) {
        self.uploads
            .entry(key.to_string())
            .or_insert(Upload { bytes, dense });
    }

    pub(crate) fn bytes(&self) -> usize {
        self.uploads.values().map(|u| u.bytes).sum()
    }

    /// Of [`Uploads::bytes`], what goes up as f32.
    pub(crate) fn dense_bytes(&self) -> usize {
        self.uploads
            .values()
            .filter(|u| u.dense)
            .map(|u| u.bytes)
            .sum()
    }
}

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
    /// Transposed to `[in, out]` at load, whatever the file held.
    Dense(Vec<f32>),
    /// Left in the file's blocks, `[out, in]`. See [`Packed`].
    Quant(Packed),
}

impl Linear {
    pub(crate) fn load(gguf: &Gguf, name: &str, in_dim: usize, out_dim: usize) -> Result<Linear> {
        let info = gguf
            .tensor(name)
            .with_context(|| format!("missing tensor '{name}'"))?;
        check_dims(info, &[in_dim as u64, out_dim as u64])?;

        // The file's blocks are kept as they are: GGUF already stores them
        // [out, in] with `in` contiguous, which is what every operation here
        // and the four-way byte dot product both want. Nothing is requantized,
        // and a format whose block does not divide the input width falls
        // through to the dense path rather than being split across rows.
        let quant = info
            .ggml_type
            .quant()
            .filter(|q| in_dim.is_multiple_of(q.spec().block));

        let weight = if let Some(quant) = quant {
            let bytes = gguf.tensor_bytes(info)?;
            Weights::Quant(Packed::from_bytes(quant, bytes, in_dim, out_dim)?)
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

    /// What [`Linear::project_shared`] will upload for this weight.
    ///
    /// A weight a kernel unpacks goes up as one quant per element plus its
    /// scales twice, once per block and once transposed per row, the two
    /// layouts the quantized kernels read. A weight a raw kernel decodes goes
    /// up as its file bytes verbatim plus one `f16` header plane, or two for a
    /// format with a minimum term. Anything else goes up as f32, whatever the
    /// file held.
    pub(crate) fn footprint(&self, into: &mut Uploads) {
        let elems = self.in_dim * self.out_dim;
        let (bytes, dense) = match &self.weight {
            Weights::Quant(packed) if packed.has_planes() => {
                let scales = elems / packed.spec().scale_run;
                (elems + 2 * scales * size_of::<f32>(), false)
            }
            Weights::Quant(packed) if packed.has_raw_scales() => {
                let blocks = elems / packed.spec().block;
                let headers = 1 + usize::from(packed.spec().has_min);
                (packed.byte_len() + headers * blocks * size_of::<u16>(), false)
            }
            _ => (elems * size_of::<f32>(), true),
        };
        into.add(&self.key, bytes, dense);
    }

    /// The same projection with its output channels permuted: output `j` of the
    /// result is output `order[j]` of this one. Nothing is requantized, a block
    /// covering inputs of one output, so a row moves whole.
    ///
    /// The rotary layout wants this. ggml rotates either consecutive pairs or
    /// pairs half a head apart and the backend implements only the second, so an
    /// architecture using the first permutes its query and key weights.
    pub(crate) fn reorder_outputs(&self, suffix: &str, order: &[usize]) -> Result<Linear> {
        ensure!(
            order.len() == self.out_dim && order.iter().all(|&j| j < self.out_dim),
            "a reordering of '{}' must be a permutation of its {} outputs",
            self.key,
            self.out_dim
        );

        let (in_dim, out_dim) = (self.in_dim, self.out_dim);
        let weight = match &self.weight {
            Weights::Quant(packed) => Weights::Quant(packed.select_outputs(order)?),
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

    /// Whether [`Linear::fuse`] on these parts stays quantized rather than
    /// falling back to a dense fusion. Requires the plane-tier format and a
    /// uniform quant across every part; `Packed::stack` on the raw tier
    /// (K-quant, IQ) is unexercised, so that tier always declines.
    pub(crate) fn should_fuse(parts: &[&Linear]) -> bool {
        match packed_parts(parts) {
            None => true,
            Some(ps) => ps[0].has_planes() && ps.iter().all(|p| p.quant() == ps[0].quant()),
        }
    }

    /// Stacks weights that share an input into one projection, replacing a
    /// launch per narrow output (q/k/v, gate/up, a delta net's four parts)
    /// with one. Stacking along the output axis appends whole blocks and
    /// requantizes nothing.
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

        // Blocks only stack if every part is in the same format. A K-quant file
        // is routinely mixed, leaving its more sensitive tensors wider, and
        // there the fused weight has to go dense.
        let packed = packed_parts(parts);
        let uniform = packed
            .as_ref()
            .is_some_and(|ps| ps.iter().all(|p| p.quant() == ps[0].quant()));

        let weight = if uniform {
            Weights::Quant(Packed::stack(
                &packed.expect("uniform implies packed"),
                out_dim,
            )?)
        } else {
            // Dense weights are held [in, out], so a row of the fused weight is
            // each part's row end to end.
            let materialized: Vec<Cow<[f32]>> = parts
                .iter()
                .map(|p| match &p.weight {
                    Weights::Dense(data) => Cow::Borrowed(data.as_slice()),
                    Weights::Quant(packed) => Cow::Owned(packed.dense()),
                })
                .collect();

            let mut dense = vec![0.0f32; in_dim * out_dim];
            for i in 0..in_dim {
                let mut at = 0;
                for (part, data) in parts.iter().zip(&materialized) {
                    let row = &data[i * part.out_dim..(i + 1) * part.out_dim];
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

    /// Write output row `index` of this weight into `out`, dequantizing it. Only
    /// the embedding table uses this: reading a token's row straight out of the
    /// quantized bytes removes the f32 copy without changing the result.
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
            Weights::Quant(packed) => packed.row_into(index, out)?,
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

    /// [`Linear::project_into`] against an activation quantized already.
    pub(crate) fn project_into_act(
        &self,
        backend: &dyn Backend,
        x: Buf,
        act: QAct,
        rows: usize,
        out: Buf,
    ) -> Result<()> {
        self.project_shared(backend, x, Some(act), rows, out)
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
        if self.is_quantized() {
            let act = backend.quantize_act(x, rows, self.in_dim)?;
            return self.add_into_act(backend, x, act, rows, dest);
        }
        self.add_dense(backend, x, rows, dest)
    }

    /// [`Linear::add_into`], but letting a backend fuse the activation's
    /// quantization into the accumulating contraction itself, one kernel
    /// instead of two. `x` is unquantized and not yet reduced; the fused
    /// kernel does that itself if it takes this at all. Falls back to
    /// [`Linear::add_into`] otherwise.
    pub(crate) fn add_projected(
        &self,
        backend: &dyn Backend,
        x: Buf,
        rows: usize,
        dest: Buf,
    ) -> Result<()> {
        if rows == 1 && self.is_quantized() {
            let fused = backend.fused_attn_out(FusedAttnOut {
                x,
                width: self.in_dim,
                w: self.quantized(backend)?,
                d_model: self.out_dim,
                dest,
            })?;
            if fused {
                return Ok(());
            }
        }
        self.add_into(backend, x, rows, dest)
    }

    /// [`Linear::add_into`] against an activation quantized already. `x` is
    /// what `act` quantizes, for a weight that has to be contracted densely.
    pub(crate) fn add_into_act(
        &self,
        backend: &dyn Backend,
        x: Buf,
        act: QAct,
        rows: usize,
        dest: Buf,
    ) -> Result<()> {
        if !self.is_quantized() {
            return self.add_dense(backend, x, rows, dest);
        }
        let w = self.quantized(backend)?;
        backend
            .matmul_quant_add(act, rows, self.in_dim, w, self.out_dim, dest)
            .with_context(|| format!("residual matmul for '{}'", self.key))
    }

    /// The accumulating projection as a separate pass, which is all a dense
    /// weight has: [`Backend::matmul`] does not add into its destination.
    fn add_dense(&self, backend: &dyn Backend, x: Buf, rows: usize, dest: Buf) -> Result<()> {
        let out = self.forward(backend, x, rows)?;
        backend.add_into(dest, out)?;
        backend.release(out);
        Ok(())
    }

    /// The normalization ahead of this projection and the projection itself
    /// as one kernel, each run of the output landing where the caller wants
    /// it, and `mix` continuing into the delta net's convolution and gates.
    /// Whatever comes back unset is the caller's to launch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_fused(
        &self,
        backend: &dyn Backend,
        x: Buf,
        gain: Buf,
        eps: f32,
        rows: usize,
        runs: &[ProjRun],
        mix: Option<FusedMix>,
    ) -> Result<Fused> {
        if rows != 1 || !self.is_quantized() {
            return Ok(Fused::default());
        }
        backend.fused_project(FusedProject {
            x,
            d_model: self.in_dim,
            gain,
            eps,
            w: self.quantized(backend)?,
            out_dim: self.out_dim,
            runs,
            mix,
        })
    }

    /// Whether the quantized contraction applies, which needs both a quantized
    /// weight and a kernel that unpacks its format.
    fn is_quantized(&self) -> bool {
        matches!(&self.weight, Weights::Quant(packed) if packed.has_planes())
    }

    /// This weight uploaded in its quantized form, for a caller that contracts
    /// against it itself rather than through one of the projections here.
    pub(crate) fn quantized(&self, backend: &dyn Backend) -> Result<QBuf> {
        let Weights::Quant(packed) = &self.weight else {
            bail!("'{}' is not held in quantized form", self.key);
        };
        backend.constant_quant(&self.key, packed)
    }

    /// Whether the raw-kernel contraction applies: a quantized weight in a
    /// format [`Backend::constant_quant`] does not unpack, but a raw kernel
    /// decodes from its own bytes.
    fn is_raw(&self) -> bool {
        matches!(&self.weight, Weights::Quant(packed) if packed.has_raw_scales())
    }

    /// This weight uploaded in its raw block bytes, for a caller that
    /// contracts against it itself.
    pub(crate) fn raw(&self, backend: &dyn Backend) -> Result<RawBuf> {
        let Weights::Quant(packed) = &self.weight else {
            bail!("'{}' is not held in quantized form", self.key);
        };
        backend.constant_raw(&self.key, packed)
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
        if self.is_quantized() {
            let w = self.quantized(backend)?;
            return match act {
                Some(act) => backend.matmul_quant_act(act, rows, self.in_dim, w, self.out_dim, out),
                None => backend.matmul_quant(x, rows, self.in_dim, w, self.out_dim, out),
            }
            .with_context(|| format!("matmul for '{}'", self.key));
        }

        if self.is_raw() {
            // A raw kernel decodes to f32 and multiplies against the
            // activation directly, so a caller's pre-quantized `act` (built
            // for the int8 contraction above) has nothing to offer it.
            let w = self.raw(backend)?;
            return backend
                .matmul_raw(x, rows, self.in_dim, w, self.out_dim, out)
                .with_context(|| format!("matmul for '{}'", self.key));
        }

        // Either the file was dense or no kernel unpacks its format, in which
        // case the weight is decoded once at upload and contracted densely.
        let w = match &self.weight {
            Weights::Dense(data) => backend.constant(&self.key, data)?,
            Weights::Quant(packed) => backend.constant_lazy(&self.key, &|| Ok(packed.dense()))?,
        };
        backend
            .matmul(x, rows, self.in_dim, w, self.out_dim, out)
            .with_context(|| format!("matmul for '{}'", self.key))
    }
}

/// Each part's [`Packed`] block, or `None` if any part is still dense.
fn packed_parts<'a>(parts: &[&'a Linear]) -> Option<Vec<&'a Packed>> {
    parts
        .iter()
        .map(|p| match &p.weight {
            Weights::Quant(packed) => Some(packed),
            Weights::Dense(_) => None,
        })
        .collect()
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

    pub(crate) fn footprint(&self, into: &mut Uploads) {
        into.add(&self.key, self.data.len() * size_of::<f32>(), true);
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

/// Gate and up, fused into one launch when [`Linear::should_fuse`] allows it
/// and run as two ordinary projections otherwise.
enum GateUp {
    /// Stacked: they read the same row, so one launch over a doubly wide
    /// output replaces two over half of it.
    Fused(Linear),
    Split { gate: Linear, up: Linear },
}

/// The SwiGLU feed-forward every architecture here ends a block with.
pub(crate) struct Ffn {
    gate_up: GateUp,
    down: Linear,
}

impl Ffn {
    pub(crate) fn load(gguf: &Gguf, prefix: &str, d_model: usize, d_ff: usize) -> Result<Ffn> {
        let gate = Linear::load(gguf, &format!("{prefix}.ffn_gate.weight"), d_model, d_ff)?;
        let up = Linear::load(gguf, &format!("{prefix}.ffn_up.weight"), d_model, d_ff)?;
        let gate_up = if Linear::should_fuse(&[&gate, &up]) {
            GateUp::Fused(Linear::fuse(&[&gate, &up])?)
        } else {
            GateUp::Split { gate, up }
        };
        Ok(Ffn {
            gate_up,
            down: Linear::load(gguf, &format!("{prefix}.ffn_down.weight"), d_ff, d_model)?,
        })
    }

    pub(crate) fn footprint(&self, into: &mut Uploads) {
        match &self.gate_up {
            GateUp::Fused(gate_up) => gate_up.footprint(into),
            GateUp::Split { gate, up } => {
                gate.footprint(into);
                up.footprint(into);
            }
        }
        self.down.footprint(into);
    }

    /// The normalization and the whole of [`Ffn::forward`] as one kernel, if
    /// the backend has one. `false` leaves the caller to take the usual
    /// path; a split gate/up always does, having no single weight to hand
    /// the fused kernel.
    pub(crate) fn forward_fused(
        &self,
        backend: &dyn Backend,
        x: Buf,
        gain: Buf,
        eps: f32,
        rows: usize,
    ) -> Result<bool> {
        let GateUp::Fused(gate_up) = &self.gate_up else {
            return Ok(false);
        };
        if rows != 1 || !gate_up.is_quantized() || !self.down.is_quantized() {
            return Ok(false);
        }
        backend.fused_mlp(FusedMlp {
            x,
            d_model: gate_up.in_dim,
            d_ff: self.down.in_dim,
            gain,
            eps,
            gate_up: gate_up.quantized(backend)?,
            down: self.down.quantized(backend)?,
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
        let joined = backend.alloc(rows * width)?;
        let dense = |buf| Plane { buf, offset: 0, pitch: width };

        // Either a fused projection's two windows, or two ordinary
        // projections' own dense buffers -- either way this ends as a
        // (gate, up) pair of planes and the buffers to release once the
        // SwiGLU has read them.
        let (gate_p, up_p, release): (Plane, Plane, [Buf; 2]) = match &self.gate_up {
            GateUp::Fused(gate_up) => {
                let both = gate_up.forward_act(backend, x, act, rows)?;
                // Past one row the two halves interleave, so the SwiGLU reads
                // them where they lie rather than pulling them apart first.
                let stacked = |offset| Plane { buf: both, offset, pitch: 2 * width };
                (stacked(0), stacked(width), [both, both])
            }
            GateUp::Split { gate, up } => {
                let gate_buf = gate.forward_act(backend, x, act, rows)?;
                let up_buf = up.forward_act(backend, x, act, rows)?;
                (dense(gate_buf), dense(up_buf), [gate_buf, up_buf])
            }
        };

        if rows == 1 {
            // One launch for the gate, the product, and the quantized copy.
            let act =
                backend.swiglu_q(gate_p.buf, gate_p.offset, up_p.buf, up_p.offset, joined, width)?;
            release_once(backend, release);
            self.down.add_into_act(backend, joined, act, rows, dest)?;
            backend.release(joined);
            return Ok(());
        }
        backend.swiglu_planes(gate_p, up_p, joined, rows, width)?;
        release_once(backend, release);
        self.down.add_into(backend, joined, rows, dest)?;
        backend.release(joined);
        Ok(())
    }
}

/// Releases `[a, b]`, once each even when `a == b` (the fused case, where
/// gate and up are two windows of the same buffer).
fn release_once(backend: &dyn Backend, [a, b]: [Buf; 2]) {
    backend.release(a);
    if b != a {
        backend.release(b);
    }
}

/// An attention block's key and value caches, `[capacity, n_kv * head_dim]`
/// each, in f16. Backend-resident: at a few thousand positions the cache is the
/// largest thing in the block, and see [`HBuf`] for why it is held narrow.
#[derive(Default)]
pub(crate) struct KvCache {
    keys: Option<HBuf>,
    values: Option<HBuf>,
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
    ) -> Result<(HBuf, HBuf)> {
        if self.capacity < total {
            let want = total.next_power_of_two().max(64);
            for slot in [&mut self.keys, &mut self.values] {
                let grown = backend.alloc_h(want * width)?;
                if let Some(old) = slot.replace(grown) {
                    backend.copy_h(old, 0, grown, 0, self.capacity * width)?;
                    backend.release_h(old);
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
                backend.release_h(buf);
            }
        }
        self.capacity = 0;
    }
}

/// Rotary cosines and sines by absolute position, `[positions, rope_dim]`, a
/// row's cosines followed by its sines. Built on the host because the language
/// has no sine and the hardware's approximate one loses accuracy across the
/// range a position reaches; precomputing also makes it a constant at decode.
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

    /// What the table costs once a sequence has reached `positions`. Doubling
    /// leaves every superseded copy resident, and those sum to just under the
    /// final one, so the pair is what a run ends up holding.
    pub(crate) fn footprint(&self, into: &mut Uploads, positions: usize) {
        let rows = positions.next_power_of_two().max(512);
        let key = format!("rope.{}.{}", self.rope_dim, self.freq_base);
        into.add(&key, 2 * rows * self.rope_dim * size_of::<f32>(), true);
    }
}
