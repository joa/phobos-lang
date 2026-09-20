use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use crate::backend::{
    Backend, Buf, Fused, FusedAttnOut, FusedMix, FusedProject, HADAMARD_BLOCK, HBuf, HeadPerm,
    ProjRun, ProjWeight, QAct, QBuf, RawBuf, takes_rows,
};
use crate::quant::{Packed, Quant};
use crate::{Gguf, TensorInfo};

mod ffn;
mod moe;
mod shared;

pub(crate) use ffn::Ffn;
pub(crate) use moe::MoeFfn;
pub(crate) use shared::Shared;

/// Output columns [`Linear::fuse`] rounds up to: the widest column tile the
/// batched projection has.
const FUSE_ALIGN: usize = 256;

/// What a model's constants will occupy on the backend once every one of them
/// is resident.
///
/// Entries are keyed the way the backend caches them, so a tied embedding and
/// LM head count once. Only weights that actually go up belong here: a llama
/// embedding table is read row by row on the host and never uploaded, so the
/// walk that fills this skips it.
#[derive(Default)]
pub(crate) struct Uploads {
    uploads: HashMap<String, Upload>,
}

struct Upload {
    bytes: usize,
    /// Held as f32 rather than left quantized: a heavily quantized file can
    /// want several times its own size on the device.
    dense: bool,
    /// Not resident: a set of experts the backend streams from the file's
    /// bytes, holding whatever share of them it has room for. Counted in
    /// [`Uploads::streamed_bytes`] and nowhere else.
    streamed: bool,
}

impl Uploads {
    fn add(&mut self, key: &str, bytes: usize, dense: bool) {
        self.uploads
            .entry(key.to_string())
            .or_insert(Upload { bytes, dense, streamed: false });
    }

    fn add_streamed(&mut self, key: &str, bytes: usize) {
        self.uploads
            .entry(key.to_string())
            .or_insert(Upload { bytes, dense: false, streamed: true });
    }

    /// Bytes of every weight the backend keeps resident.
    pub(crate) fn bytes(&self) -> usize {
        self.uploads.values().filter(|u| !u.streamed).map(|u| u.bytes).sum()
    }

    /// Bytes of the weights that stream, as the file holds them.
    pub(crate) fn streamed_bytes(&self) -> usize {
        self.uploads.values().filter(|u| u.streamed).map(|u| u.bytes).sum()
    }

    /// Expert sets that stream.
    pub(crate) fn streamed_sets(&self) -> usize {
        self.uploads.values().filter(|u| u.streamed).count()
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
    fold: Option<Box<Fold>>,
}

/// A weight stored Hadamard-folded ([`crate::hadamard`]): an input one,
/// whose activation goes through the transform ahead of every projection,
/// or a lookup table, whose rows are restored after the lookup.
#[derive(Clone)]
enum Fold {
    Input { signs: Arc<Vec<f32>>, perm: Option<HeadPerm> },
    Table { signs: Arc<Vec<f32>> },
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

        // GGUF already stores blocks [out, in] with `in` contiguous, the
        // layout every operation here and the four-way byte dot product
        // want. Nothing is requantized: a format whose block does not divide
        // the input width falls through to the dense path instead of being
        // split across rows.
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

        let fold = match gguf.folding() {
            Some(f) if f.folds(name) => Some(Box::new(Fold::Input { signs: f.signs(in_dim)?, perm: None })),
            Some(f) if f.restores(name) => Some(Box::new(Fold::Table { signs: f.signs(in_dim)? })),
            _ => None,
        };
        if let Some(f) = gguf.folding().filter(|_| fold.is_some()) {
            ensure!(
                f.block == HADAMARD_BLOCK,
                "'{name}' is folded in blocks of {}; only {HADAMARD_BLOCK} is implemented",
                f.block
            );
        }

        Ok(Linear {
            weight,
            in_dim,
            out_dim,
            key: name.to_string(),
            fold,
        })
    }

    /// For a folded input weight, regroup the activation's heads ahead of the
    /// transform; see [`HeadPerm`].
    pub(crate) fn regroup_heads(&mut self, regroup: HeadPerm) {
        if let Some(Fold::Input { perm, .. }) = self.fold.as_deref_mut() {
            *perm = Some(regroup);
        }
    }

    /// Whether this weight's input has to go through the Hadamard transform.
    pub(crate) fn folded(&self) -> bool {
        matches!(self.fold.as_deref(), Some(Fold::Input { .. }))
    }

    /// The key a dense weight goes up under: its own, or for one narrow
    /// enough for [`Backend::matmul_rows`], that of the layout it reads.
    fn dense_key(&self) -> String {
        match &self.weight {
            Weights::Dense(_) if takes_rows(self.in_dim, self.out_dim) => format!("{}.rows", self.key),
            _ => self.key.clone(),
        }
    }

    /// A dense `[in, out]` weight back in the file's `[out, in]` order.
    fn file_rows(&self, data: &[f32]) -> Vec<f32> {
        let mut rows = vec![0.0f32; data.len()];
        for (i, row) in data.chunks_exact(self.out_dim).enumerate() {
            for (o, &v) in row.iter().enumerate() {
                rows[o * self.in_dim + i] = v;
            }
        }
        rows
    }

    /// What [`Linear::project_shared`] will upload for this weight.
    ///
    /// A quantized weight a kernel unpacks goes up as one quant per element
    /// plus its scales twice, per block and per row transposed, the two
    /// layouts the quantized kernels read. A weight a raw kernel decodes goes
    /// up as its file bytes plus an `f16` header plane per scale term.
    /// Anything else goes up as f32.
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
        into.add(&self.dense_key(), bytes, dense);
    }

    /// The same projection with its output channels permuted: output `j` of the
    /// result is output `order[j]` of this one. Nothing is requantized: a block
    /// covers one output's inputs, so a row moves whole.
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
            fold: self.fold.clone(),
        })
    }

    /// Whether [`Linear::fuse`] on these parts stays quantized rather than
    /// falling back to a dense fusion. Requires a uniform quant across every
    /// part, in the plane tier or PTQ1_0 (`Packed::stack` on the rest of the
    /// raw tier is unexercised), and parts that are either all unfolded or
    /// all read through the same transform, which the stack then keeps.
    /// Parts held dense fuse dense only when every part is: a quantized
    /// projection fused with an F32 one would go up as f32 many times its
    /// size (a delta net's qkv and gate beside its alpha and beta, 98 MiB a
    /// block where the file holds 27), and reading that back every token
    /// costs more than the launch the fusion saves.
    pub(crate) fn should_fuse(parts: &[&Linear]) -> bool {
        if !Linear::same_fold(parts) {
            return false;
        }
        match packed_parts(parts) {
            None => parts
                .iter()
                .all(|p| p.fold.is_none() && matches!(p.weight, Weights::Dense(_))),
            Some(ps) => {
                (ps[0].has_planes() || ps[0].quant() == Quant::PTQ1_0)
                    && ps.iter().all(|p| p.quant() == ps[0].quant())
            }
        }
    }

    /// Whether every part reads its input through the same transform, or
    /// none does.
    fn same_fold(parts: &[&Linear]) -> bool {
        let folds = |p: &Linear| p.fold.as_deref().map(|f| matches!(f, Fold::Input { .. }));
        parts.iter().all(|p| p.rotation() == parts[0].rotation() && folds(p) == folds(parts[0]))
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
        ensure!(
            Linear::same_fold(parts),
            "fusing weights read through different transforms ('{key}')"
        );

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
            fold: first.fold.clone(),
        })
    }

    /// Dense parts sharing an input, stacked along the output axis into one
    /// projection narrow enough for [`Backend::matmul_rows`], unpadded; `None`
    /// where any part is held otherwise or the stack is too wide.
    pub(crate) fn stack(parts: &[&Linear]) -> Option<Linear> {
        let in_dim = parts.first()?.in_dim;
        let out_dim = parts.iter().map(|p| p.out_dim).sum();
        let sources = parts
            .iter()
            .map(|p| match &p.weight {
                Weights::Dense(data) if p.in_dim == in_dim && p.fold.is_none() => Some((data, p.out_dim)),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .filter(|_| takes_rows(in_dim, out_dim))?;
        let mut data = vec![0.0f32; in_dim * out_dim];
        for (i, row) in data.chunks_exact_mut(out_dim).enumerate() {
            let mut at = 0;
            for (src, width) in &sources {
                row[at..at + width].copy_from_slice(&src[i * width..(i + 1) * width]);
                at += width;
            }
        }
        let key = parts.iter().map(|p| p.key.as_str()).collect::<Vec<_>>().join("+");
        Some(Linear { weight: Weights::Dense(data), in_dim, out_dim, key, fold: None })
    }

    /// The dense `[in, out]` weight as the backend holds it for
    /// [`Backend::matmul`], or `None` for a weight held any other way.
    pub(crate) fn dense_plain(&self, backend: &dyn Backend) -> Result<Option<Buf>> {
        match &self.weight {
            Weights::Dense(data) if self.fold.is_none() && !takes_rows(self.in_dim, self.out_dim) => {
                backend.constant(&self.key, data).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// Write output row `index` of this weight into `out`, dequantizing it.
    /// Only the embedding table uses this, reading a token's row straight out
    /// of the quantized bytes instead of keeping a dense f32 copy.
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
        if let Some(Fold::Table { signs }) = self.fold.as_deref() {
            crate::hadamard::restore_row(out, signs, HADAMARD_BLOCK);
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
        if self.folded() {
            let input = self.share(backend, x, None, rows)?;
            let done = self.add_plain(backend, input.x, input.act, rows, dest);
            input.release(backend);
            return done;
        }
        self.add_plain(backend, x, None, rows, dest)
    }

    /// [`Linear::add_into`] on an input already carried through any
    /// transform, and `act` its quantized copy where there is one.
    fn add_plain(&self, backend: &dyn Backend, x: Buf, act: Option<QAct>, rows: usize, dest: Buf) -> Result<()> {
        if self.is_quantized() {
            let act = act.map_or_else(|| backend.quantize_act(x, rows, self.in_dim), Ok)?;
            return self.add_act_plain(backend, x, act, rows, dest);
        }
        self.add_dense(backend, x, act, rows, dest)
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
        if rows == 1 && self.is_quantized() && self.fold.is_none() {
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
        act: Option<QAct>,
        rows: usize,
        dest: Buf,
    ) -> Result<()> {
        // `act` quantizes `x` as it is, which a folded weight cannot read.
        match act {
            Some(act) if !self.folded() => self.add_act_plain(backend, x, act, rows, dest),
            _ => self.add_into(backend, x, rows, dest),
        }
    }

    fn add_act_plain(
        &self,
        backend: &dyn Backend,
        x: Buf,
        act: QAct,
        rows: usize,
        dest: Buf,
    ) -> Result<()> {
        if !self.is_quantized() {
            return self.add_dense(backend, x, None, rows, dest);
        }
        let w = self.quantized(backend)?;
        backend
            .matmul_quant_add(act, rows, self.in_dim, w, self.out_dim, dest)
            .with_context(|| format!("residual matmul for '{}'", self.key))
    }

    /// The accumulating projection as a separate pass, which is all a dense
    /// weight has: [`Backend::matmul`] does not add into its destination.
    fn add_dense(&self, backend: &dyn Backend, x: Buf, act: Option<QAct>, rows: usize, dest: Buf) -> Result<()> {
        let out = backend.alloc(rows * self.out_dim)?;
        self.project_plain(backend, x, act, rows, out)?;
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
        if rows != 1 {
            return Ok(Fused::default());
        }
        let Some(w) = self.proj_weight(backend)? else {
            return Ok(Fused::default());
        };
        backend.fused_project(FusedProject {
            x,
            d_model: self.in_dim,
            gain,
            eps,
            weights: &[(w, self.out_dim)],
            runs,
            mix,
        })
    }

    /// This weight in the form a fused projection contracts against, or
    /// `None` for one held densely.
    pub(crate) fn proj_weight(&self, backend: &dyn Backend) -> Result<Option<ProjWeight>> {
        let Weights::Quant(packed) = &self.weight else {
            return Ok(None);
        };
        if self.fold.is_some() {
            return Ok(None);
        }
        if packed.has_planes() {
            return Ok(Some(ProjWeight::Q8(backend.constant_quant(&self.key, packed)?)));
        }
        if packed.has_raw_scales() {
            return Ok(Some(ProjWeight::Raw(backend.constant_raw(&self.key, packed)?, packed.quant())));
        }
        Ok(None)
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

    /// The raw format this weight is held in, where a raw kernel decodes it.
    pub(crate) fn raw_quant(&self) -> Option<Quant> {
        match &self.weight {
            Weights::Quant(packed) if packed.has_raw_scales() => Some(packed.quant()),
            _ => None,
        }
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
        act: Option<QAct>,
        rows: usize,
    ) -> Result<Buf> {
        let out = backend.alloc(rows * self.out_dim)?;
        self.project_shared(backend, x, act, rows, out)?;
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
        // A caller's `act` quantizes the untransformed input, so a folded
        // weight quantizes its own.
        if self.folded() {
            let input = self.share(backend, x, None, rows)?;
            let done = self.project_plain(backend, input.x, input.act, rows, out);
            input.release(backend);
            return done;
        }
        self.project_plain(backend, x, act, rows, out)
    }

    fn project_plain(
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
            // A raw kernel that decodes to f32 has no use for a caller's
            // pre-quantized `act`; one that contracts on the integer tensor
            // cores does, and it is cheaper to hand it over than to have the
            // backend quantize the same rows again per weight.
            let w = self.raw(backend)?;
            return match act {
                Some(act) => backend.matmul_raw_act(act, x, rows, self.in_dim, w, self.out_dim, out),
                None => backend.matmul_raw(x, rows, self.in_dim, w, self.out_dim, out),
            }
            .with_context(|| format!("matmul for '{}'", self.key));
        }

        if let Weights::Dense(data) = &self.weight
            && takes_rows(self.in_dim, self.out_dim)
        {
            let w = backend.constant_lazy(&self.dense_key(), &|| Ok(self.file_rows(data)))?;
            return backend
                .matmul_rows(x, rows, self.in_dim, w, self.out_dim, out)
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

/// An attention block's key and value caches, `[capacity, n_kv * head_dim]`
/// each, in f16 and backend-resident. See [`HBuf`] for why it is held narrow.
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

    /// What the table costs once a sequence has reached `positions`: doubling
    /// leaves superseded copies resident, so a run ends up holding about
    /// twice the final table.
    pub(crate) fn footprint(&self, into: &mut Uploads, positions: usize) {
        let rows = positions.next_power_of_two().max(512);
        let key = format!("rope.{}.{}", self.rope_dim, self.freq_base);
        into.add(&key, 2 * rows * self.rope_dim * size_of::<f32>(), true);
    }
}
