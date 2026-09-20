// The mixture-of-experts feed-forward: a router, the experts it chooses
// among, and a shared expert every row goes through.

use std::sync::Arc;

use anyhow::{Result, ensure};

use crate::Gguf;
use crate::backend::{Backend, Buf, Moe};
use crate::experts::ExpertSet;

use super::{Ffn, Gain, Linear, Shared, Uploads};

/// A block's routed feed-forward.
///
/// The router and the shared expert are ordinary weights, resident wherever
/// the backend keeps its constants. The experts are not: they stay in the
/// file's bytes and reach the backend through [`Backend::constant_experts`],
/// which is free to hold as many or as few of them as it can.
pub(crate) struct MoeFfn {
    /// `[d_model, n_expert]`, F32 in the file.
    router: Linear,
    experts: Arc<ExpertSet>,
    shared: Ffn,
    /// The shared expert's gate, a `[d_model]` vector the row is dotted with.
    shared_gate: Gain,
    /// What the expert set registers under.
    key: String,
    n_used: usize,
}

impl MoeFfn {
    pub(crate) fn load(
        gguf: &Gguf,
        prefix: &str,
        d_model: usize,
        n_expert: usize,
        n_used: usize,
        d_expert: usize,
        d_shared: usize,
    ) -> Result<MoeFfn> {
        ensure!(
            0 < n_used && n_used <= n_expert,
            "{prefix} routes to {n_used} of {n_expert} experts"
        );
        let router = Linear::load(gguf, &format!("{prefix}.ffn_gate_inp.weight"), d_model, n_expert)?;
        ensure!(!router.folded(), "a Hadamard-folded router is not supported");
        Ok(MoeFfn {
            router,
            experts: ExpertSet::load(gguf, prefix, n_expert, d_model, d_expert)?,
            shared: Ffn::load_suffixed(gguf, prefix, "_shexp", d_model, d_shared)?,
            shared_gate: Gain::load(gguf, &format!("{prefix}.ffn_gate_inp_shexp.weight"), d_model)?,
            key: format!("{prefix}.experts"),
            n_used,
        })
    }

    pub(crate) fn footprint(&self, into: &mut Uploads) {
        self.router.footprint(into);
        self.shared.footprint(into);
        self.shared_gate.footprint(into);
        into.add_streamed(&self.key, self.experts.byte_len());
    }

    /// Hands the expert set to the backend, once; every block does this
    /// ahead of the first pass so a backend laying out a cache sees them
    /// all. Later calls are a lookup.
    pub(crate) fn register(&self, backend: &dyn Backend) -> Result<()> {
        backend.constant_experts(&self.key, &self.experts).map(|_| ())
    }

    /// The weight that reads the block's input; see [`Linear::share`].
    pub(crate) fn input(&self) -> &Linear {
        &self.router
    }

    /// The router alone, on an already normalized `x`, into `logits`
    /// (`[rows, n_expert]`): for a caller asking what this block would
    /// choose for a row without running it.
    pub(crate) fn router_into(
        &self,
        backend: &dyn Backend,
        x: Buf,
        rows: usize,
        logits: Buf,
    ) -> Result<()> {
        self.router.project_into(backend, x, rows, logits)
    }

    /// The whole feed-forward added into `dest`: router, chosen experts,
    /// gated shared expert. `routes`, if given, receives each row's chosen
    /// expert ids as [`Moe::routes`] describes.
    pub(crate) fn forward(
        &self,
        backend: &dyn Backend,
        x: Shared,
        rows: usize,
        dest: Buf,
        routes: Option<Buf>,
    ) -> Result<()> {
        let d_model = self.router.in_dim;
        let logits = self.router.forward_shared(backend, x, rows)?;

        // The shared expert adds into a zeroed row of its own, since the
        // combine scales it before it reaches the residual.
        let shared_out = backend.zeroed(rows * d_model)?;
        self.shared.forward(backend, x, rows, shared_out)?;
        let gate = backend.alloc(rows)?;
        backend.matmul(x.x, rows, d_model, self.shared_gate.buf(backend)?, 1, gate)?;

        backend.moe(Moe {
            x: x.x,
            act: x.act,
            rows,
            d_model,
            d_ff: self.experts.gate.n(),
            logits,
            n_expert: self.experts.count(),
            n_used: self.n_used,
            experts: backend.constant_experts(&self.key, &self.experts)?,
            shared: Some((shared_out, gate)),
            dest,
            routes,
        })?;
        for buf in [logits, shared_out, gate] {
            backend.release(buf);
        }
        Ok(())
    }
}
