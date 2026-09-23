// The host reference's mixture-of-experts feed-forward.
//
// Experts stay in their file bytes and are decoded one at a time into a
// scratch as a row chooses them: held dense the way the trunk's weights are,
// a model's experts would take four bytes a weight of host memory, several
// times what the file does.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};

use super::{HostBackend, sigmoid, silu};
use crate::backend::{ExpertsBuf, Moe, route};
use crate::experts::ExpertSet;

impl HostBackend {
    pub(super) fn register_experts(&self, key: &str, set: &Arc<ExpertSet>) -> ExpertsBuf {
        if let Some(&buf) = self.expert_keys.borrow().get(key) {
            return buf;
        }
        let mut experts = self.experts.borrow_mut();
        experts.push(Arc::clone(set));
        let buf = ExpertsBuf(experts.len() - 1);
        self.expert_keys.borrow_mut().insert(key.to_string(), buf);
        buf
    }

    pub(super) fn run_moe(&self, req: Moe) -> Result<()> {
        let set = self
            .experts
            .borrow()
            .get(req.experts.0)
            .cloned()
            .context("use of an unknown expert set handle")?;
        let (rows, d, d_ff) = (req.rows, req.d_model, req.d_ff);
        ensure!(
            set.gate.n() == d_ff && set.gate.k() == d && set.up.n() == d_ff && set.up.k() == d,
            "gate and up experts are [{}, {}], the request wants [{d_ff}, {d}]",
            set.gate.n(),
            set.gate.k()
        );
        ensure!(
            set.down.n() == d && set.down.k() == d_ff,
            "down experts are [{}, {}], the request wants [{d}, {d_ff}]",
            set.down.n(),
            set.down.k()
        );
        ensure!(
            set.count() == req.n_expert && req.n_used <= req.n_expert,
            "{} experts held, the request routes {} of {}",
            set.count(),
            req.n_used,
            req.n_expert
        );

        // Inputs copied out first, so the destination can be taken for
        // writing without aliasing any of them.
        let (x, logits, shared) = {
            let slabs = self.slabs.borrow();
            let take = |buf: crate::backend::Buf, len: usize, what: &str| -> Result<Vec<f32>> {
                let src = &slabs[buf.0];
                ensure!(src.len() >= len, "{what} holds {} elements, {len} wanted", src.len());
                Ok(src[..len].to_vec())
            };
            let shared = match req.shared {
                Some((out, gate)) => Some((take(out, rows * d, "shared expert output")?, take(gate, rows, "shared gate")?)),
                None => None,
            };
            (take(req.x, rows * d, "input")?, take(req.logits, rows * req.n_expert, "router logits")?, shared)
        };

        // Gate, up and down in turn through one dense scratch.
        let mut weight = vec![0.0f32; d_ff * d];
        let (mut g, mut u, mut h) = (vec![0.0f32; d_ff], vec![0.0f32; d_ff], vec![0.0f32; d_ff]);
        let mut y = vec![0.0f32; rows * d];
        let mut routes = Vec::with_capacity(rows * req.n_used);
        for r in 0..rows {
            let xr = &x[r * d..(r + 1) * d];
            let yr = &mut y[r * d..(r + 1) * d];
            for (e, w) in route(&logits[r * req.n_expert..(r + 1) * req.n_expert], req.n_used) {
                routes.push(e as f32);
                set.gate.dequantize(e, &mut weight)?;
                for (j, out) in g.iter_mut().enumerate() {
                    *out = dot(&weight[j * d..(j + 1) * d], xr);
                }
                set.up.dequantize(e, &mut weight)?;
                for (j, out) in u.iter_mut().enumerate() {
                    *out = dot(&weight[j * d..(j + 1) * d], xr);
                }
                for ((out, &gate), &up) in h.iter_mut().zip(&g).zip(&u) {
                    *out = silu(gate) * up;
                }
                set.down.dequantize(e, &mut weight)?;
                for (i, out) in yr.iter_mut().enumerate() {
                    *out += w * dot(&weight[i * d_ff..(i + 1) * d_ff], &h);
                }
            }
            if let Some((out, gate)) = &shared {
                let scale = sigmoid(gate[r]);
                for (acc, &v) in yr.iter_mut().zip(&out[r * d..(r + 1) * d]) {
                    *acc += scale * v;
                }
            }
        }

        self.writing(req.dest, |_, dst| {
            ensure!(dst.len() >= rows * d, "moe destination is too small");
            for (acc, &v) in dst.iter_mut().zip(&y) {
                *acc += v;
            }
            Ok(())
        })?;
        if let Some(buf) = req.routes {
            self.writing(buf, |_, dst| {
                ensure!(dst.len() >= routes.len(), "moe routes buffer is too small");
                dst[..routes.len()].copy_from_slice(&routes);
                Ok(())
            })?;
        }
        Ok(())
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
