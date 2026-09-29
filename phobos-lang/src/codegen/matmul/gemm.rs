// The register matmul in three parts: seed the accumulators before the
// k-loop, carry them through it, and drain them after it. Keeping the parts
// separate lets the caller emit the loop bounds and the epilogue operands in
// between.

use super::*;

use crate::codegen::lower::Lowered;
use crate::shape;
use crate::ir::{Ir, OpId, ValueId};

/// Which of the three paths runs a register matmul, with its tiling.
#[derive(Clone, Copy, Debug)]
pub(in crate::codegen) enum GemmPath {
    /// Vector contractions in registers: each lane owns a tm x tn sub-tile
    /// on an lm x ln lane grid.
    Reg { tm: i64, tn: i64, lm: i64, ln: i64 },
    /// wmma fragments on a wm x wn warp grid.
    Wmma { wm: i64, wn: i64 },
    /// mma.sync fragments on a wm x wn warp grid.
    MmaSync { wm: i64, wn: i64 },
}

/// The shape decisions of one register matmul, made before anything is
/// emitted.
#[derive(Clone, Copy, Debug)]
pub(in crate::codegen) struct GemmPlan<'c> {
    pub(in crate::codegen) m: i64,
    pub(in crate::codegen) n: i64,
    pub(in crate::codegen) kk: i64,
    pub(in crate::codegen) acc_elem: Type<'c>,
    pub(in crate::codegen) path: GemmPath,
}

/// The warp's origin, computed by the k-loop and used by the epilogue:
/// (tid, w, wt, m0, n0), as [`Codegen::warp_block_origin`] returns it.
pub(in crate::codegen) type GemmOrigin<'c> = (
    Value<'c, 'c>,
    Value<'c, 'c>,
    Value<'c, 'c>,
    Value<'c, 'c>,
    Value<'c, 'c>,
);

/// The accumulators passed between the three parts.
#[derive(Clone)]
pub(in crate::codegen) struct GemmAcc<'c> {
    pub(in crate::codegen) plan: GemmPlan<'c>,
    pub(in crate::codegen) regs: Vec<Value<'c, 'c>>,
    /// Set by the loop.
    pub(in crate::codegen) origin: Option<GemmOrigin<'c>>,
}

/// The two operand slices of the k-loop.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::codegen) enum GemmOperand {
    A,
    B,
}

/// Where the k-loop's operand slices come from.
pub(in crate::codegen) enum GemmSource<'a> {
    /// The body of the graph's loop: its iv, the ops before the `gemm_dot`,
    /// and the dot's two slice operands, each defined by one of those ops.
    ///
    /// The a slice's ops come first, then the b slice's. Neither uses the
    /// other's values, so each can be lowered on its own.
    Graph {
        ir: &'a Ir,
        iv: ValueId,
        prefix: &'a [OpId],
        a: ValueId,
        b: ValueId,
    },
}

/// One coefficient of the epilogue's scaling.
#[derive(Clone, Copy)]
pub(in crate::codegen) enum GemmScale<'c> {
    /// A value already emitted.
    Value(Value<'c, 'c>),
    /// The f32 constant one.
    One,
}

impl<'c> Codegen<'c> {
    /// Decides the path and its tiling for an [m, n] accumulator over a k
    /// extent of `kk`.
    pub(in crate::codegen) fn gemm_plan(
        &self,
        m: i64,
        n: i64,
        kk: i64,
        acc_elem: Type<'c>,
    ) -> Result<GemmPlan<'c>> {
        let path = if let Some((wm, wn)) = self.has_wmma()
            .then(|| shape::wmma_plan(m, n, kk, self.cta_threads))
            .flatten() {
            if self.has_mma_sync() {
                GemmPath::MmaSync { wm, wn }
            } else {
                GemmPath::Wmma { wm, wn }
            }
        } else {
            let (tm, tn) = shape::sub_tile(m, n, self.cta_threads);
            let (tiles_m, tiles_n) = (m / tm, n / tn);
            let (lm, ln) = shape::lane_grid(tiles_m, tiles_n, tm, tn)
                .ok_or_else(|| anyhow!("matmul fusion without a lane grid"))?;
            GemmPath::Reg { tm, tn, lm, ln }
        };
        Ok(GemmPlan {
            m,
            n,
            kk,
            acc_elem,
            path,
        })
    }

    /// Seeds the accumulators from the init scalar.
    pub(in crate::codegen) fn gemm_seed(
        &mut self,
        block: &Block<'c>,
        plan: GemmPlan<'c>,
        init: Value<'c, 'c>,
    ) -> Result<GemmAcc<'c>> {
        let regs = match plan.path {
            GemmPath::Reg { .. } => self.reg_seed(block, &plan, init)?,
            GemmPath::Wmma { .. } => self.wmma_seed(block, &plan, init)?,
            GemmPath::MmaSync { .. } => self.mma_sync_seed(block, &plan, init)?,
        };
        Ok(GemmAcc {
            plan,
            regs,
            origin: None,
        })
    }

    /// Runs the k-loop over `bounds`, staging each iteration's operands
    /// from `src`.
    pub(in crate::codegen) fn gemm_loop(
        &mut self,
        block: &Block<'c>,
        acc: GemmAcc<'c>,
        bounds: (Value<'c, 'c>, Value<'c, 'c>, Value<'c, 'c>),
        src: &GemmSource<'_>,
    ) -> Result<GemmAcc<'c>> {
        match acc.plan.path {
            GemmPath::Reg { .. } => self.reg_loop(block, acc, bounds, src),
            GemmPath::Wmma { .. } | GemmPath::MmaSync { .. } => {
                self.tc_loop(block, acc, bounds, src)
            }
        }
    }

    /// Drains the finished accumulators to `view`, scaled.
    pub(in crate::codegen) fn gemm_store(
        &mut self,
        block: &Block<'c>,
        acc: GemmAcc<'c>,
        view: MemVal<'c>,
        alpha: Option<GemmScale<'c>>,
        beta: Option<GemmScale<'c>>,
    ) -> Result<()> {
        match acc.plan.path {
            GemmPath::Reg { .. } => self.reg_store(block, acc, view, alpha, beta),
            GemmPath::Wmma { .. } => self.wmma_store_acc(block, acc, view, alpha, beta),
            GemmPath::MmaSync { .. } => self.mma_sync_store(block, acc, view, alpha, beta),
        }
    }

    /// The origin the loop recorded. Fails if the loop has not run.
    pub(in crate::codegen) fn gemm_finals(acc: &GemmAcc<'c>) -> Result<GemmOrigin<'c>> {
        acc.origin
            .ok_or_else(|| anyhow!("a register matmul drained before its loop ran"))
    }

    /// The element types of the two operand slices, before any staging.
    pub(in crate::codegen) fn gemm_operand_elems(
        &self,
        src: &GemmSource<'_>,
    ) -> (Option<Type<'c>>, Option<Type<'c>>) {
        match src {
            GemmSource::Graph { ir, a, b, .. } => {
                let elem = |v: ValueId| ir.ty(v).elem().map(|e| self.ir_scalar_type(e));
                (elem(*a), elem(*b))
            }
        }
    }

    /// One operand slice of iteration `kt`, lowered into `block`. Every call
    /// lowers the slice afresh.
    pub(in crate::codegen) fn gemm_operand(
        &mut self,
        block: &Block<'c>,
        src: &GemmSource<'_>,
        which: GemmOperand,
        kt: Value<'c, 'c>,
    ) -> Result<MemVal<'c>> {
        match src {
            GemmSource::Graph {
                ir,
                iv,
                prefix,
                a,
                b,
            } => {
                // Lower the slice's ops in a new layer that binds the iv to
                // `kt`: up to a's definition for a, then on to b's for b.
                //
                // These ops get no barrier windows of their own. A masked
                // slice becomes a shared tile here, and its trailing barrier
                // orders the staging copy that follows. No graph op reads
                // that copy, so the membar pass must never elide the
                // barrier. The tile is scratch of the loop's window.
                let def = |v: ValueId| {
                    ir.def_op(v)
                        .and_then(|d| prefix.iter().position(|&o| o == d))
                        .ok_or_else(|| anyhow!("a gemm operand not defined in the loop body"))
                };
                let (range, v) = match which {
                    GemmOperand::A => (0..def(*a)? + 1, *a),
                    GemmOperand::B => (def(*a)? + 1..def(*b)? + 1, *b),
                };
                self.lowered.push(HashMap::new());
                self.set_ir(*iv, Lowered::Scalar(kt));
                let out = prefix[range]
                    .iter()
                    .try_for_each(|&o| self.emit_op_inner(block, ir, o))
                    .and_then(|()| self.tile_ir(ir, v));
                self.lowered.pop();
                out
            }
        }
    }

    /// Both operand slices of iteration `kt`, a then b.
    pub(in crate::codegen) fn gemm_operands(
        &mut self,
        block: &Block<'c>,
        src: &GemmSource<'_>,
        kt: Value<'c, 'c>,
    ) -> Result<(MemVal<'c>, MemVal<'c>)> {
        let a = self.gemm_operand(block, src, GemmOperand::A, kt)?;
        let b = self.gemm_operand(block, src, GemmOperand::B, kt)?;
        Ok((a, b))
    }

    /// Prepares the alpha and beta of a GEMM epilogue,
    /// alpha*acc [+ beta*prev], as f32. Broadcasts them to `vec_t` when the
    /// store is aligned.
    pub(in crate::codegen) fn epilogue_scaling(
        &mut self,
        block: &Block<'c>,
        alpha: Option<GemmScale<'c>>,
        beta: Option<GemmScale<'c>>,
        vec_t: Type<'c>,
        aligned: bool,
    ) -> Result<(Option<Value<'c, 'c>>, Option<Value<'c, 'c>>)> {
        let prep = |cg: &mut Self, s: GemmScale<'c>| -> Result<Value<'c, 'c>> {
            let v = match s {
                GemmScale::Value(v) => v,
                GemmScale::One => cg.const_f32(block, 1.0)?,
            };
            let v = cg.coerce(block, v, cg.f32_t)?; // TODO(joa): always f32 currently
            if aligned {
                cg.vec_broadcast(block, v, vec_t)
            } else {
                Ok(v)
            }
        };
        let alpha = match alpha {
            Some(a) => Some(prep(self, a)?),
            None => None,
        };
        let beta = match beta {
            Some(b) => Some(prep(self, b)?),
            None => None,
        };
        Ok((alpha, beta))
    }
}
