// The register matmul in thirds: the accumulators are seeded before the
// k-loop, carried through it, and drained after it. Keeping the thirds
// apart lets whoever drives them emit the loop bounds and the epilogue's
// operands in between.

use super::*;

use crate::codegen::lower::Lowered;
use crate::ir::{Ir, OpId, ValueId};

/// Which of the three paths runs a register matmul, with the tiling that
/// path decided from the shapes alone.
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

/// The warp's origin values the k-loop computes and the epilogue drains
/// from: (tid, w, wt, m0, n0), as [`Codegen::warp_block_origin`] returns them.
pub(in crate::codegen) type GemmOrigin<'c> = (
    Value<'c, 'c>,
    Value<'c, 'c>,
    Value<'c, 'c>,
    Value<'c, 'c>,
    Value<'c, 'c>,
);

/// The accumulators between the thirds.
#[derive(Clone)]
pub(in crate::codegen) struct GemmAcc<'c> {
    pub(in crate::codegen) plan: GemmPlan<'c>,
    pub(in crate::codegen) regs: Vec<Value<'c, 'c>>,
    /// Set by the loop third.
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
    /// The body of the graph's loop: its iv argument, the ops before the
    /// `gemm_dot`, and the dot's two slice operands, each defined by one of
    /// those ops. The a slice's ops come first, then the b slice's, and
    /// neither slice's ops use the other's values, so each can be lowered
    /// on its own.
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
    /// The identity, an f32 one.
    One,
}

impl<'c> Codegen<'c> {
    /// Decides the path and its tiling for an [m, n] accumulator over a
    /// k extent of kk.
    pub(in crate::codegen) fn gemm_plan(
        &self,
        m: i64,
        n: i64,
        kk: i64,
        acc_elem: Type<'c>,
    ) -> Result<GemmPlan<'c>> {
        let path = if let Some((wm, wn)) = self.has_wmma().then(|| self.wmma_plan(m, n, kk)).flatten() {
            if self.has_mma_sync() {
                GemmPath::MmaSync { wm, wn }
            } else {
                GemmPath::Wmma { wm, wn }
            }
        } else {
            let (tm, tn) = self.sub_tile(m, n);
            let (tiles_m, tiles_n) = (m / tm, n / tn);
            let (lm, ln) = Self::lane_grid(tiles_m, tiles_n, tm, tn)
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

    /// The finished accumulators and the origin the loop third recorded.
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

    /// One operand slice of iteration `kt`, lowered into `block`. The
    /// slices are lowered in the order the caller asks for them, and every
    /// call lowers its slice afresh.
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
                // The slice's ops, lowered afresh under a layer binding the
                // iv to this call's kt: up to a's definition for a, from
                // there to b's for b. Lowered without windows of their own:
                // a masked slice materializes into a shared tile here, and
                // its trailing barrier orders the staging copy this loop
                // makes next, which no op of the graph reads, so the barrier
                // pass must never see it as a candidate. The tile is scratch
                // of the loop's window.
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

    /// Precomputes the alpha and beta operands of a GEMM epilogue
    /// (alpha*acc [+ beta*prev_load]), broadcast to vec_t when the store is
    /// aligned. Each coefficient is prepared whole before the next.
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
