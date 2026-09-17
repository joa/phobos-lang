use anyhow::{Result, anyhow, bail};
use melior::{
    dialect::arith,
    ir::{Block, Type, attribute::FloatAttribute},
};

use super::{Lowered, dims};
use crate::codegen::tile::{QFormat, QgFormat};
use crate::codegen::matmul::GemmScale;
use crate::codegen::{Codegen, FragAcc, MemVal};
use crate::ir::{self, Coeff, Intrinsic, Ir, OpId, OpKind, RawFmt};

impl<'c> Codegen<'c> {
    pub(super) fn emit_intrinsic_op(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        match ir.kind(op).clone() {
            OpKind::Intrinsic(i) => {
                let n = ir.operands(op).len();
                match i {
                    Intrinsic::RmsNormQ => {
                        let x = self.tile_ir(ir, ir.operand(op, 0))?;
                        let g = self.tile_ir(ir, ir.operand(op, 1))?;
                        let eps = self.scalar_ir(ir.operand(op, 2))?;
                        let (o, q, s) = if n == 6 {
                            (
                                Some(self.tile_ir(ir, ir.operand(op, 3))?),
                                self.tile_ir(ir, ir.operand(op, 4))?,
                                self.tile_ir(ir, ir.operand(op, 5))?,
                            )
                        } else {
                            (
                                None,
                                self.tile_ir(ir, ir.operand(op, 3))?,
                                self.tile_ir(ir, ir.operand(op, 4))?,
                            )
                        };
                        let inv = self.rms_norm_q_raw(block, x, g, eps, o, q, s)?;
                        self.set_ir(ir.result(op), Lowered::Scalar(inv));
                    }
                    Intrinsic::WarpPartial => {
                        let tiles = self.tiles_ir(ir, op, 0..6)?;
                        let scalars = [6, 7, 8, 9]
                            .map(|k| self.scalar_ir(ir.operand(op, k)))
                            .into_iter()
                            .collect::<Result<Vec<_>>>()?;
                        let refs: [&MemVal<'c>; 6] = std::array::from_fn(|k| &tiles[k]);
                        let scalars: [_; 4] = std::array::from_fn(|k| scalars[k]);
                        self.warp_partial_raw(block, refs, scalars)?;
                        let zero = self.const_index(block, 0)?;
                        self.set_ir(ir.result(op), Lowered::Scalar(zero));
                    }
                    other => {
                        let tiles = self.tiles_ir(ir, op, 0..n)?;
                        let out = self.intrinsic_fresh(block, ir, op, other, &tiles)?;
                        for t in &tiles {
                            self.release(t);
                        }
                        self.set_ir(ir.result(op), Lowered::Tile(out));
                    }
                }
            }
            OpKind::IntrinsicInto(i) => {
                let n = ir.operands(op).len();
                let tiles = self.tiles_ir(ir, op, 0..n - 1)?;
                let dst = self.tile_ir(ir, ir.operand(op, n - 1))?;
                self.intrinsic_into(block, i, &tiles, &dst)?;
                for t in &tiles {
                    self.release(t);
                }
            }
            OpKind::FragInit(init) => {
                let ir::Type::Frags(f) = ir.ty(ir.result(op)) else {
                    bail!("frag_init of a non-fragment");
                };
                let acc_t = Type::vector(&[2, 2], self.f32_t);
                let init = self.push(
                    block,
                    arith::constant(
                        self.ctx,
                        FloatAttribute::new(self.ctx, self.f32_t, init).into(),
                        self.loc,
                    ),
                )?;
                let seed = self.vec_broadcast(block, init, acc_t)?;
                let fa = FragAcc {
                    frags: Vec::new(),
                    m: f.m,
                    n: f.n,
                    wm: f.wm,
                    wn: f.wn,
                };
                let (fm, fnn) = fa.warp_frags();
                let frags = vec![seed; (fm * fnn * 2) as usize];
                self.set_ir(ir.result(op), Lowered::Frags(FragAcc { frags, ..fa }));
            }
            OpKind::FragScale(bop) => {
                let fa = self.frags_ir(ir.operand(op, 0))?;
                let col = self.tile_ir(ir, ir.operand(op, 1))?;
                let frags = self.frag_scale_raw(block, &fa, bop, &col)?;
                self.set_ir(ir.result(op), Lowered::Frags(FragAcc { frags, ..fa }));
            }
            OpKind::FragDot => {
                let fa = self.frags_ir(ir.operand(op, 0))?;
                let a = self.tile_ir(ir, ir.operand(op, 1))?;
                let b = self.tile_ir(ir, ir.operand(op, 2))?;
                let frags = self.frag_dot_raw(block, &fa, &a, &b)?;
                self.release(&a);
                self.release(&b);
                self.set_ir(ir.result(op), Lowered::Frags(FragAcc { frags, ..fa }));
            }
            OpKind::FragStore => {
                let fa = self.frags_ir(ir.operand(op, 0))?;
                let dst = self.tile_ir(ir, ir.operand(op, 1))?;
                self.frag_store(block, &dst, &fa)?;
            }
            OpKind::GemmInit => {
                let ir::Type::Gemm(g) = ir.ty(ir.result(op)) else {
                    bail!("gemm_init of a non-accumulator");
                };
                let plan = self.gemm_plan(g.m, g.n, g.k, self.ir_scalar_type(g.acc))?;
                let init = self.scalar_ir(ir.operand(op, 0))?;
                let acc = self.gemm_seed(block, plan, init)?;
                self.set_ir(ir.result(op), Lowered::Gemm(acc));
            }
            OpKind::GemmStore { alpha, beta } => {
                let acc = self.gemm_ir(ir.operand(op, 0))?;
                let view = self.tile_ir(ir, ir.operand(op, 1))?;
                self.check_shapes(&view.shape, &[acc.plan.m, acc.plan.n], "matmul epilogue store")?;
                // The match declines partial output tiles, so the drains, which
                // have no per-element bounds guard, never see one.
                if view.is_masked() {
                    bail!("internal: register matmul epilogue reached a partial output tile");
                }
                let mut next = 2;
                let mut scale = |c: Coeff| -> Result<Option<GemmScale<'c>>> {
                    Ok(match c {
                        Coeff::Absent => None,
                        Coeff::One => Some(GemmScale::One),
                        Coeff::Given => {
                            next += 1;
                            Some(GemmScale::Value(self.scalar_ir(ir.operand(op, next - 1))?))
                        }
                    })
                };
                let alpha = scale(alpha)?;
                let beta = scale(beta)?;
                self.gemm_store(block, acc, view, alpha, beta)?;
            }
            other => bail!("{} is not an intrinsic", other.name()),
        }
        Ok(())
    }

    /// An intrinsic that yields a fresh buffer.
    fn intrinsic_fresh(
        &mut self,
        block: &Block<'c>,
        ir: &Ir,
        op: OpId,
        i: Intrinsic,
        t: &[MemVal<'c>],
    ) -> Result<MemVal<'c>> {
        let fresh_out = |cg: &mut Self| -> Result<MemVal<'c>> {
            let ir::Type::Tile(ty) = ir.ty(ir.result(op)) else {
                bail!("{} to a non-tile", i.name());
            };
            let elem = cg.ir_scalar_type(ty.elem);
            cg.alloc_tile_shaped(block, elem, &dims(&ty.shape))
        };
        Ok(match i {
            Intrinsic::QdotT => self.tile_qdot_t(block, &t[0], &t[1], &t[2], &t[3])?,
            Intrinsic::QmmaT => self.tile_qmma_t(block, &t[0], &t[1], &t[2], &t[3])?,
            Intrinsic::RawQdotI8(fmt) => {
                let qg = qg_format(fmt, &i)?;
                self.tile_qdot_i8_reg_t(block, qg, &t[0], &t[1], &t[2], &t[3], &t[4..])?
            }
            Intrinsic::RawQdot(fmt) => match fmt {
                RawFmt::Iq1s => self.tile_iq1s_qdot_t(block, &t[0], &t[1], &t[2], &t[3])?,
                RawFmt::Iq1m => self.tile_iq1m_qdot_t(block, &t[0], &t[1], &t[2], &t[3])?,
                RawFmt::Iq2xxs => {
                    self.tile_iq2xxs_qdot_t(block, &t[0], &t[1], &t[2], &t[3], &t[4])?
                }
                RawFmt::Iq2s => self.tile_iq2s_qdot_t(block, &t[0], &t[1], &t[2], &t[3], &t[4])?,
                RawFmt::Iq2xs => {
                    self.tile_iq2xs_qdot_t(block, &t[0], &t[1], &t[2], &t[3], &t[4])?
                }
                RawFmt::Iq3xxs => {
                    self.tile_iq3xxs_qdot_t(block, &t[0], &t[1], &t[2], &t[3], &t[4])?
                }
                RawFmt::Iq3s => self.tile_iq3s_qdot_t(block, &t[0], &t[1], &t[2], &t[3], &t[4])?,
                RawFmt::Iq4xs => self.tile_iq4xs_qdot_t(block, &t[0], &t[1], &t[2], &t[3])?,
                RawFmt::Q2k => self.tile_q2k_qdot_t(block, &t[0], &t[1], &t[2], &t[3])?,
                RawFmt::Q3k => self.tile_q3k_qdot_t(block, &t[0], &t[1], &t[2])?,
                other => bail!("no {}_qdot_t", other.name()),
            },
            Intrinsic::RawQmma(RawFmt::Iq1s) => {
                self.tile_iq1s_qmma_t(block, &t[0], &t[1], &t[2], &t[3], &t[4])?
            }
            Intrinsic::RawQmma(other) => bail!("no {}_qmma_t", other.name()),
            Intrinsic::RawQmmaStaged(fmt) => {
                let out = fresh_out(self)?;
                self.qmma_staged_into(block, fmt, t, &out)?;
                out
            }
            Intrinsic::RawQgemm(fmt) => {
                let out = fresh_out(self)?;
                let qg = qg_format(fmt, &i)?;
                self.qgemm_into(block, qg, &t[0], &t[1], &t[2], &t[3], &t[4..], &out)?;
                out
            }
            Intrinsic::Gather => self.tile_gather(block, &t[0], &t[1])?,
            Intrinsic::ArgSel => {
                let out = fresh_out(self)?;
                self.tile_argsel_bc(block, &t[0], &t[1], &t[2], &t[3], &out)?;
                out
            }
            Intrinsic::RawQdecode(_) | Intrinsic::RmsNormQ | Intrinsic::WarpPartial => {
                bail!("{} yields no fresh buffer", i.name())
            }
        })
    }

    /// An intrinsic writing its destination, the `*_into` forms `store_tile`
    /// takes.
    fn intrinsic_into(
        &mut self,
        block: &Block<'c>,
        i: Intrinsic,
        t: &[MemVal<'c>],
        dst: &MemVal<'c>,
    ) -> Result<()> {
        match i {
            Intrinsic::QmmaT => self.qmma_t_into(block, &t[0], &t[1], &t[2], &t[3], dst),
            Intrinsic::RawQmma(RawFmt::Iq1s) => {
                self.iq1s_qmma_t_into(block, &t[0], &t[1], &t[2], &t[3], &t[4], dst)
            }
            Intrinsic::RawQgemm(fmt) => {
                let qg = qg_format(fmt, &i)?;
                self.qgemm_into(block, qg, &t[0], &t[1], &t[2], &t[3], &t[4..], dst)
            }
            Intrinsic::RawQmmaStaged(fmt) => self.qmma_staged_into(block, fmt, t, dst),
            Intrinsic::RawQdecode(fmt) => {
                let qf = QFormat::from_intrinsic(&i.name())
                    .ok_or_else(|| anyhow!("no decode for {}", fmt.name()))?;
                self.qdecode_t_into(block, qf, &t[0], &t[1], &t[2..], dst)
            }
            other => bail!("{} has no into form", other.name()),
        }
    }

    fn qmma_staged_into(
        &mut self,
        block: &Block<'c>,
        fmt: RawFmt,
        t: &[MemVal<'c>],
        out: &MemVal<'c>,
    ) -> Result<()> {
        match fmt {
            RawFmt::Iq1s => self.iq1s_qmma_staged_into(block, &t[0], &t[1], &t[2], &t[3], &t[4], out),
            RawFmt::Iq2s => {
                self.iq2s_qmma_staged_into(block, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], out)
            }
            RawFmt::Iq2xs => {
                self.iq2xs_qmma_staged_into(block, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], out)
            }
            RawFmt::Iq3xxs => {
                self.iq3xxs_qmma_staged_into(block, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], out)
            }
            RawFmt::Iq3s => {
                self.iq3s_qmma_staged_into(block, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], out)
            }
            RawFmt::Iq2xxs => {
                self.iq2xxs_qmma_staged_into(block, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], out)
            }
            other => bail!("no {}_qmma_staged_t", other.name()),
        }
    }
}

/// The emitter's grouped-format enum for a raw format, by the intrinsic's
/// own name so the two tables cannot drift.
fn qg_format(fmt: RawFmt, i: &Intrinsic) -> Result<QgFormat> {
    let name = i.name();
    QgFormat::from_intrinsic(&name)
        .or_else(|| QgFormat::from_qdot_i8(&name))
        .ok_or_else(|| anyhow!("{} is not a grouped raw format", fmt.name()))
}
