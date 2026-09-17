mod control;
mod intrinsics;
mod pipeline;
pub(super) mod membar;
pub(super) mod plan;
mod scalar;
mod tiles;

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use melior::ir::{
    Attribute, Block, BlockLike, Operation, Region, RegionLike, Type,
    attribute::{StringAttribute, TypeAttribute},
    operation::OperationBuilder,
    r#type::{FunctionType, MemRefType},
};

use super::matmul::GemmAcc;
use super::{Codegen, DYN, FragAcc, MemVal, detach};
use crate::ir::{self, Ir, OpId, OpKind, ValueId};

/// What an IR value became in MLIR.
#[derive(Clone)]
pub(super) enum Lowered<'c> {
    Scalar(melior::ir::Value<'c, 'c>),
    /// A `var` scalar's private slot.
    Slot(melior::ir::Value<'c, 'c>),
    Tile(MemVal<'c>),
    Frags(FragAcc<'c>),
    Gemm(GemmAcc<'c>),
}

impl<'c> Codegen<'c> {
    /// The kernel as a `gpu.func`, from its graph: a walk over the ops
    /// that calls the tile emitters in turn. Which operands each arm
    /// releases matters, since the pool's reuse order is part of the
    /// emitted module.
    ///
    /// Values are looked up through a stack of layers. The plain walk uses
    /// one; a body instantiated more than once, a pipelined loop's halves
    /// and prefetches, pushes a layer with the substitutions that instance
    /// needs.
    pub(super) fn emit_kernel_ir(&mut self, ir: &Ir) -> Result<Operation<'c>> {
        let entry = ir.entry();
        let param_types = ir
            .args(entry)
            .iter()
            .map(|&a| self.ir_type(ir.ty(a)))
            .collect::<Result<Vec<_>>>()?;
        let block_args: Vec<_> = param_types.iter().map(|&t| (t, self.loc)).collect();
        let mlir_entry = Block::new(&block_args);

        self.lowered.push(HashMap::new());
        for (i, &a) in ir.args(entry).iter().enumerate() {
            let arg = detach(mlir_entry.argument(i)?.into());
            let lowered = match ir.ty(a) {
                ir::Type::Tensor(t) => {
                    let elem = self.ir_scalar_type(t.elem);
                    Lowered::Tile(MemVal {
                        mem: arg,
                        elem,
                        shape: dims(&t.shape),
                        row_stride: None,
                        align_div: 1,
                        swizzle: None,
                        global: None,
                        shared: false,
                        owned: false,
                        mask: Vec::new(),
                    })
                }
                ir::Type::Scalar(_) => Lowered::Scalar(arg),
                other => bail!("kernel '{}': a {other} parameter", ir.kernel.name),
            };
            self.set_ir(a, lowered);
        }

        let body = self.emit_ops(&mlir_entry, ir, ir.ops(entry));
        self.lowered.pop();
        body?;

        self.kernel_return(&mlir_entry)?;

        let fn_region = Region::new();
        fn_region.append_block(mlir_entry);

        let fn_type: Type = FunctionType::new(self.ctx, &param_types, &[]).into();
        let mut attrs = vec![
            (
                self.id("sym_name"),
                StringAttribute::new(self.ctx, &ir.kernel.name).into(),
            ),
            (self.id("function_type"), TypeAttribute::new(fn_type).into()),
            (self.id("gpu.kernel"), Attribute::unit(self.ctx)),
        ];
        if let Some(launch) = self.launch {
            attrs.extend(self.launch_attrs(launch));
        }
        Ok(OperationBuilder::new("gpu.func", self.loc)
            .add_attributes(&attrs)
            .add_regions([fn_region])
            .build()?)
    }

    // Types.

    pub(super) fn ir_scalar_type(&self, s: ir::Scalar) -> Type<'c> {
        match s {
            ir::Scalar::F16 => self.f16_t,
            ir::Scalar::BF16 => self.bf16_t,
            ir::Scalar::F32 => self.f32_t,
            ir::Scalar::F64 => self.f64_t,
            ir::Scalar::I8 => self.i8_t,
            ir::Scalar::I32 => self.i32_t,
            ir::Scalar::I64 => self.i64_t,
            ir::Scalar::Bool => self.bool_t,
            ir::Scalar::Index => self.index_t,
        }
    }

    fn ir_type(&self, ty: &ir::Type) -> Result<Type<'c>> {
        Ok(match ty {
            ir::Type::Scalar(s) => self.ir_scalar_type(*s),
            ir::Type::Tensor(t) => MemRefType::new(
                self.ir_scalar_type(t.elem),
                &dims(&t.shape),
                None,
                Some(self.global_space()),
            )
            .into(),
            other => bail!("no MLIR type for {other}"),
        })
    }

    // The value stack.

    pub(super) fn set_ir(&mut self, v: ValueId, lowered: Lowered<'c>) {
        self.lowered
            .last_mut()
            .expect("a value layer is open while lowering")
            .insert(v, lowered);
    }

    fn get_ir(&self, v: ValueId) -> Result<&Lowered<'c>> {
        self.lowered
            .iter()
            .rev()
            .find_map(|layer| layer.get(&v))
            .ok_or_else(|| anyhow!("{v} was used before it was lowered"))
    }

    pub(super) fn scalar_ir(&self, v: ValueId) -> Result<melior::ir::Value<'c, 'c>> {
        match self.get_ir(v)? {
            Lowered::Scalar(x) => Ok(*x),
            _ => bail!("{v} is not a scalar"),
        }
    }

    pub(super) fn slot_ir(&self, v: ValueId) -> Result<melior::ir::Value<'c, 'c>> {
        match self.get_ir(v)? {
            Lowered::Slot(x) => Ok(*x),
            _ => bail!("{v} is not a scalar slot"),
        }
    }

    /// A tile operand as the emitters want it. `owned` is what the old
    /// binding rules said: a fresh buffer nothing named, and so one the
    /// consuming op may return to the pool.
    pub(super) fn tile_ir(&self, ir: &Ir, v: ValueId) -> Result<MemVal<'c>> {
        match self.get_ir(v)? {
            Lowered::Tile(mv) => {
                let mut mv = mv.clone();
                mv.owned = ir.name(v).is_none()
                    && ir.def_op(v).is_some_and(|op| ir.kind(op).makes_buffer());
                Ok(mv)
            }
            _ => bail!("{v} is not a tile"),
        }
    }

    pub(super) fn frags_ir(&self, v: ValueId) -> Result<FragAcc<'c>> {
        match self.get_ir(v)? {
            Lowered::Frags(fa) => Ok(fa.clone()),
            _ => bail!("{v} is not a fragment accumulator"),
        }
    }

    pub(super) fn gemm_ir(&self, v: ValueId) -> Result<GemmAcc<'c>> {
        match self.get_ir(v)? {
            Lowered::Gemm(acc) => Ok(acc.clone()),
            _ => bail!("{v} is not a gemm accumulator"),
        }
    }

    /// The tile operands `range` of `op`.
    pub(super) fn tiles_ir(
        &self,
        ir: &Ir,
        op: OpId,
        range: std::ops::Range<usize>,
    ) -> Result<Vec<MemVal<'c>>> {
        range.map(|i| self.tile_ir(ir, ir.operand(op, i))).collect()
    }

    /// Every op of a block but its terminator, which the op owning the
    /// block appends itself.
    pub(super) fn emit_ops(&mut self, block: &Block<'c>, ir: &Ir, ops: &[OpId]) -> Result<()> {
        for &op in ops {
            if ir.kind(op).is_terminator() {
                continue;
            }
            self.emit_op(block, ir, op)?;
        }
        Ok(())
    }

    /// One op, bracketed in the trace when recording, with each buffer it
    /// yields tied to the allocation the pool gave it.
    pub(super) fn emit_op(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        let recording = matches!(self.policy, super::SharedPolicy::Record);
        if recording {
            self.trace.events.push(plan::Event::OpBegin(op));
        }
        let skip = self.elided.get(&op).copied();
        if skip.is_some() {
            self.skip_barrier = skip;
            self.barrier_calls = 0;
        }
        let emitted = self.emit_op_inner(block, ir, op);
        if skip.is_some() {
            self.skip_barrier = None;
        }
        emitted?;
        if recording {
            self.trace.events.push(plan::Event::OpEnd(op));
            if ir.kind(op).makes_buffer() {
                for &r in ir.results(op) {
                    let name = match self.get_ir(r) {
                        Ok(Lowered::Tile(mv)) => mv.global.clone(),
                        _ => None,
                    };
                    if let Some(name) = name {
                        self.trace.assign(r, &name);
                    }
                }
            }
        }
        Ok(())
    }

    /// One op with no window in the trace: what a loop lowers on its own
    /// behalf.
    pub(in crate::codegen) fn emit_op_inner(&mut self, block: &Block<'c>, ir: &Ir, op: OpId) -> Result<()> {
        match ir.kind(op) {
            OpKind::Const(_)
            | OpKind::Unary(_)
            | OpKind::Binary(_)
            | OpKind::IndexOp(_)
            | OpKind::Cast(_)
            | OpKind::IndexCast(_)
            | OpKind::ProgramId(_)
            | OpKind::Dim(_)
            | OpKind::Load
            | OpKind::Store
            | OpKind::AtomicAdd
            | OpKind::Barrier
            | OpKind::GridBarrier => self.emit_scalar_op(block, ir, op),
            OpKind::For(_) | OpKind::While | OpKind::If => self.emit_control_op(block, ir, op),
            OpKind::Yield | OpKind::Condition => {
                bail!("a terminator reached the op walk")
            }
            // Lowered by the loop carrying its accumulator.
            OpKind::GemmDot => bail!("a gemm_dot reached the op walk"),
            OpKind::Intrinsic(_)
            | OpKind::IntrinsicInto(_)
            | OpKind::FragInit(_)
            | OpKind::FragScale(_)
            | OpKind::FragDot
            | OpKind::FragStore
            | OpKind::GemmInit
            | OpKind::GemmStore { .. } => self.emit_intrinsic_op(block, ir, op),
            _ => self.emit_tile_op(block, ir, op),
        }
    }

    /// The terminator of `block` as the values it yields.
    pub(super) fn yielded(&self, ir: &Ir, block: ir::BlockId) -> Result<Vec<ValueId>> {
        let last = ir
            .ops(block)
            .last()
            .ok_or_else(|| anyhow!("{block} has no terminator"))?;
        Ok(ir.operands(*last).to_vec())
    }
}

/// An IR shape as the emitters spell it, with [`DYN`] for a dynamic extent.
pub(super) fn dims(shape: &[ir::Extent]) -> Vec<i64> {
    shape.iter().map(|e| e.fixed().unwrap_or(DYN)).collect()
}
