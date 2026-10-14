use std::fmt;

use anyhow::{Result, bail};

use super::{Binding, Build, Rv};
use crate::ast::{Expr, Stmt, Sub, Type as AstType};
use crate::ast::BinOp;
use crate::ir::{Bounds, ForInfo, IndexOp, OpKind, Pipeline, Scalar, Stage, Type, ValueId};

/// What a block may hold in static shared memory, on every chip.
const STATIC_SHARED_LIMIT_BYTES: i64 = 48 * 1024;

/// Why a loop body was not turned into a pipelined loop.
pub(crate) enum PipelineDecline {
    NoStagedPrefix,
    PartialSlice,
    StagedNameWritten,
    UnknownElementWidth,
    SharedMemoryBudget { needed_bytes: i64, limit_bytes: i64 },
}

impl fmt::Display for PipelineDecline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoStagedPrefix => {
                f.write_str("no leading run of staged tensor-slice `var` statements")
            }
            Self::PartialSlice => f.write_str(
                "a staged slice's shape cannot be proven to tile evenly (partial slice)",
            ),
            Self::StagedNameWritten => {
                f.write_str("a staged name is written again later in the loop body")
            }
            Self::UnknownElementWidth => {
                f.write_str("a staged slice's element type has no known byte width")
            }
            Self::SharedMemoryBudget {
                needed_bytes,
                limit_bytes,
            } => write!(
                f,
                "doubling the staged buffers would need {needed_bytes} bytes of shared \
                 memory, over the {limit_bytes}-byte ceiling"
            ),
        }
    }
}

/// A matched pipeline candidate: the staged `(name, slice)` prefix, the
/// compute statements after it, and what the `For` records.
pub(crate) type Candidate<'a> = (Vec<(&'a str, &'a Expr)>, &'a [Stmt], Pipeline);

impl Build {
    /// Matches a loop body shaped "var t = <static tensor slice>; ...". The
    /// budget is checked against an empty pool, since only a `@dynshared`
    /// kernel has dynamic bytes handed out by this point.
    pub(crate) fn pipeline_candidate<'a>(
        &self,
        body: &'a [Stmt],
    ) -> Result<Candidate<'a>, PipelineDecline> {
        let mut staged = Vec::new();
        let mut rest = body;
        while let [
            Stmt::Var {
                name,
                ty: None,
                value: Some(value),
                ..
            },
            tail @ ..,
        ] = rest
        {
            if self.slice_static_shape(value).is_none() {
                break;
            }
            staged.push((name.as_str(), value));
            rest = tail;
        }
        if staged.is_empty() {
            return Err(PipelineDecline::NoStagedPrefix);
        }
        if staged.iter().any(|(_, v)| self.slice_is_partial(v)) {
            return Err(PipelineDecline::PartialSlice);
        }
        let names: Vec<&str> = staged.iter().map(|(n, _)| *n).collect();
        if rest.iter().any(|s| s.writes_any(&names)) {
            return Err(PipelineDecline::StagedNameWritten);
        }
        let mut needed_bytes = 0i64;
        for (_, expr) in &staged {
            let shape = self
                .slice_static_shape(expr)
                .expect("checked above: staged slices have a static shape");
            let elem = self
                .slice_tensor_elem(expr)
                .expect("checked above: staged slices index a named tensor");
            let Some(width) = elem.bytes() else {
                return Err(PipelineDecline::UnknownElementWidth);
            };
            let bytes = width * shape.iter().product::<i64>();
            needed_bytes += (bytes + 15) & !15;
        }
        let doubled = needed_bytes * 2;
        // A `@dynshared` launch opts into the chip's whole per-block budget.
        let limit_bytes = if self.ir.kernel.dynamic_shared {
            self.target.smem_per_block
        } else {
            STATIC_SHARED_LIMIT_BYTES
        };
        if doubled > limit_bytes {
            return Err(PipelineDecline::SharedMemoryBudget {
                needed_bytes: doubled,
                limit_bytes,
            });
        }
        let info = Pipeline {
            staged: staged.len(),
            ends_with_tile_op: self.ends_with_tile_op(rest),
            doubled_bytes: doubled,
        };
        Ok((staged, rest, info))
    }

    /// A pipelined loop: the staged prefix as stage ops the emitter turns
    /// into its buffer pairs, and the compute built with the staged names
    /// bound as views.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_pipelined_for(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        staged: &[(&str, &Expr)],
        rest: &[Stmt],
        info: Pipeline,
        hoisted: &[ValueId],
    ) -> Result<()> {
        let (lo, hi, st, iv_div) = self.loop_bounds(start, end, step)?;
        self.emit_pipelined_loop(var, [lo, hi, st], iv_div, staged, rest, info, hoisted)
    }

    /// A loop over a dynamic extent whose staged slices tile evenly only
    /// once it is trimmed: the whole tiles pipelined, then the ragged rest
    /// as one plain, masked iteration.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_pipelined_split_for(
        &mut self,
        var: &str,
        start: &Expr,
        end: &Expr,
        step: Option<&Expr>,
        body: &[Stmt],
        staged: &[(&str, &Expr)],
        rest: &[Stmt],
        info: Pipeline,
        hoisted: &[ValueId],
    ) -> Result<()> {
        let (lo, hi, st, iv_div) = self.loop_bounds(start, end, step)?;
        let span = self.value(OpKind::Binary(BinOp::Sub), &[hi, lo], Type::INDEX);
        let chunks = self.value(OpKind::IndexOp(IndexOp::DivU), &[span, st], Type::INDEX);
        let covered = self.value(OpKind::Binary(BinOp::Mul), &[chunks, st], Type::INDEX);
        let full = self.value(OpKind::Binary(BinOp::Add), &[lo, covered], Type::INDEX);

        self.trimmed_ivs.push(var.to_string());
        let main = self.emit_pipelined_loop(var, [lo, full, st], iv_div, staged, rest, info, hoisted);
        self.trimmed_ivs.pop();
        main?;

        let ragged = self.value(OpKind::Binary(BinOp::Lt), &[full, hi], Type::Scalar(Scalar::Bool));
        let tail = self.ir.new_block(&[]);
        let outer = self.ragged_iv.replace(var.to_string());
        let built = self.in_block(tail, |b| {
            b.emit_scope(&[(var, Binding::Let { value: full, div: iv_div })], body)?;
            b.stmt(OpKind::Yield, &[]);
            Ok::<(), anyhow::Error>(())
        });
        self.ragged_iv = outer;
        built?;
        self.op(OpKind::If, &[ragged], Vec::new(), vec![tail]);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_pipelined_loop(
        &mut self,
        var: &str,
        [lo, hi, st]: [ValueId; 3],
        iv_div: i64,
        staged: &[(&str, &Expr)],
        rest: &[Stmt],
        info: Pipeline,
        hoisted: &[ValueId],
    ) -> Result<()> {
        let block = self.ir.new_block(&[crate::ir::Type::INDEX]);
        let iv = self.ir.args(block)[0];
        self.ir.set_name(iv, var);
        self.in_block(block, |b| {
            b.push_scope();
            b.bind(var, Binding::Let { value: iv, div: iv_div });
            let mut bindings: Vec<(&str, Binding)> = Vec::new();
            for (name, expr) in staged {
                let Rv::Tile(src) = b.emit_expr(expr)? else {
                    bail!("staged value must be a tensor slice");
                };
                let (elem, shape) = (b.elem(src), b.shape(src));
                let pad = b.should_pad_stage(elem, &shape);
                let ty = if pad { b.padded_ty(elem, &shape) } else { b.like(src) };
                let buf = b.value(OpKind::Stage(Stage { pad, sync: true }), &[src], ty);
                bindings.push((name, Binding::View(buf)));
            }
            b.emit_scope(&bindings, rest)?;
            b.pop_scope();
            b.stmt(OpKind::Yield, &[]);
            Ok::<(), anyhow::Error>(())
        })?;
        let mut operands = vec![lo, hi, st];
        operands.extend_from_slice(hoisted);
        self.op(
            OpKind::For(ForInfo {
                bounds: Bounds::Dynamic,
                ragged: false,
                carried: 0,
                hoisted: hoisted.len(),
                pipeline: Some(info),
            }),
            &operands,
            Vec::new(),
            vec![block],
        );
        Ok(())
    }

    /// Whether the last statement lowers to a tile op, which always ends
    /// with its own barrier.
    fn ends_with_tile_op(&self, stmts: &[Stmt]) -> bool {
        match stmts.last() {
            Some(Stmt::Assign { target, .. }) => match target {
                Expr::Var(n) => matches!(self.lookup(n), Some(Binding::Tile(_))),
                Expr::Index { subs, .. } => subs.iter().any(|s| !matches!(s, Sub::Point(_))),
                _ => false,
            },
            Some(Stmt::Var {
                ty: Some(AstType::Tile(..)),
                ..
            }) => true,
            _ => false,
        }
    }
}
