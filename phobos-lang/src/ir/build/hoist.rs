use std::collections::{HashMap, HashSet};

use anyhow::Result;

use super::{Binding, Build, DYN};
use crate::ast::{Expr, Scalar as AstScalar, Stmt, Type as AstType};
use crate::ir::{Layout, OpKind, Scalar, Space, Swizzle, TileType, Type, ValueId};
use crate::shape;

/// Names and shapes the body scan has resolved so far.
#[derive(Default)]
struct HoistScan {
    declared: HashSet<String>,
    decls: HashMap<String, (bool, Vec<i64>)>,
}

impl Build {
    /// Stages the body's hoistable dot operands into the current block, the
    /// loop's preheader, and returns the `(source, buffer)` frame.
    /// Stages a loop's invariant dot operands in its preheader: one
    /// `HoistStage` op per candidate, and a barrier when there were any.
    pub(crate) fn hoist_dot_staging(&mut self, body: &[Stmt]) -> Result<Vec<(ValueId, ValueId)>> {
        let mut frame = Vec::new();
        if !self.has_wmma() {
            return Ok(frame);
        }
        let Some(cands) = self.hoist_candidates(body) else {
            return Ok(frame);
        };
        for src in cands {
            if self.hoisted_stage(src).is_some() {
                continue;
            }
            let shape = self.shape(src);
            let ty = if self.has_mma_sync() {
                swizzled_ty(&shape)
            } else {
                self.shared_ty(Scalar::F16, &shape)
            };
            let buf = self.value(OpKind::HoistStage, &[src], ty);
            frame.push((src, buf));
        }
        if !frame.is_empty() {
            self.stmt(OpKind::Barrier, &[]);
        }
        Ok(frame)
    }

    /// The preheader-staged buffer for a dot operand, when a surrounding
    /// loop hoisted it.
    pub(crate) fn hoisted_stage(&self, src: ValueId) -> Option<ValueId> {
        self.hoisted
            .iter()
            .rev()
            .flatten()
            .find(|(v, _)| *v == src)
            .map(|(_, buf)| *buf)
    }

    fn hoist_candidates(&self, body: &[Stmt]) -> Option<Vec<ValueId>> {
        let mut scan = HoistScan::default();
        let mut cands = Vec::new();
        self.hoist_scan(body, &mut scan, &mut cands).then_some(cands)
    }

    fn hoist_scan(&self, stmts: &[Stmt], scan: &mut HoistScan, cands: &mut Vec<ValueId>) -> bool {
        for stmt in stmts {
            match stmt {
                Stmt::Let { .. } | Stmt::Var { .. } => {
                    let Some((name, ty, Some(value))) = stmt.as_decl() else {
                        continue;
                    };
                    if let Expr::Call { callee, args } = value
                        && (callee == "dot" || callee == "dot_t")
                    {
                        let out_f32 = match ty {
                            Some(AstType::Tile(sc, _)) => *sc == AstScalar::F32,
                            Some(_) => false,
                            None => args
                                .first()
                                .and_then(|a| self.hoist_operand(scan, a))
                                .is_some_and(|(is_f32, _)| is_f32),
                        };
                        self.hoist_consider(scan, cands, callee == "dot_t", args, out_f32);
                    }
                    self.hoist_record_decl(scan, name, ty, value);
                }
                Stmt::Assign { target, value, .. } => {
                    if self.is_global_store(target) {
                        return false;
                    }
                    if let Expr::Call { callee, args } = value
                        && (callee == "dot" || callee == "dot_t")
                    {
                        let out_f32 = match target {
                            Expr::Var(n) => match scan.decls.get(n) {
                                Some((is_f32, _)) => *is_f32,
                                None => match self.lookup(n) {
                                    Some(Binding::Tile(mv)) => self.elem(mv) == Scalar::F32,
                                    Some(Binding::Frags(_)) => true,
                                    _ => false,
                                },
                            },
                            t @ Expr::Index { .. } => self.slice_tensor_elem(t) == Some(Scalar::F32),
                            _ => false,
                        };
                        self.hoist_consider(scan, cands, callee == "dot_t", args, out_f32);
                    }
                }
                Stmt::For { body, .. } => {
                    if !self.hoist_scan(body, scan, cands) {
                        return false;
                    }
                }
                Stmt::If { then, r#else, .. } => {
                    if self.writes_global(then)
                        || r#else.as_deref().is_some_and(|e| self.writes_global(e))
                    {
                        return false;
                    }
                }
                Stmt::While { body, .. } => {
                    if self.writes_global(body) {
                        return false;
                    }
                }
                Stmt::Expr(_) => {}
            }
        }
        true
    }

    fn hoist_consider(
        &self,
        scan: &HoistScan,
        cands: &mut Vec<ValueId>,
        transpose_b: bool,
        args: &[Expr],
        out_f32: bool,
    ) {
        if !out_f32 {
            return;
        }
        let [ae, be] = args else { return };
        let Some((_, ash)) = self.hoist_operand(scan, ae) else {
            return;
        };
        let Some((_, bsh)) = self.hoist_operand(scan, be) else {
            return;
        };
        let (&[m, kk], &[b0, b1]) = (&ash[..], &bsh[..]) else {
            return;
        };
        let (n, bk) = if transpose_b { (b0, b1) } else { (b1, b0) };
        if bk != kk || shape::wmma_plan(m, n, kk, self.cta_threads).is_none() {
            return;
        }
        for e in [ae, be] {
            if let Expr::Var(name) = e
                && !scan.declared.contains(name)
                && let Some(Binding::View(mv)) = self.lookup(name)
                && self.shape(mv).len() == 2
                && !self.shape(mv).contains(&DYN)
                && is_f16_or_f32(self.elem(mv))
                && self.is_global_mem(mv)
                && !cands.contains(&mv)
            {
                cands.push(mv);
            }
        }
    }

    fn hoist_record_decl(&self, scan: &mut HoistScan, name: &str, ty: Option<&AstType>, value: &Expr) {
        scan.declared.insert(name.to_string());
        let entry = match ty {
            Some(AstType::Tile(sc @ (AstScalar::F16 | AstScalar::F32), dims)) => {
                self.tile_shape(dims).ok().map(|s| (*sc == AstScalar::F32, s))
            }
            None => match (self.slice_static_shape(value), self.slice_tensor_elem(value)) {
                (Some(s), Some(e)) if is_f16_or_f32(e) => Some((e == Scalar::F32, s)),
                _ => None,
            },
            _ => None,
        };
        if let Some(e) = entry {
            scan.decls.insert(name.to_string(), e);
        }
    }

    fn hoist_operand(&self, scan: &HoistScan, expr: &Expr) -> Option<(bool, Vec<i64>)> {
        let Expr::Var(name) = expr else { return None };
        if let Some(e) = scan.decls.get(name) {
            return Some(e.clone());
        }
        if scan.declared.contains(name) {
            return None;
        }
        match self.lookup(name)? {
            Binding::Tile(mv) | Binding::View(mv)
                if !self.shape(mv).contains(&DYN) && is_f16_or_f32(self.elem(mv)) =>
            {
                Some((self.elem(mv) == Scalar::F32, self.shape(mv)))
            }
            _ => None,
        }
    }

    pub(crate) fn is_global_store(&self, target: &Expr) -> bool {
        matches!(
            target,
            Expr::Index { base, .. }
                if matches!(&**base, Expr::Var(t)
                    if matches!(self.lookup(t), Some(Binding::Tensor(_))))
        )
    }

    fn writes_global(&self, stmts: &[Stmt]) -> bool {
        stmts.iter().any(|s| match s {
            Stmt::Assign { target, .. } => self.is_global_store(target),
            Stmt::For { body, .. } | Stmt::While { body, .. } => self.writes_global(body),
            Stmt::If { then, r#else, .. } => {
                self.writes_global(then) || r#else.as_deref().is_some_and(|e| self.writes_global(e))
            }
            _ => false,
        })
    }

    /// Whether a value lives in global memory: a tensor or a slice of one.
    fn is_global_mem(&self, v: ValueId) -> bool {
        match self.ir.ty(v) {
            Type::Tensor(_) => true,
            Type::Tile(t) => t.space == Space::Global,
            _ => false,
        }
    }
}

pub(crate) fn is_f16_or_f32(s: Scalar) -> bool {
    matches!(s, Scalar::F16 | Scalar::F32)
}

/// The type `alloc_tile_swizzled` gives an f16 staging buffer of `shape`.
pub(crate) fn swizzled_ty(shape: &[i64]) -> Type {
    let width = *shape.last().expect("tile values are not rank-0");
    let blocks = (width / 8).max(1);
    let bits = blocks.trailing_zeros().min(3);
    Type::Tile(TileType {
        elem: Scalar::F16,
        shape: shape.iter().map(|&d| super::extent(d)).collect(),
        layout: Layout {
            row_stride: None,
            swizzle: (bits > 0).then_some(Swizzle {
                bits,
                shift: 0,
                elem_log: 3,
            }),
            align_div: super::alloc_align_div(shape),
        },
        space: Space::Shared,
    })
}
