use std::collections::HashSet;

use super::*;

/// Loop-invariant dot staging hoisted to the loop preheader.
///
/// The tensor-core dot paths (mma_sync_dot, the legacy wmma_dot body, and
/// frag_dot) stage both operands from global into shared f16 buffers on
/// every call. When the dot sits in a loop and an operand is a let-bound
/// slice of a global tensor defined outside it (the flash-attention q in
/// `dot_t(q, k)`), that copy is identical every iteration. emit_for scans
/// the body up front, stages such operands once into the block containing
/// the loop, and records (source value, buffer) on a per-loop stack frame;
/// the staging sites then reuse the buffer via [`Codegen::hoisted_stage`]
/// and skip their release (the loop epilogue returns it to the pool).
///
/// Correctness rests on three checks:
/// - The operand resolves to a Binding::View of global memory that is
///   already in scope when the loop is entered, so its subview (and every
///   index feeding it) dominates the loop.
/// - The body stores to no global memory at all (checked recursively,
///   including if/while), so nothing can invalidate the staged copy. Tile
///   writes cannot alias a tensor.
/// - Consumers match on the source's SSA value, not its name, so a body
///   that shadows the name simply misses the cache and stages in-loop.
///
/// The prescan mirrors the wmma_dot gates (wmma_plan on the dot's m/n/kk,
/// f32 output, f16/f32 operands) so a hoisted buffer is only ever emitted
/// for a dot that will actually run on the tensor cores; a gate drift only
/// wastes the preheader copy, never miscompiles, since the fallback paths
/// keep reading the original operand.
impl<'p, 'c> Codegen<'p, 'c> {
    /// Stages the body's hoistable dot operands into `block` (the loop's
    /// preheader) and returns the frame emit_for pushes for the loop. The
    /// closing barrier orders the staged writes before any tile-op write
    /// that follows, which a zero-trip loop would otherwise race.
    pub(super) fn hoist_dot_staging(
        &mut self,
        block: &Block<'c>,
        body: &[Stmt],
    ) -> Result<Vec<(Value<'c, 'c>, MemVal<'c>)>> {
        let mut frame = Vec::new();
        if !self.wmma() {
            return Ok(frame);
        }
        let Some(cands) = self.hoist_candidates(body) else {
            return Ok(frame);
        };
        for src in cands {
            // An enclosing loop already staged this source.
            if self.hoisted_stage(&src).is_some() {
                continue;
            }
            let buf = if self.mma_sync() {
                self.alloc_tile_swizzled(block, self.f16_t, &src.shape)?
            } else {
                self.alloc_tile_shaped(block, self.f16_t, &src.shape)?
            };
            self.stage_to_f16(block, &src, &buf, false)?;
            frame.push((src.mem, buf));
        }
        if !frame.is_empty() {
            self.barrier(block)?;
        }
        Ok(frame)
    }

    /// The preheader-staged f16 buffer for a dot operand, when a
    /// surrounding loop hoisted it. The caller skips both the in-loop
    /// staging and the release.
    pub(super) fn hoisted_stage(&self, src: &MemVal<'c>) -> Option<MemVal<'c>> {
        self.hoisted_stages
            .iter()
            .rev()
            .flatten()
            .find(|(v, _)| *v == src.mem)
            .map(|(_, buf)| buf.clone())
    }

    /// The body's hoistable dot operand views, or None when the body
    /// stores to global memory anywhere (a staged copy cannot see such
    /// writes, so nothing may be hoisted).
    fn hoist_candidates(&self, body: &[Stmt]) -> Option<Vec<MemVal<'c>>> {
        let mut scan = HoistScan::default();
        let mut cands = Vec::new();
        self.hoist_scan(body, &mut scan, &mut cands)
            .then_some(cands)
    }

    /// Collects hoistable operands into cands, returning false as soon as a
    /// global store makes the whole body ineligible.
    fn hoist_scan(
        &self,
        stmts: &[Stmt],
        scan: &mut HoistScan,
        cands: &mut Vec<MemVal<'c>>,
    ) -> bool {
        for stmt in stmts {
            match stmt {
                Stmt::Let { name, ty, value } | Stmt::Var { name, ty, value } => {
                    if let Expr::Call { callee, args } = value
                        && (callee == "dot" || callee == "dot_t")
                    {
                        let out_f32 = match ty {
                            Some(AstType::Tile(sc, _)) => *sc == Scalar::F32,
                            // Untyped: emit_call materializes into a buffer
                            // of the lhs operand's element type.
                            Some(_) => false,
                            None => args
                                .first()
                                .and_then(|a| self.hoist_operand(scan, a))
                                .is_some_and(|(is_f32, _)| is_f32),
                        };
                        self.hoist_consider(scan, cands, callee == "dot_t", args, out_f32);
                    }
                    self.hoist_record_decl(scan, name, ty.as_ref(), value);
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
                                    Some(Binding::Tile(mv)) => mv.elem == self.f32_t,
                                    Some(Binding::Frags(_)) => true,
                                    _ => false,
                                },
                            },
                            t @ Expr::Index { .. } => self.slice_tensor_elem(t) == Some(self.f32_t),
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
                // Dots inside if/while are not worth speculative staging,
                // but their bodies must still be write-free.
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

    /// Checks one dot against the tensor-core gates and collects any
    /// operand that is an in-scope global view of the right shape.
    fn hoist_consider(
        &self,
        scan: &HoistScan,
        cands: &mut Vec<MemVal<'c>>,
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
        if bk != kk || self.wmma_plan(m, n, kk).is_none() {
            return;
        }
        for e in [ae, be] {
            if let Expr::Var(name) = e
                && !scan.declared.contains(name)
                && let Some(Binding::View(mv)) = self.lookup(name)
                && mv.shape.len() == 2
                && !mv.shape.contains(&DYN)
                && self.is_f16_or_f32(mv.elem)
                && self.is_global_mem(&mv)
                && !cands.iter().any(|c| c.mem == mv.mem)
            {
                cands.push(mv);
            }
        }
    }

    /// Records an in-body declaration: its name (loop-variant, never
    /// hoistable) and, when resolvable, its element class and static shape
    /// for the dot gate checks.
    fn hoist_record_decl(
        &self,
        scan: &mut HoistScan,
        name: &str,
        ty: Option<&AstType>,
        value: &Expr,
    ) {
        scan.declared.insert(name.to_string());
        let entry = match ty {
            Some(AstType::Tile(sc @ (Scalar::F16 | Scalar::F32), dims)) => {
                self.tile_shape(dims).ok().map(|s| (*sc == Scalar::F32, s))
            }
            None => match (
                self.slice_static_shape(value),
                self.slice_tensor_elem(value),
            ) {
                (Some(s), Some(e)) if self.is_f16_or_f32(e) => Some((e == self.f32_t, s)),
                _ => None,
            },
            _ => None,
        };
        if let Some(e) = entry {
            scan.decls.insert(name.to_string(), e);
        }
    }

    /// A dot operand's element class and static shape: an in-body decl the
    /// walk has seen, or a tile/view binding already in scope.
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
                if !mv.shape.contains(&DYN) && self.is_f16_or_f32(mv.elem) =>
            {
                Some((mv.elem == self.f32_t, mv.shape))
            }
            _ => None,
        }
    }

    /// Whether the statement stores through a subscripted global tensor.
    fn is_global_store(&self, target: &Expr) -> bool {
        matches!(
            target,
            Expr::Index { base, .. }
                if matches!(&**base, Expr::Var(t)
                    if matches!(self.lookup(t), Some(Binding::Tensor(_))))
        )
    }

    /// Whether any statement (recursively) stores to global memory.
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

    /// True when a memref value lives in global memory (a tensor parameter
    /// or a slice of one); only those sources are safe to hoist, since tile
    /// writes inside the loop cannot alias them.
    fn is_global_mem(&self, mv: &MemVal<'c>) -> bool {
        MemRefType::try_from(mv.mem.r#type())
            .ok()
            .and_then(|t| t.memory_space())
            .is_some_and(|s| {
                let global: Attribute = IntegerAttribute::new(self.i64_t, MEM_GLOBAL).into();
                s == global
            })
    }
}

/// Names and shapes the body scan has resolved so far.
#[derive(Default)]
struct HoistScan {
    /// Every name the body declares; such operands are loop-variant.
    declared: HashSet<String>,
    /// Resolvable in-body decls as (is_f32, static shape).
    decls: HashMap<String, (bool, Vec<i64>)>,
}
