use anyhow::{Result, anyhow};

use super::{
    BlockId, Bounds, Coeff, Def, GemmType, Intrinsic, Ir, Map, OpId, OpKind, Scalar, Space, Type,
    ValueId,
};

/// Checks every structural invariant of the arena, the use lists,
/// dominance, terminators and each kind's typing rule. Runs after the
/// build, after every pass under test, and under `debug_assertions` in the
/// pipeline. Collects every violation rather than stopping at the first,
/// since a pass that broke one thing usually broke it in several places.
pub fn verify(ir: &Ir) -> Result<()> {
    let mut v = Verifier {
        ir,
        errors: Vec::new(),
    };
    v.structure();
    v.values();
    for op in ir.all_ops() {
        v.op(op);
    }
    if v.errors.is_empty() {
        Ok(())
    } else {
        Err(anyhow!("{}", v.errors.join("\n")))
    }
}

struct Verifier<'a> {
    ir: &'a Ir,
    errors: Vec<String>,
}

impl Verifier<'_> {
    fn err(&mut self, op: OpId, msg: impl Into<String>) {
        self.errors
            .push(format!("{op} ({}): {}", self.ir.kind(op).name(), msg.into()));
    }

    fn structure(&mut self) {
        let ir = self.ir;
        let entry = ir.entry();
        if ir.parent_op(entry).is_some() {
            self.errors.push(format!("entry {entry} has a parent op"));
        }
        // Every block is reachable from the entry and owned by the op that
        // lists it; every op sits in exactly one block, which lists it once.
        let mut reached = vec![false; ir.blocks.len()];
        let mut seen = vec![0usize; ir.ops.len()];
        let mut stack = vec![entry];
        while let Some(block) = stack.pop() {
            reached[block.index()] = true;
            for &op in ir.ops(block) {
                if !ir.is_alive(op) {
                    self.errors.push(format!("{block} lists erased {op}"));
                    continue;
                }
                seen[op.index()] += 1;
                if ir.parent_block(op) != block {
                    self.errors.push(format!(
                        "{op} sits in {block} but names {} as its parent",
                        ir.parent_block(op)
                    ));
                }
                for &inner in ir.blocks_of(op) {
                    if ir.parent_op(inner) != Some(op) {
                        self.errors
                            .push(format!("{inner} is owned by {op} but names another parent"));
                    }
                    if !ir.kind(op).has_blocks() {
                        self.err(op, "owns a block but is not a control-flow op");
                    }
                    stack.push(inner);
                }
            }
        }
        for (i, block) in ir.blocks.iter().enumerate() {
            if block.is_some() && !reached[i] {
                self.errors.push(format!("^{i} belongs to no op"));
            }
        }
        for (i, op) in ir.ops.iter().enumerate() {
            if op.is_some() && seen[i] != 1 {
                self.errors
                    .push(format!("op{i} appears in {} blocks, not one", seen[i]));
            }
        }
    }

    fn values(&mut self) {
        let ir = self.ir;
        for (i, value) in ir.values.iter().enumerate() {
            let Some(value) = value else { continue };
            let v = ValueId::from_index(i);
            match value.def {
                Def::Result { op, index } => {
                    if !ir.is_alive(op) || ir.results(op).get(index) != Some(&v) {
                        self.errors.push(format!("{v} names a definition that does not name it"));
                    }
                }
                Def::Arg { block, index } => {
                    if ir.blocks[block.index()].is_none() || ir.args(block).get(index) != Some(&v)
                    {
                        self.errors.push(format!("{v} names a block that does not list it"));
                    }
                }
            }
            for u in &value.uses {
                if !ir.is_alive(u.op) || ir.operands(u.op).get(u.index) != Some(&v) {
                    self.errors
                        .push(format!("{v} records a use at {}#{} that does not read it", u.op, u.index));
                }
            }
        }
        for op in ir.all_ops() {
            for (index, &v) in ir.operands(op).iter().enumerate() {
                let Some(value) = ir.values.get(v.index()).and_then(Option::as_ref) else {
                    self.err(op, format!("operand {index} is an erased value"));
                    continue;
                };
                let n = value
                    .uses
                    .iter()
                    .filter(|u| u.op == op && u.index == index)
                    .count();
                if n != 1 {
                    self.err(op, format!("{v} lists operand {index} {n} times, not once"));
                }
            }
        }
    }

    fn op(&mut self, op: OpId) {
        let ir = self.ir;
        for (index, &v) in ir.operands(op).iter().enumerate() {
            if ir.values.get(v.index()).is_some_and(Option::is_some) && !ir.dominates(v, op) {
                self.err(op, format!("operand {index} ({v}) is not visible here"));
            }
        }
        let block = ir.parent_block(op);
        let last = ir.ops(block).last() == Some(&op);
        if ir.kind(op).is_terminator() && !last {
            self.err(op, "a terminator that is not the last op of its block");
        }
        if ir.kind(op).is_terminator() && ir.parent_op(block).is_none() {
            self.err(op, "a terminator in the entry block");
        }
        self.kind(op);
    }

    // Typing helpers.

    fn scalar(&self, v: ValueId) -> Option<Scalar> {
        self.ir.ty(v).as_scalar()
    }

    fn is_tile(&self, v: ValueId) -> bool {
        matches!(self.ir.ty(v), Type::Tile(_))
    }

    fn is_mem(&self, v: ValueId) -> bool {
        matches!(self.ir.ty(v), Type::Tile(_) | Type::Tensor(_))
    }

    fn rank(&self, v: ValueId) -> usize {
        self.ir.ty(v).shape().map_or(0, <[_]>::len)
    }

    fn want_operands(&mut self, op: OpId, n: usize) -> bool {
        let got = self.ir.operands(op).len();
        if got != n {
            self.err(op, format!("expects {n} operands, has {got}"));
            return false;
        }
        true
    }

    fn want_results(&mut self, op: OpId, n: usize) -> bool {
        let got = self.ir.results(op).len();
        if got != n {
            self.err(op, format!("expects {n} results, has {got}"));
            return false;
        }
        true
    }

    fn want_tiles(&mut self, op: OpId, range: std::ops::Range<usize>) {
        for i in range {
            if let Some(&v) = self.ir.operands(op).get(i)
                && !self.is_tile(v)
            {
                self.err(op, format!("operand {i} must be a tile, is {}", self.ir.ty(v)));
            }
        }
    }

    fn want_tile_result(&mut self, op: OpId) {
        if self.want_results(op, 1) && !self.is_tile(self.ir.result(op)) {
            self.err(op, "result must be a tile");
        }
    }

    fn want_scalar(&mut self, op: OpId, i: usize, what: &str) -> Option<Scalar> {
        let v = *self.ir.operands(op).get(i)?;
        let s = self.scalar(v);
        if s.is_none() {
            self.err(op, format!("{what} must be a scalar, is {}", self.ir.ty(v)));
        }
        s
    }

    fn want_type(&mut self, op: OpId, i: usize, ty: &Type, what: &str) {
        if let Some(&v) = self.ir.operands(op).get(i)
            && self.ir.ty(v) != ty
        {
            self.err(op, format!("{what} must be {ty}, is {}", self.ir.ty(v)));
        }
    }

    /// A block's terminator must be `kind` with operands of `types`.
    /// Operand `i` as the gemm accumulator it must be.
    fn gemm_acc(&mut self, op: OpId, i: usize) -> Option<GemmType> {
        match self.ir.ty(*self.ir.operands(op).get(i)?) {
            Type::Gemm(g) => Some(*g),
            other => {
                self.err(op, format!("operand {i} must be a gemm accumulator, is {other}"));
                None
            }
        }
    }

    /// Operand `i` as a `shape` tile the register matmul's paths may index
    /// with no bounds guard: an unmasked tensor slice, or, where the loop
    /// stages it anyway, the shared tile a masked one was materialized into.
    fn want_gemm_slice(&mut self, op: OpId, i: usize, shape: &[i64], shared_ok: bool) {
        let Some(&v) = self.ir.operands(op).get(i) else { return };
        let masked = self
            .ir
            .def_op(v)
            .is_some_and(|d| matches!(self.ir.kind(d), OpKind::Slice(s) if s.masked.iter().any(|&m| m)));
        match self.ir.ty(v) {
            Type::Tile(t) if (t.space == Space::Global && !masked) || (t.space == Space::Shared && shared_ok) => {
                if t.static_shape().as_deref() != Some(shape) {
                    self.err(op, format!("operand {i} must be a {shape:?} tile, is {}", self.ir.ty(v)));
                }
            }
            other if shared_ok => {
                self.err(op, format!("operand {i} must be an unmasked slice or a shared tile, is {other}"))
            }
            other => self.err(op, format!("operand {i} must be an unmasked global slice, is {other}")),
        }
    }

    fn want_terminator(&mut self, op: OpId, block: BlockId, kind: &OpKind, types: &[Type]) {
        let Some(&term) = self.ir.ops(block).last() else {
            self.err(op, format!("{block} is empty and has no terminator"));
            return;
        };
        if self.ir.kind(term) != kind {
            self.err(
                op,
                format!("{block} ends in {} rather than {}", self.ir.kind(term).name(), kind.name()),
            );
            return;
        }
        let operands = self.ir.operands(term);
        if operands.len() != types.len() {
            self.err(
                op,
                format!("{block} yields {} values, {} expected", operands.len(), types.len()),
            );
            return;
        }
        for (i, (&v, ty)) in operands.iter().zip(types).enumerate() {
            if self.ir.ty(v) != ty {
                self.err(op, format!("{block} yields {} at {i}, {ty} expected", self.ir.ty(v)));
            }
        }
    }

    fn want_block_args(&mut self, op: OpId, block: BlockId, types: &[Type]) {
        let args = self.ir.args(block);
        if args.len() != types.len() {
            self.err(
                op,
                format!("{block} takes {} arguments, {} expected", args.len(), types.len()),
            );
            return;
        }
        for (i, (&a, ty)) in args.iter().zip(types).enumerate() {
            if self.ir.ty(a) != ty {
                self.err(op, format!("{block} argument {i} is {}, {ty} expected", self.ir.ty(a)));
            }
        }
    }

    fn kind(&mut self, op: OpId) {
        let ir = self.ir;
        let kind = ir.kind(op).clone();
        let n_ops = ir.operands(op).len();
        let n_blocks = ir.blocks_of(op).len();
        if !kind.has_blocks() && n_blocks != 0 {
            self.err(op, "owns blocks");
        }
        match &kind {
            OpKind::Const(lit) => {
                self.want_operands(op, 0);
                if self.want_results(op, 1) {
                    let want = match lit {
                        super::Literal::Int(_) => Scalar::Index,
                        super::Literal::Float(_) => Scalar::F32,
                        super::Literal::Bool(_) => Scalar::Bool,
                    };
                    if self.scalar(ir.result(op)) != Some(want) {
                        self.err(op, format!("result must be {want}"));
                    }
                }
            }
            OpKind::Unary(un) => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    let s = self.want_scalar(op, 0, "operand");
                    let ok = match (un, s) {
                        (super::UnOp::Neg, Some(s)) => s.is_numeric(),
                        (super::UnOp::Not, Some(s)) => s == Scalar::Bool,
                        _ => true,
                    };
                    if !ok {
                        self.err(op, "operand type does not take this operator");
                    }
                    if ir.ty(ir.result(op)) != ir.ty(ir.operand(op, 0)) {
                        self.err(op, "result type differs from the operand's");
                    }
                }
            }
            OpKind::Binary(bin) => {
                if self.want_operands(op, 2) && self.want_results(op, 1) {
                    let (a, b) = (self.want_scalar(op, 0, "lhs"), self.want_scalar(op, 1, "rhs"));
                    if let (Some(a), Some(b)) = (a, b) {
                        if a != b {
                            self.err(op, format!("operands differ: {a} vs {b}"));
                        }
                        let want = if bin.is_compare() { Scalar::Bool } else { a };
                        if self.scalar(ir.result(op)) != Some(want) {
                            self.err(op, format!("result must be {want}"));
                        }
                        if a == Scalar::Bool && !matches!(bin, super::BinOp::Eq | super::BinOp::Ne) {
                            self.err(op, "bool operands take only eq and ne");
                        }
                        if !a.is_numeric() && a != Scalar::Bool {
                            self.err(op, "operands are not arithmetic");
                        }
                    }
                }
            }
            OpKind::IndexOp(iop) => {
                if self.want_operands(op, 2) && self.want_results(op, 1) {
                    self.want_type(op, 0, &Type::INDEX, "lhs");
                    self.want_type(op, 1, &Type::INDEX, "rhs");
                    let want = match iop {
                        super::IndexOp::CmpUlt => Scalar::Bool,
                        _ => Scalar::Index,
                    };
                    if self.scalar(ir.result(op)) != Some(want) {
                        self.err(op, format!("result must be {want}"));
                    }
                }
            }
            OpKind::Cast(want) => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    if let Some(s) = self.want_scalar(op, 0, "operand")
                        && !(s.is_numeric() && want.is_numeric())
                    {
                        self.err(op, format!("cannot convert {s} to {want}"));
                    }
                    if self.scalar(ir.result(op)) != Some(*want) {
                        self.err(op, format!("result must be {want}"));
                    }
                }
            }
            OpKind::IndexCast(want) => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    if let Some(s) = self.want_scalar(op, 0, "operand") {
                        let ok = (s == Scalar::Index && want.is_int())
                            || (s.is_int() && *want == Scalar::Index);
                        if !ok {
                            self.err(op, format!("index_cast does not go from {s} to {want}"));
                        }
                    }
                    if self.scalar(ir.result(op)) != Some(*want) {
                        self.err(op, format!("result must be {want}"));
                    }
                }
            }
            OpKind::ProgramId(d) => {
                self.want_operands(op, 0);
                if *d > 2 {
                    self.err(op, "dimension past z");
                }
                if self.want_results(op, 1) && self.scalar(ir.result(op)) != Some(Scalar::Index) {
                    self.err(op, "result must be index");
                }
            }
            OpKind::Dim(d) => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    match ir.ty(ir.operand(op, 0)) {
                        Type::Tensor(t) if *d < t.shape.len() => {}
                        Type::Tensor(_) => self.err(op, "dimension past the tensor's rank"),
                        other => self.err(op, format!("operand must be a tensor, is {other}")),
                    }
                    if self.scalar(ir.result(op)) != Some(Scalar::Index) {
                        self.err(op, "result must be index");
                    }
                }
            }
            OpKind::Load => {
                if n_ops == 0 || !self.is_mem(ir.operand(op, 0)) {
                    self.err(op, "first operand must be a tensor or tile");
                } else {
                    let mem = ir.operand(op, 0);
                    let rank = self.rank(mem);
                    if self.want_operands(op, 1 + rank) {
                        for i in 1..=rank {
                            self.want_type(op, i, &Type::INDEX, "index");
                        }
                    }
                    if self.want_results(op, 1) && self.scalar(ir.result(op)) != ir.ty(mem).elem() {
                        self.err(op, "result must be the element type");
                    }
                }
            }
            OpKind::Store => {
                if n_ops < 2 || !self.is_mem(ir.operand(op, 1)) {
                    self.err(op, "second operand must be a tensor or tile");
                } else {
                    let mem = ir.operand(op, 1);
                    let rank = self.rank(mem);
                    if self.want_operands(op, 2 + rank) {
                        for i in 2..2 + rank {
                            self.want_type(op, i, &Type::INDEX, "index");
                        }
                    }
                    if self.scalar(ir.operand(op, 0)) != ir.ty(mem).elem() {
                        self.err(op, "stored value must be the element type");
                    }
                }
                self.want_results(op, 0);
            }
            OpKind::AtomicAdd => {
                if self.want_operands(op, 3) {
                    match ir.ty(ir.operand(op, 0)) {
                        Type::Tensor(t) if t.elem == Scalar::I32 => {}
                        other => self.err(op, format!("operand 0 must be an i32 tensor, is {other}")),
                    }
                    self.want_type(op, 1, &Type::INDEX, "slot");
                    self.want_type(op, 2, &Type::Scalar(Scalar::I32), "value");
                }
                if self.want_results(op, 1) && self.scalar(ir.result(op)) != Some(Scalar::I32) {
                    self.err(op, "result must be i32");
                }
            }
            OpKind::For(info) => {
                if !self.want_operands(op, info.operand_count()) {
                    return;
                }
                let bounds = info.bound_operands();
                for (i, what) in ["lo", "hi", "step", "hi"].into_iter().enumerate().take(bounds) {
                    self.want_type(op, i, &Type::INDEX, what);
                }
                if info.ragged && !matches!(info.bounds, Bounds::Dynamic) {
                    self.err(op, "a ragged loop has dynamic bounds");
                }
                let carried: Vec<Type> = ir.operands(op)[bounds..bounds + info.carried]
                    .iter()
                    .map(|&v| ir.ty(v).clone())
                    .collect();
                for i in bounds + info.carried..n_ops {
                    match ir.ty(ir.operand(op, i)) {
                        Type::Tile(t) if t.space == Space::Shared => {}
                        other => self.err(op, format!("hoisted operand {i} must be a shared tile, is {other}")),
                    }
                }
                let results: Vec<Type> = ir.results(op).iter().map(|&v| ir.ty(v).clone()).collect();
                if results != carried {
                    self.err(op, "results must match the carried operands");
                }
                if carried.iter().any(|t| matches!(t, Type::Gemm(_)))
                    && (info.carried != 1 || info.hoisted != 0 || info.ragged || info.pipeline.is_some())
                {
                    self.err(op, "a gemm accumulator is carried alone, with nothing hoisted or pipelined");
                }
                let want_blocks = 1 + usize::from(info.ragged);
                if n_blocks != want_blocks {
                    self.err(op, format!("owns {want_blocks} blocks, has {n_blocks}"));
                    return;
                }
                let body = ir.blocks_of(op)[0];
                let mut args = vec![Type::INDEX];
                args.extend(carried.iter().cloned());
                self.want_block_args(op, body, &args);
                self.want_terminator(op, body, &OpKind::Yield, &carried);
                if info.ragged {
                    let replay = ir.blocks_of(op)[1];
                    self.want_block_args(op, replay, &[]);
                    self.want_terminator(op, replay, &OpKind::Yield, &[]);
                }
            }
            OpKind::While => {
                let carried: Vec<Type> = ir.operands(op).iter().map(|&v| ir.ty(v).clone()).collect();
                let results: Vec<Type> = ir.results(op).iter().map(|&v| ir.ty(v).clone()).collect();
                if results != carried {
                    self.err(op, "results must match the carried operands");
                }
                if n_blocks != 2 {
                    self.err(op, "owns a before and an after block");
                    return;
                }
                let (before, after) = (ir.blocks_of(op)[0], ir.blocks_of(op)[1]);
                self.want_block_args(op, before, &carried);
                let mut cond = vec![Type::BOOL];
                cond.extend(carried.iter().cloned());
                self.want_terminator(op, before, &OpKind::Condition, &cond);
                self.want_block_args(op, after, &carried);
                self.want_terminator(op, after, &OpKind::Yield, &carried);
            }
            OpKind::If => {
                if self.want_operands(op, 1) {
                    self.want_type(op, 0, &Type::BOOL, "condition");
                }
                let results: Vec<Type> = ir.results(op).iter().map(|&v| ir.ty(v).clone()).collect();
                if n_blocks == 0 || n_blocks > 2 {
                    self.err(op, "owns a then block and at most an else block");
                    return;
                }
                if n_blocks == 1 && !results.is_empty() {
                    self.err(op, "yields results but has no else block");
                }
                for &block in ir.blocks_of(op) {
                    self.want_block_args(op, block, &[]);
                    self.want_terminator(op, block, &OpKind::Yield, &results);
                }
            }
            OpKind::Yield | OpKind::Condition => {
                self.want_results(op, 0);
                let parent = ir.parent_op(ir.parent_block(op));
                let ok = matches!(
                    (parent.map(|p| ir.kind(p)), &kind),
                    (Some(OpKind::For(_) | OpKind::If), OpKind::Yield) | (Some(OpKind::While), _)
                );
                if !ok {
                    self.err(op, "does not end a block of the op that takes it");
                }
            }
            OpKind::Barrier => {
                self.want_operands(op, 0);
                self.want_results(op, 0);
            }
            OpKind::GridBarrier => {
                if self.want_operands(op, 1) {
                    match ir.ty(ir.operand(op, 0)) {
                        Type::Tensor(t) if t.elem == Scalar::I32 => {}
                        other => self.err(op, format!("operand must be an i32 tensor, is {other}")),
                    }
                }
                if self.want_results(op, 1) && self.scalar(ir.result(op)) != Some(Scalar::Index) {
                    self.err(op, "result must be index");
                }
            }
            OpKind::GemmInit => {
                if self.want_operands(op, 1) {
                    self.want_scalar(op, 0, "init");
                }
                if self.want_results(op, 1) && !matches!(ir.ty(ir.result(op)), Type::Gemm(_)) {
                    self.err(op, "result must be a gemm accumulator");
                }
            }
            OpKind::GemmDot => {
                if self.want_operands(op, 3)
                    && let Some(g) = self.gemm_acc(op, 0)
                {
                    self.want_gemm_slice(op, 1, &[g.m, g.k], true);
                    self.want_gemm_slice(op, 2, &[g.k, g.n], true);
                }
                if self.want_results(op, 1) && ir.ty(ir.result(op)) != ir.ty(ir.operand(op, 0)) {
                    self.err(op, "result must be the accumulator's type");
                }
            }
            OpKind::GemmStore { alpha, beta } => {
                if self.want_operands(op, 2 + alpha.operands() + beta.operands())
                    && let Some(g) = self.gemm_acc(op, 0)
                {
                    self.want_gemm_slice(op, 1, &[g.m, g.n], false);
                    let mut i = 2;
                    for (c, what) in [(alpha, "alpha"), (beta, "beta")] {
                        if *c == Coeff::Given {
                            self.want_scalar(op, i, what);
                            i += 1;
                        }
                    }
                }
                self.want_results(op, 0);
            }
            OpKind::FragInit(_) => {
                self.want_operands(op, 0);
                if self.want_results(op, 1) && !matches!(ir.ty(ir.result(op)), Type::Frags(_)) {
                    self.err(op, "result must be fragments");
                }
            }
            OpKind::FragScale(_) | OpKind::FragDot | OpKind::FragStore => {
                let n = match kind {
                    OpKind::FragDot => 3,
                    _ => 2,
                };
                if self.want_operands(op, n) {
                    if !matches!(ir.ty(ir.operand(op, 0)), Type::Frags(_)) {
                        self.err(op, "operand 0 must be fragments");
                    }
                    self.want_tiles(op, 1..n);
                }
                if matches!(kind, OpKind::FragStore) {
                    self.want_results(op, 0);
                } else if self.want_results(op, 1)
                    && ir.ty(ir.result(op)) != ir.ty(ir.operand(op, 0))
                {
                    self.err(op, "result must be the accumulator's type");
                }
            }
            OpKind::AssumeAlign => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    let (src, out) = (ir.ty(ir.operand(op, 0)), ir.ty(ir.result(op)));
                    if !matches!(src, Type::Tensor(_)) || src != out {
                        self.err(op, "takes a tensor and yields the same type");
                    }
                }
            }
            OpKind::Alloc => {
                self.want_operands(op, 0);
                if self.want_results(op, 1) {
                    match ir.ty(ir.result(op)) {
                        Type::Tile(t) if t.space == Space::Global => {
                            self.err(op, "cannot allocate in global memory")
                        }
                        Type::Tile(t) if !t.is_static() => self.err(op, "shape must be static"),
                        Type::Tile(_) => {}
                        other => self.err(op, format!("result must be a tile, is {other}")),
                    }
                }
            }
            OpKind::Slice(s) => {
                if s.masked.len() != s.rank() {
                    self.err(op, "mask and sizes differ in rank");
                }
                if !self.want_operands(op, s.operand_count()) {
                    return;
                }
                let src = ir.operand(op, 0);
                if !self.is_mem(src) {
                    self.err(op, "source must be a tensor or tile");
                } else if self.rank(src) != s.rank() {
                    self.err(op, "rank differs from the source's");
                }
                for i in 1..s.operand_count() {
                    self.want_type(op, i, &Type::INDEX, "offset, size or extent");
                }
                if self.want_results(op, 1) {
                    match ir.ty(ir.result(op)) {
                        Type::Tile(t) => {
                            if t.shape != s.sizes {
                                self.err(op, "result shape differs from the sizes");
                            }
                            let want = match ir.ty(src) {
                                Type::Tensor(_) => Space::Global,
                                Type::Tile(st) => st.space,
                                _ => Space::Global,
                            };
                            if t.space != want {
                                self.err(op, format!("result must live in {want} memory"));
                            }
                            if Some(t.elem) != ir.ty(src).elem() {
                                self.err(op, "result element type differs from the source's");
                            }
                        }
                        other => self.err(op, format!("result must be a tile, is {other}")),
                    }
                }
            }
            OpKind::Flat => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    self.want_tiles(op, 0..1);
                    let (src, out) = (ir.ty(ir.operand(op, 0)), ir.ty(ir.result(op)));
                    if let (Type::Tile(s), Type::Tile(o)) = (src, out) {
                        let want = s.static_shape().filter(|sh| sh.len() == 2).map(|sh| sh[0] * sh[1]);
                        let got = o.static_shape().filter(|sh| sh.len() == 2 && sh[0] == 1).map(|sh| sh[1]);
                        if want.is_none() || want != got || s.elem != o.elem || s.space != o.space {
                            self.err(op, "result must be the source viewed as one row");
                        }
                    } else {
                        self.err(op, "result must be a tile");
                    }
                }
            }
            OpKind::Stage(_) | OpKind::Materialize | OpKind::HoistStage => {
                if self.want_operands(op, 1) && self.want_results(op, 1) {
                    self.want_tiles(op, 0..1);
                    if let (Type::Tile(s), Type::Tile(o)) = (ir.ty(ir.operand(op, 0)), ir.ty(ir.result(op))) {
                        if o.space != Space::Shared {
                            self.err(op, "result must be a shared tile");
                        }
                        if o.shape != s.shape {
                            self.err(op, "result must have the source's shape");
                        }
                        let elem_ok = match kind {
                            OpKind::HoistStage => o.elem == Scalar::F16,
                            _ => o.elem == s.elem,
                        };
                        if !elem_ok {
                            self.err(op, "result has the wrong element type");
                        }
                    } else {
                        self.err(op, "result must be a tile");
                    }
                }
            }
            OpKind::Copy { .. } | OpKind::Convert | OpKind::Chain(_) | OpKind::Accumulate => {
                if self.want_operands(op, 2) {
                    self.want_tiles(op, 0..2);
                    if matches!(kind, OpKind::Copy { .. })
                        && ir.ty(ir.operand(op, 0)).elem() != ir.ty(ir.operand(op, 1)).elem()
                    {
                        self.err(op, "copy between element types; use convert");
                    }
                }
                self.want_results(op, 0);
            }
            OpKind::Fill => {
                if self.want_operands(op, 2) {
                    self.want_tiles(op, 1..2);
                    if self.scalar(ir.operand(op, 0)) != ir.ty(ir.operand(op, 1)).elem() {
                        self.err(op, "value must be the destination's element type");
                    }
                }
                self.want_results(op, 0);
            }
            OpKind::Map(m) | OpKind::MapInto(m) => {
                let into = matches!(kind, OpKind::MapInto(_));
                let n = m.tile_operands() + usize::from(matches!(m, Map::Scalar { .. })) + usize::from(into);
                if self.want_operands(op, n) {
                    match m {
                        Map::Scalar { .. } => {
                            self.want_tiles(op, 0..1);
                            self.want_scalar(op, 1, "scalar");
                        }
                        _ => self.want_tiles(op, 0..m.tile_operands()),
                    }
                    if into {
                        self.want_tiles(op, n - 1..n);
                    }
                }
                if into {
                    self.want_results(op, 0);
                } else {
                    self.want_tile_result(op);
                }
            }
            OpKind::Fused(tree) => {
                let mut leaves = Vec::new();
                tree.operands(&mut leaves);
                if n_ops < 2 {
                    self.err(op, "needs at least one leaf and a destination");
                    return;
                }
                self.want_tiles(op, n_ops - 1..n_ops);
                let mut read = Vec::new();
                tree.operands(&mut read);
                for i in read {
                    if i >= n_ops - 1 {
                        self.err(op, format!("tree reads operand {i}, past the leaves"));
                    }
                }
                self.check_tree(op, tree);
                self.want_results(op, 0);
            }
            OpKind::ScaledAdd => {
                if self.want_operands(op, 5) {
                    self.want_scalar(op, 0, "s1");
                    self.want_scalar(op, 2, "s2");
                    self.want_tiles(op, 1..2);
                    self.want_tiles(op, 3..5);
                }
                self.want_results(op, 0);
            }
            OpKind::Reduce(_) => {
                if self.want_operands(op, 1) {
                    self.want_tiles(op, 0..1);
                }
                self.want_tile_result(op);
            }
            OpKind::Dot { .. } => {
                if self.want_operands(op, 2) {
                    self.want_tiles(op, 0..2);
                }
                self.want_tile_result(op);
            }
            OpKind::DotInto { .. } => {
                if self.want_operands(op, 3) {
                    self.want_tiles(op, 0..3);
                }
                self.want_results(op, 0);
            }
            OpKind::Intrinsic(i) => {
                if matches!(i, Intrinsic::RawQdecode(_)) {
                    self.err(op, "only writes a destination; use the into form");
                }
                self.intrinsic_operands(op, *i, n_ops);
                match i {
                    Intrinsic::RmsNormQ => {
                        if self.want_results(op, 1) && self.scalar(ir.result(op)) != Some(Scalar::F32) {
                            self.err(op, "result must be the f32 inverse rms");
                        }
                    }
                    Intrinsic::WarpPartial => {
                        if self.want_results(op, 1) && self.scalar(ir.result(op)) != Some(Scalar::Index) {
                            self.err(op, "result must be index");
                        }
                    }
                    _ => self.want_tile_result(op),
                }
            }
            OpKind::IntrinsicInto(i) => {
                if n_ops == 0 {
                    self.err(op, "needs a destination");
                } else {
                    self.intrinsic_operands(op, *i, n_ops - 1);
                    self.want_tiles(op, n_ops - 1..n_ops);
                }
                self.want_results(op, 0);
            }
        }
    }

    /// An intrinsic's own operands, before any destination. Each intrinsic
    /// checks its own shapes when it is emitted; here only what every one
    /// shares: no fragments, and `rms_norm_q_t`'s `eps` is a scalar.
    fn intrinsic_operands(&mut self, op: OpId, i: Intrinsic, n: usize) {
        for k in 0..n {
            let v = self.ir.operand(op, k);
            if matches!(i, Intrinsic::RmsNormQ) && k == 2 {
                self.want_scalar(op, k, "eps");
            } else if matches!(self.ir.ty(v), Type::Frags(_) | Type::Gemm(_)) {
                self.err(op, format!("operand {k} cannot be an accumulator"));
            }
        }
    }

    fn check_tree(&mut self, op: OpId, tree: &super::Tree) {
        match tree {
            super::Tree::Leaf(i) => {
                if let Some(&v) = self.ir.operands(op).get(*i)
                    && !self.is_tile(v)
                {
                    self.err(op, format!("tree leaf {i} must be a tile"));
                }
            }
            super::Tree::Scalar(i) => {
                if let Some(&v) = self.ir.operands(op).get(*i)
                    && self.scalar(v).is_none()
                {
                    self.err(op, format!("tree scalar {i} must be a scalar"));
                }
            }
            super::Tree::Unary(_, x) => self.check_tree(op, x),
            super::Tree::Binary(_, a, b) | super::Tree::Max(a, b) => {
                self.check_tree(op, a);
                self.check_tree(op, b);
            }
        }
    }
}
