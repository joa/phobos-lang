use super::{BlockId, Ir, OpId, OpKind, Type, ValueId};

/// An insertion point, resolved to a block and index at the time of use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum At {
    End(BlockId),
    Start(BlockId),
    Before(OpId),
    After(OpId),
    Index(BlockId, usize),
}

/// Appends ops at a moving insertion point. `at` is re-resolved for every
/// op, so a builder positioned before an op stays before it as ops are
/// added, and one at the end of a block follows the end.
pub struct Builder<'a> {
    pub ir: &'a mut Ir,
    at: At,
}

impl<'a> Builder<'a> {
    pub fn new(ir: &'a mut Ir, at: At) -> Builder<'a> {
        Builder { ir, at }
    }

    pub fn at_end(ir: &'a mut Ir, block: BlockId) -> Builder<'a> {
        Builder::new(ir, At::End(block))
    }

    pub fn set_at(&mut self, at: At) {
        self.at = at;
    }

    pub fn at(&self) -> At {
        self.at
    }

    /// The block the next op lands in.
    pub fn block(&self) -> BlockId {
        match self.at {
            At::End(b) | At::Start(b) | At::Index(b, _) => b,
            At::Before(op) | At::After(op) => self.ir.parent_block(op),
        }
    }

    /// Creates an op at the insertion point. A builder positioned `After`
    /// an op advances past what it just made, so a run of ops stays in
    /// creation order.
    pub fn op(
        &mut self,
        kind: OpKind,
        operands: &[ValueId],
        result_types: Vec<Type>,
        blocks: Vec<BlockId>,
    ) -> OpId {
        let op = self.ir.create_op(self.at, kind, operands, result_types, blocks);
        if let At::After(_) = self.at {
            self.at = At::After(op);
        }
        op
    }

    /// An op with one result, returning the result.
    pub fn value(&mut self, kind: OpKind, operands: &[ValueId], ty: Type) -> ValueId {
        let op = self.op(kind, operands, vec![ty], Vec::new());
        self.ir.result(op)
    }

    /// An op with no results.
    pub fn stmt(&mut self, kind: OpKind, operands: &[ValueId]) -> OpId {
        self.op(kind, operands, Vec::new(), Vec::new())
    }

    /// A fresh block for an op this builder is about to create.
    pub fn block_with(&mut self, arg_types: &[Type]) -> BlockId {
        self.ir.new_block(arg_types)
    }

    /// Runs `f` with a builder at the end of `block`, then returns here.
    pub fn in_block<T>(&mut self, block: BlockId, f: impl FnOnce(&mut Builder<'_>) -> T) -> T {
        let mut inner = Builder::at_end(self.ir, block);
        f(&mut inner)
    }
}
