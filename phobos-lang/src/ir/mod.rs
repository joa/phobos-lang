pub mod build;
pub mod builder;
pub mod ops;
pub mod print;
pub mod types;
pub mod verify;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

pub use builder::{At, Builder};
pub use ops::*;
pub use types::*;

macro_rules! define_id {
    ($name:ident, $prefix:literal, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(u32);

        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }

            fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).expect("arena index fits u32"))
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}{}", $prefix, self.0)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}{}", $prefix, self.0)
            }
        }
    };
}

define_id!(OpId, "op", "An operation in the arena.");
define_id!(ValueId, "%", "An SSA value: an op result or a block argument.");
define_id!(BlockId, "^", "A block: an argument list and ops in program order.");

/// One operand position of one op.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Use {
    pub op: OpId,
    pub index: usize,
}

/// Where a value comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Def {
    Result { op: OpId, index: usize },
    Arg { block: BlockId, index: usize },
}

#[derive(Debug)]
struct OpData {
    kind: OpKind,
    operands: Vec<ValueId>,
    results: Vec<ValueId>,
    blocks: Vec<BlockId>,
    parent: BlockId,
}

#[derive(Debug)]
struct ValueData {
    ty: Type,
    def: Def,
    uses: Vec<Use>,
    /// The source name the value was bound to, when it had one. Only ever
    /// informative: the printer shows it, and the lowering treats a named
    /// tile as one the source may read again.
    name: Option<String>,
}

#[derive(Debug)]
struct BlockData {
    args: Vec<ValueId>,
    ops: Vec<OpId>,
    /// None only for the entry block, and for a block created but not yet
    /// handed to an op.
    parent: Option<OpId>,
}

/// What a kernel is, apart from its body: everything the attributes said.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelInfo {
    pub name: String,
    pub launch: Option<crate::ast::Launch>,
    pub cta_threads: i64,
    pub dynamic_shared: bool,
    pub pad_stage: bool,
    pub tensorcore: bool,
    pub mma_sync: bool,
    pub pipeline_assert: bool,
    /// `@autotune` symbols resolved to their chosen value.
    pub shape_env: BTreeMap<String, i64>,
}

/// One kernel's graph: an SSA arena with structured control flow, and the
/// home of every analysis and rewrite that runs before the MLIR lowering.
///
/// Ids are the only handles; nothing hands out a reference into the arena
/// that outlives a call. `For`, `While` and `If` own their blocks, so a
/// value dominates a use when its defining op precedes the use in the same
/// block or in an enclosing one. The mutation methods below are the only
/// way to change an operand, and each leaves operands and use lists
/// agreeing. An erased op or value leaves a tombstone, so a stale id panics
/// instead of resolving to something else.
#[derive(Debug)]
pub struct Ir {
    ops: Vec<Option<OpData>>,
    values: Vec<Option<ValueData>>,
    blocks: Vec<Option<BlockData>>,
    entry: BlockId,
    pub kernel: KernelInfo,
}

impl Ir {
    /// A graph with an entry block whose arguments are the kernel's
    /// parameters, in order.
    pub fn new(kernel: KernelInfo, params: &[Type]) -> Ir {
        let mut ir = Ir {
            ops: Vec::new(),
            values: Vec::new(),
            blocks: Vec::new(),
            entry: BlockId(0),
            kernel,
        };
        ir.entry = ir.new_block(params);
        ir
    }

    pub fn entry(&self) -> BlockId {
        self.entry
    }

    // Accessors. Each panics on a tombstone: a stale id is a bug in the
    // pass holding it.

    fn op_data(&self, op: OpId) -> &OpData {
        self.ops
            .get(op.index())
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("{op} is erased or was never created"))
    }

    fn op_data_mut(&mut self, op: OpId) -> &mut OpData {
        self.ops
            .get_mut(op.index())
            .and_then(Option::as_mut)
            .unwrap_or_else(|| panic!("{op} is erased or was never created"))
    }

    fn value_data(&self, value: ValueId) -> &ValueData {
        self.values
            .get(value.index())
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("{value} is erased or was never created"))
    }

    fn value_data_mut(&mut self, value: ValueId) -> &mut ValueData {
        self.values
            .get_mut(value.index())
            .and_then(Option::as_mut)
            .unwrap_or_else(|| panic!("{value} is erased or was never created"))
    }

    fn block_data(&self, block: BlockId) -> &BlockData {
        self.blocks
            .get(block.index())
            .and_then(Option::as_ref)
            .unwrap_or_else(|| panic!("{block} is erased or was never created"))
    }

    fn block_data_mut(&mut self, block: BlockId) -> &mut BlockData {
        self.blocks
            .get_mut(block.index())
            .and_then(Option::as_mut)
            .unwrap_or_else(|| panic!("{block} is erased or was never created"))
    }

    pub fn is_alive(&self, op: OpId) -> bool {
        self.ops.get(op.index()).is_some_and(Option::is_some)
    }

    pub fn kind(&self, op: OpId) -> &OpKind {
        &self.op_data(op).kind
    }

    /// The kind is the op's attribute store, so a pass that records a
    /// decision (an offset, a plan) writes it here. Operands, results and
    /// blocks are not reachable this way.
    pub fn kind_mut(&mut self, op: OpId) -> &mut OpKind {
        &mut self.op_data_mut(op).kind
    }

    pub fn operands(&self, op: OpId) -> &[ValueId] {
        &self.op_data(op).operands
    }

    pub fn operand(&self, op: OpId, index: usize) -> ValueId {
        self.op_data(op).operands[index]
    }

    pub fn results(&self, op: OpId) -> &[ValueId] {
        &self.op_data(op).results
    }

    /// The single result of an op that has exactly one.
    pub fn result(&self, op: OpId) -> ValueId {
        match self.op_data(op).results[..] {
            [v] => v,
            ref rs => panic!("{op} has {} results, not one", rs.len()),
        }
    }

    pub fn blocks_of(&self, op: OpId) -> &[BlockId] {
        &self.op_data(op).blocks
    }

    pub fn parent_block(&self, op: OpId) -> BlockId {
        self.op_data(op).parent
    }

    pub fn parent_op(&self, block: BlockId) -> Option<OpId> {
        self.block_data(block).parent
    }

    pub fn ops(&self, block: BlockId) -> &[OpId] {
        &self.block_data(block).ops
    }

    pub fn args(&self, block: BlockId) -> &[ValueId] {
        &self.block_data(block).args
    }

    /// The op's index within its block.
    pub fn position(&self, op: OpId) -> usize {
        let block = self.parent_block(op);
        self.ops(block)
            .iter()
            .position(|&o| o == op)
            .unwrap_or_else(|| panic!("{op} is not in its parent {block}"))
    }

    pub fn ty(&self, value: ValueId) -> &Type {
        &self.value_data(value).ty
    }

    pub fn def(&self, value: ValueId) -> Def {
        self.value_data(value).def
    }

    /// The op defining a value, None for a block argument.
    pub fn def_op(&self, value: ValueId) -> Option<OpId> {
        match self.def(value) {
            Def::Result { op, .. } => Some(op),
            Def::Arg { .. } => None,
        }
    }

    pub fn uses(&self, value: ValueId) -> &[Use] {
        &self.value_data(value).uses
    }

    pub fn name(&self, value: ValueId) -> Option<&str> {
        self.value_data(value).name.as_deref()
    }

    pub fn set_name(&mut self, value: ValueId, name: impl Into<String>) {
        self.value_data_mut(value).name = Some(name.into());
    }

    /// The block's ancestors from the nearest outwards, as (op, block)
    /// pairs: the op that owns the block and the block that op sits in.
    pub fn ancestors(&self, block: BlockId) -> Vec<(OpId, BlockId)> {
        let mut out = Vec::new();
        let mut cur = block;
        while let Some(op) = self.parent_op(cur) {
            cur = self.parent_block(op);
            out.push((op, cur));
        }
        out
    }

    /// Every op under `block` in program order, nested blocks included,
    /// each op before the ops of its blocks.
    pub fn walk(&self, block: BlockId, f: &mut impl FnMut(OpId)) {
        for &op in self.ops(block) {
            f(op);
            for &inner in self.blocks_of(op) {
                self.walk(inner, f);
            }
        }
    }

    /// [`Ir::walk`] over the whole kernel, collected.
    pub fn all_ops(&self) -> Vec<OpId> {
        let mut out = Vec::new();
        self.walk(self.entry, &mut |op| out.push(op));
        out
    }

    // Mutation. Every method here leaves operands and use lists agreeing.

    /// A block with the given argument types, attached to no op yet. It has
    /// to be handed to [`Ir::create_op`] before the graph verifies.
    pub fn new_block(&mut self, arg_types: &[Type]) -> BlockId {
        let block = BlockId::from_index(self.blocks.len());
        let args = arg_types
            .iter()
            .enumerate()
            .map(|(index, ty)| self.new_value(ty.clone(), Def::Arg { block, index }))
            .collect();
        self.blocks.push(Some(BlockData {
            args,
            ops: Vec::new(),
            parent: None,
        }));
        block
    }

    fn new_value(&mut self, ty: Type, def: Def) -> ValueId {
        let value = ValueId::from_index(self.values.len());
        self.values.push(Some(ValueData {
            ty,
            def,
            uses: Vec::new(),
            name: None,
        }));
        value
    }

    /// Creates an op at `at`, with one result per entry of `result_types`
    /// and owning `blocks`, each of which must be unattached.
    pub fn create_op(
        &mut self,
        at: At,
        kind: OpKind,
        operands: &[ValueId],
        result_types: Vec<Type>,
        blocks: Vec<BlockId>,
    ) -> OpId {
        let op = OpId::from_index(self.ops.len());
        let (parent, index) = self.resolve(at);
        for &v in operands {
            self.value_data(v);
        }
        for &b in &blocks {
            assert!(
                b != self.entry && self.block_data(b).parent.is_none(),
                "{b} already belongs to an op"
            );
        }
        let results = result_types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| self.new_value(ty, Def::Result { op, index }))
            .collect();
        self.ops.push(Some(OpData {
            kind,
            operands: operands.to_vec(),
            results,
            blocks: blocks.clone(),
            parent,
        }));
        for (index, &v) in operands.iter().enumerate() {
            self.value_data_mut(v).uses.push(Use { op, index });
        }
        for b in blocks {
            self.block_data_mut(b).parent = Some(op);
        }
        self.block_data_mut(parent).ops.insert(index, op);
        op
    }

    /// The block and index an insertion point names, at the time of the call.
    fn resolve(&self, at: At) -> (BlockId, usize) {
        match at {
            At::End(block) => (block, self.ops(block).len()),
            At::Start(block) => (block, 0),
            At::Before(op) => (self.parent_block(op), self.position(op)),
            At::After(op) => (self.parent_block(op), self.position(op) + 1),
            At::Index(block, index) => {
                assert!(index <= self.ops(block).len(), "index past the end of {block}");
                (block, index)
            }
        }
    }

    pub fn set_operand(&mut self, op: OpId, index: usize, value: ValueId) {
        let old = self.op_data(op).operands[index];
        if old == value {
            return;
        }
        self.value_data(value);
        self.value_data_mut(old)
            .uses
            .retain(|u| !(u.op == op && u.index == index));
        self.op_data_mut(op).operands[index] = value;
        self.value_data_mut(value).uses.push(Use { op, index });
    }

    /// Every use of `old` now reads `new`, which must have the same type.
    /// `old` keeps its definition and is left with no uses.
    pub fn replace_all_uses(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        assert!(
            self.ty(old) == self.ty(new),
            "replacing {old}: {} with {new}: {}",
            self.ty(old),
            self.ty(new)
        );
        let uses = std::mem::take(&mut self.value_data_mut(old).uses);
        for u in &uses {
            self.op_data_mut(u.op).operands[u.index] = new;
        }
        self.value_data_mut(new).uses.extend(uses);
    }

    /// Removes an op, its results and everything in its blocks. Refuses
    /// while any result is still used, rather than leave a dangling
    /// operand behind.
    pub fn erase_op(&mut self, op: OpId) {
        for &r in self.results(op) {
            let uses = self.uses(r);
            assert!(
                uses.is_empty(),
                "erasing {op} while {r} is used by {}",
                uses.iter().map(|u| u.op.to_string()).collect::<Vec<_>>().join(", ")
            );
        }
        for b in self.blocks_of(op).to_vec() {
            for inner in self.ops(b).to_vec().into_iter().rev() {
                self.erase_op(inner);
            }
            for a in self.args(b).to_vec() {
                let uses = self.uses(a);
                assert!(uses.is_empty(), "erasing {op} while its {a} is still used");
                self.values[a.index()] = None;
            }
            self.blocks[b.index()] = None;
        }
        for (index, v) in self.operands(op).to_vec().into_iter().enumerate() {
            self.value_data_mut(v)
                .uses
                .retain(|u| !(u.op == op && u.index == index));
        }
        let parent = self.parent_block(op);
        self.block_data_mut(parent).ops.retain(|&o| o != op);
        for r in self.results(op).to_vec() {
            self.values[r.index()] = None;
        }
        self.ops[op.index()] = None;
    }

    /// Detaches an op from its block and inserts it at `at`. Dominance is
    /// the caller's to keep, and the verifier's to check.
    pub fn move_op(&mut self, op: OpId, at: At) {
        let from = self.parent_block(op);
        self.block_data_mut(from).ops.retain(|&o| o != op);
        let (to, index) = self.resolve(at);
        self.block_data_mut(to).ops.insert(index, op);
        self.op_data_mut(op).parent = to;
    }

    /// Whether `op` textually contains `inner`: `inner` sits in one of its
    /// blocks, at any depth.
    pub fn contains(&self, op: OpId, inner: OpId) -> bool {
        self.ancestors(self.parent_block(inner))
            .iter()
            .any(|&(a, _)| a == op)
    }

    /// Whether a value defined at `def` is visible from an operand of `at`:
    /// the definition's block is the use's block or an ancestor of it, and
    /// the definition precedes the use at that level. An op's own results
    /// are not visible inside its blocks.
    pub fn dominates(&self, value: ValueId, at: OpId) -> bool {
        let (def_block, def_index) = match self.def(value) {
            Def::Arg { block, .. } => (block, None),
            Def::Result { op, .. } => (self.parent_block(op), Some(self.position(op))),
        };
        let mut level_op = at;
        let mut level_block = self.parent_block(at);
        loop {
            if level_block == def_block {
                return match def_index {
                    None => true,
                    Some(d) => d < self.position(level_op),
                };
            }
            match self.parent_op(level_block) {
                Some(parent) => {
                    level_op = parent;
                    level_block = self.parent_block(parent);
                }
                None => return false,
            }
        }
    }
}
