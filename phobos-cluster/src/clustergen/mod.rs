// Kernel to cluster program.
//
// [`Analyzer`] holds the state and the rest of this module is its work,
// split by phase: walk, classify, finalize, then lower.

mod analyze;
mod classify;
mod epilogue;
mod finalize;
mod lower;

use lower::*;

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};
use phobos_lang::ast::{
    AssignOp, AttrArg, BinOp, Dim, Expr, Kernel, Scalar, Stmt, Sub, Type as AstType,
};

use crate::ir::{
    ClusterProgram, ClusterStmt, Coord, GridAxis, LeafKernel, ScalarDecl, SearchDim, SuperTile,
    TensorDecl,
};
use crate::tile::{AccessMode, DataType};

/// Compile a @cluster kernel to the parametric cluster IR.
pub fn compile(kernel: &Kernel) -> Result<ClusterProgram> {
    Analyzer::new(kernel)?.run()
}

fn data_type(s: Scalar) -> DataType {
    match s {
        Scalar::F16 => DataType::F16,
        Scalar::BF16 => DataType::BF16,
        Scalar::F32 => DataType::F32,
        Scalar::F64 => DataType::F64,
        Scalar::I8 => DataType::I8,
        Scalar::I32 => DataType::I32,
        Scalar::I64 => DataType::I64,
        Scalar::Bool => DataType::Bool,
    }
}

/// Cluster scale scalar value.
#[derive(Clone, Debug)]
enum ScalarValue {
    /// program_id(i)
    Pid(usize),
    /// program_id(i) * SUPER
    PidSuper(usize, String),
}

/// What a name binds to at cluster scale.
#[derive(Clone, Debug)]
enum Binding {
    Scalar(ScalarValue),
    /// Supertile view (let a = A[..]).
    Ref(SuperTile),
    /// The accumulator tile (var acc: tile<..>[..] = 0.0).
    Scratch,
    /// Cluster-loop iv; payload is the step's super sym.
    LoopVar(String),
    /// A leaf-internal (device-scale) loop iv -> used only by the single-leaf path.
    /// Payload is the dim the loop uses; a slice offset by it covers
    /// the whole axis, so the cluster never tiles that axis (see [`Coord::Full`]).
    DeviceLoop(String),
}

enum Statement {
    /// Placeholder at the scratch declaration; becomes the init compute.
    InitPlaceholder,
    Compute(Pending),
    Loop {
        var: String,
        dim: Dim,
        super_sym: String,
        body: Vec<Statement>,
    },
}

struct Pending {
    /// Reads collected from the value expression.
    ///
    /// Key is tensor index.
    reads: Vec<(usize, SuperTile)>,
    /// Direct-compute write target; None when the target is the scratch.
    target: Option<(usize, SuperTile, AccessMode)>,
    uses_scratch: bool,
}

struct Scratch {
    init: Expr,
}

/// The chain's C[<grid slice>] = <epilogue> store.
struct Define {
    tensor: usize,
    coords: Vec<Coord>,
    /// Super sym per axis (for the synthesized init leaf's slice).
    supers: Vec<String>,
    /// Top-level statement index, for the step leaf's store rewrite.
    stmt_idx: usize,
    /// The GEMM epilogue when the store is alpha*acc + beta*c_old rather than
    /// a bare = acc. None keeps the plain chain (zero-init, += acc step).
    epilogue: Option<Epilogue>,
}

/// A GEMM-shaped accumulator epilogue: C[..] = [alpha *] acc [+ [beta *] c_old]
/// where c_old is a prior load of the same output supertile.
///
/// Copy-elision keeps the accumulator on C's buffer, so the epilogue folds:
/// the step leaf accumulates alpha*acc (stored as alpha*acc + c_old, which
/// stays on the fused register-accumulator path with an implicit beta of 1),
/// and the init leaf seeds C with beta*c_old instead of zero-filling. The
/// running sum then lands on beta*C_orig + alpha*sum(dot) = alpha*acc + beta*C_orig.
struct Epilogue {
    /// The accumulator scratch var name.
    acc: String,
    /// Coefficient on acc (None is identity 1.0), applied per step.
    alpha: Option<Expr>,
    /// The prior-C term: Some((beta, c_old)) when the epilogue reads C back.
    /// beta is the coefficient (None is identity 1.0); c_old names the load.
    prev: Option<(Option<Expr>, String)>,
}

/// How to seed the accumulator output before the chain runs.
struct InitInfo {
    /// No init compute at all (beta is identity, so C keeps its original value).
    skip: bool,
    /// C's access mode in the init compute (Write to zero-fill, RMW to scale C).
    c_mode: AccessMode,
    /// Scalar indices the init compute carries (the beta coefficient's params).
    scalars: Vec<usize>,
}

struct Analyzer<'a> {
    kernel: &'a Kernel,
    super_dims: Vec<SearchDim>,
    super_set: HashSet<String>,
    tensors: Vec<TensorDecl>,
    tindex: HashMap<String, usize>,
    scalars: Vec<ScalarDecl>,
    symbols: HashMap<String, Binding>,
    /// Per-tensor, per-axis supertile sym discovered from slice extents.
    tensor_syms: Vec<Vec<Option<String>>>,
    /// Grid axes discovered from slice uses, indexed by pid.
    grid: Vec<Option<GridAxis>>,
    scratch: Option<Scratch>,
    define: Option<Define>,
    n_computes: usize,
}

#[cfg(test)]
mod tests;
