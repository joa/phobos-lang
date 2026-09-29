// Kernel to cluster program. [`Analyzer`] holds the state, and the submodules
// are its phases: walk, classify, finalize, then lower.

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

/// Compile an `@cluster` kernel to the parametric cluster IR.
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

/// A scalar value at cluster scale.
#[derive(Clone, Debug)]
enum ScalarValue {
    /// `program_id(i)`
    Pid(usize),
    /// `program_id(i) * SUPER`
    PidSuper(usize, String),
}

/// What a name binds to at cluster scale.
#[derive(Clone, Debug)]
enum Binding {
    Scalar(ScalarValue),
    /// Supertile view (`let a = A[..]`).
    Ref(SuperTile),
    /// The accumulator tile (`var acc: tile<..>[..] = 0.0`).
    Scratch,
    /// Cluster-loop induction variable, carrying the step's super sym.
    LoopVar(String),
    /// A device-scale loop inside a leaf, used only by the single-leaf path.
    /// It offsets a full-axis slice, so the cluster never tiles that axis
    /// (see [`Coord::Full`]).
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
    /// Reads collected from the value expression, keyed by tensor index.
    reads: Vec<(usize, SuperTile)>,
    /// Direct-compute write target. None when the target is the scratch.
    target: Option<(usize, SuperTile, AccessMode)>,
    uses_scratch: bool,
}

struct Scratch {
    init: Expr,
}

/// The chain's `C[<grid slice>] = <epilogue>` store.
struct Define {
    tensor: usize,
    coords: Vec<Coord>,
    /// Super sym per axis, for the synthesized init leaf's slice.
    supers: Vec<String>,
    /// Top-level statement index, for the step leaf's store rewrite.
    stmt_idx: usize,
    /// The GEMM epilogue, when the store is `alpha*acc + beta*c_old`.
    /// None means a plain `= acc` store: zero-init, then `+= acc` per step.
    epilogue: Option<Epilogue>,
}

/// A GEMM-shaped accumulator epilogue, `C[..] = [alpha *] acc [+ [beta *] c_old]`,
/// where `c_old` is a prior load of the same output supertile.
///
/// The accumulator lives in C's buffer. The step leaf applies alpha each
/// step, and the init leaf seeds C with `beta*c_old` instead of zeros.
struct Epilogue {
    /// The accumulator scratch var name.
    acc: String,
    /// Coefficient on acc, applied per step. None means 1.0.
    alpha: Option<Expr>,
    /// The prior-C term `(beta, c_old)`, when the epilogue reads C back.
    /// A None beta means 1.0. `c_old` names the load.
    prev: Option<(Option<Expr>, String)>,
}

/// How to seed the accumulator output before the chain runs.
struct InitInfo {
    /// No init compute: beta is 1.0, so C keeps its original value.
    skip: bool,
    /// C's access mode in the init compute: Write to zero-fill, RMW to scale C.
    c_mode: AccessMode,
    /// Scalar indices the init compute carries, the beta coefficient's params.
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
