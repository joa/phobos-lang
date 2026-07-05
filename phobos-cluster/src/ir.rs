use phobos_lang::ast::{Dim, Kernel};

use crate::tile::{AccessMode, DataType};

/// @cluster search dimension
#[derive(Clone, Debug)]
pub struct SearchDim {
    pub name: String,
    pub choices: Vec<i64>,
}

/// Distributed tensor
#[derive(Clone, Debug)]
pub struct TensorDecl {
    pub name: String,
    pub data_type: DataType,
    pub dims: Vec<Dim>,
    pub super_syms: Vec<String>,
    /// Read = LOAD
    /// Write = pure output (no initial read; zero-init)
    /// RMW = LOAD and STORE
    pub mode: AccessMode,
}

#[derive(Clone, Debug)]
pub struct ScalarDecl {
    pub name: String,
    pub data_type: DataType,
    pub param_pos: usize,
}

#[derive(Clone, Debug)]
pub struct GridAxis {
    pub pid: usize,
    pub dim: Dim,
    pub super_sym: String,
}

#[derive(Clone, Debug)]
pub struct LeafKernel {
    pub kernel: Kernel,
    /// Parameter access modes (in order)
    pub modes: Vec<AccessMode>,
}

/// Cluster-loop iteration index.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Coord {
    /// program_id(pid) index into [`ClusterProgram::grid`]
    Grid(usize),
    /// Iteration index of the named cluster loop (0-based).
    Loop(String),
    // Not tiled -> [:]
    Full,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SuperTile {
    pub tensor: usize,
    pub coords: Vec<Coord>,
}

#[derive(Clone, Debug)]
pub enum ClusterStmt {
    Compute {
        leaf: usize,
        args: Vec<(SuperTile, AccessMode)>,
        scalars: Vec<usize>,
    },
    /// Cluster-level loop with dim / super_sym iterations.
    Loop {
        var: String,
        dim: Dim,
        super_sym: String,
        body: Vec<ClusterStmt>,
    },
}

#[derive(Clone, Debug)]
pub struct ClusterProgram {
    pub name: String,
    pub super_dims: Vec<SearchDim>,
    pub tensors: Vec<TensorDecl>,
    pub scalars: Vec<ScalarDecl>,
    pub leaves: Vec<LeafKernel>,
    pub grid: Vec<GridAxis>,
    pub body: Vec<ClusterStmt>,
}
