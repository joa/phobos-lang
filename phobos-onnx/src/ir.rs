use std::collections::HashMap;
use std::sync::Arc;

/// The subset of ONNX `TensorProto.DataType` this crate models. Anything else
/// lands in [`DataType::Other`], carrying the raw code so it can be reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    F16,
    F32,
    F64,
    I8,
    U8,
    I32,
    I64,
    Bool,
    Other(i32),
}

/// One axis of a shape: a fixed extent, a symbol shared across the graph
/// (`batch`, `seq`), or unknown.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Fixed(i64),
    Symbol(String),
    Unknown,
}

/// A tensor shape. `None` means undeclared; `Some(dims)` gives the rank, whose
/// per-axis extents may still be unknown.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Shape(pub Option<Vec<Dim>>);

impl Shape {
    /// Every extent known.
    pub fn is_static(&self) -> bool {
        matches!(&self.0, Some(dims) if dims.iter().all(|d| matches!(d, Dim::Fixed(_))))
    }

    /// Element count when the shape is fully static.
    pub fn numel(&self) -> Option<i64> {
        let dims = self.0.as_ref()?;
        dims.iter()
            .map(|d| match d {
                Dim::Fixed(n) => Some(*n),
                _ => None,
            })
            .product()
    }
}

/// Name, element type and shape of a graph edge.
#[derive(Clone, Debug)]
pub struct ValueInfo {
    pub name: String,
    pub data_type: Option<DataType>,
    pub shape: Shape,
}

/// Materialized data for an initializer.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F16(Vec<u16>), // raw IEEE binary16 bit patterns
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    /// Undecoded payload for data types that are not materialized yet.
    Raw(Vec<u8>),
}

/// A constant tensor, an ONNX initializer.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data_type: DataType,
    pub dims: Vec<i64>,
    pub data: TensorData,
}

/// Only the attribute kinds transformer graphs use; the rest are dropped at
/// load.
#[derive(Clone, Debug)]
pub enum Attribute {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensor(Tensor),
}

/// One ONNX operator invocation.
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,
    /// Raw ONNX `op_type`, e.g. "MatMul" or "LayerNormalization".
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attribute>,
}

/// A whole graph, its topology implied by node order.
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub name: String,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
    pub nodes: Vec<Node>,
    /// Constant tensors by edge name. Reference-counted so the passes that
    /// rebuild the graph share weight payloads instead of deep-copying them.
    pub initializers: HashMap<String, Arc<Tensor>>,
    /// Declared type and shape for named edges.
    pub values: HashMap<String, ValueInfo>,
}

impl Graph {
    pub fn is_constant(&self, name: &str) -> bool {
        self.initializers.contains_key(name)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub graph: Graph,
    pub ir_version: i64,
    pub producer_name: String,
    /// domain -> opset version.
    pub opset: HashMap<String, i64>,
}
