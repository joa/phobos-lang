// The trace format, version 1, as `website/src/lib/viz/trace.ts` declares it.
// Field names follow that file; every optional field there is an `Option` or
// an empty `Vec` here, left out of the JSON when absent.

use std::collections::BTreeMap;

use serde::Serialize;

pub type Cta = [i64; 3];

/// A half-open range per dimension.
pub type Region = Vec<[i64; 2]>;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Trace {
    pub schema: u32,
    pub name: String,
    pub kernel: String,
    pub source: Source,
    pub compiler: Compiler,
    pub params: BTreeMap<String, i64>,
    pub dims: BTreeMap<String, i64>,
    pub launch: Launch,
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
    pub buffers: Vec<Buffer>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub barriers: Vec<Barrier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared_peak_bytes: Option<i64>,
    pub ctas: Vec<Cta>,
    pub steps: Vec<Step>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub summaries: Vec<Summary>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    pub origin: Origin,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fence: Option<String>,
    pub text: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hidden_lines: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<Variant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Origin {
    File,
    Fence,
}

/// An attribute added to or dropped from a kernel's source before it is
/// compiled. Serialized with `add` before `drop`, which the source hash
/// depends on.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Variant {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub add: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Compiler {
    pub commit: String,
    pub chip: String,
    pub index_bits: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Launch {
    pub grid: Cta,
    pub threads: i64,
    pub lanes: i64,
    pub warps: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    In,
    Out,
    Inout,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Tensor {
    pub name: String,
    pub elem: String,
    pub elem_bytes: i64,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub role: Role,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Span {
    pub line: u32,
    pub col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Uniformity {
    Uniform,
    Warp,
    Varying,
}

#[derive(Clone, Debug, Serialize)]
pub struct Op {
    pub id: usize,
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
    pub ir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uniformity: Option<Uniformity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instr: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    #[serde(rename = "loop", skip_serializing_if = "Option::is_none")]
    pub loop_info: Option<LoopInfo>,
}

/// A `for` or `while`: its induction variable and how often it ran.
#[derive(Clone, Debug, Serialize)]
pub struct LoopInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub var: Option<String>,
    pub trips: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Buffer {
    pub id: usize,
    pub name: String,
    pub bytes: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_bytes: Option<i64>,
    pub first_op: usize,
    pub last_op: usize,
    pub scratch: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elem: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<i64>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Barrier {
    pub after_op: usize,
    pub kept: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Access {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buffer: Option<usize>,
    #[serde(rename = "box")]
    pub region: Region,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<[i64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub masked_elems: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMap {
    pub shape: Vec<i64>,
    pub width: i64,
    pub threads: i64,
    pub vector_bytes: i64,
    #[serde(rename = "async", skip_serializing_if = "Option::is_none")]
    pub async_copy: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneKind {
    Shuffle,
    Collective,
    Exchange,
    LaneLoad,
    LaneStore,
    Dp4a,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Xor,
    Down,
    Up,
    From,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Run {
    pub lane: usize,
    pub access: Access,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneEvent {
    pub kind: LaneKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<Mode>,
    pub width: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partners: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stages: Option<Vec<Vec<i64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runs: Option<Vec<Run>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_bytes: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub align_bytes: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub barriers: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct StepBarrier {
    pub kept: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Step {
    pub op: usize,
    pub cta: Cta,
    pub iter: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<Access>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<Access>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub map: Option<ThreadMap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lanes: Option<LaneEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub barrier: Option<StepBarrier>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub cta: Cta,
    pub op: usize,
    pub iter: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<Access>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<Access>,
}
