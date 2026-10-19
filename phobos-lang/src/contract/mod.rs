// The launch contract: what a host has to supply, beyond the PTX, to launch
// one compiled kernel correctly. It is read off the graph the emitter
// lowered and the shared memory the emitter placed, so it describes the
// `.entry` the PTX declares rather than what the source suggests.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ast::{Dim, Kernel, Type as AstType};
use crate::ir::{Ir, OpKind, Scalar, Type};

/// The contract of one kernel of a compiled module.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub kernel: String,
    pub chip: String,
    /// The width of every offset, size and stride slot.
    pub index_bits: u32,
    /// The CTA size the kernel was compiled for, from `@launch` or the
    /// default. The block a launch uses must be this.
    pub cta_threads: i64,
    /// The shared memory the module declares for the kernel.
    pub static_shared_bytes: i64,
    /// The shared memory a launch must pass, nonzero only under
    /// `@dynshared`.
    pub dynamic_shared_bytes: i64,
    /// Set for a `@persistent` kernel, whose grid must be co-resident.
    pub persistent: Option<Persistent>,
    /// The kernel's parameters as the launch binds them: the declared ones,
    /// then published scratch, then the barrier word.
    pub params: Vec<Param>,
    /// The `.entry` parameters, in order.
    pub slots: Vec<Slot>,
    /// Every symbolic extent read at run time, with where the kernel reads
    /// it from and what it must be a multiple of.
    pub dims: Vec<SymDim>,
    /// The `@autotune` constants this variant was compiled at.
    pub autotune: BTreeMap<String, i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Persistent {
    /// The grid's block count `@persistent(n)` compiled in; otherwise the
    /// launch sizes the grid from the occupancy it reaches.
    pub blocks: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A parameter the source declares.
    Declared,
    /// A scratch tensor a `@persistent` kernel publishes to device memory,
    /// uninitialized, alive for the launch.
    Scratch,
    /// The `i32` `[2, 1]` word `grid_barrier()` waits on, zeroed once
    /// before the first launch and never shared by concurrent kernels.
    BarrierWord,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub role: Role,
    /// The element type, or the scalar's type.
    pub elem: String,
    /// A tensor's extents, each a constant or the symbol a launch binds;
    /// `None` for a scalar.
    pub shape: Option<Vec<Extent>>,
    /// The alignment the kernel assumes of a tensor's base address.
    pub align_bytes: i64,
    /// The first of this parameter's slots.
    pub slot: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Extent {
    Fixed(i64),
    Sym(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymDim {
    pub name: String,
    /// The parameter the kernel reads the extent from: a scalar of that
    /// name, or the first tensor that has it as an axis.
    pub param: usize,
    /// That tensor's axis; `None` for a scalar.
    pub axis: Option<usize>,
    /// The 16-byte row-pitch ABI's 4, raised by what `@aligned` promises,
    /// which the kernel traps on at entry.
    pub multiple_of: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Part {
    /// A memref descriptor's allocated pointer.
    Allocated,
    /// Its aligned pointer, the same address.
    Aligned,
    /// Its offset, always 0.
    Offset,
    Size(usize),
    /// A row-major stride; the kernel's memrefs have the identity layout,
    /// so it is passed but never read.
    Stride(usize),
    /// A scalar parameter's value.
    Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    Ptr,
    Int,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    pub param: usize,
    pub part: Part,
    pub class: Class,
    pub bytes: u32,
}

/// The base alignment an `AssumeAlign` promises, so rows can vectorize.
const ASSUMED_ALIGN_BYTES: i64 = 16;

/// The row-pitch ABI every dynamic extent is assumed to keep.
const EXTENT_MULTIPLE: i64 = 4;

impl Contract {
    /// The contract of `kernel` as `ir` lowers it, with the shared memory
    /// the emission placed.
    pub(crate) fn of(
        base: &phobos_base::context::Context,
        kernel: &Kernel,
        ir: &Ir,
        static_shared_bytes: i64,
        dynamic_shared_bytes: i64,
    ) -> anyhow::Result<Contract> {
        let phobos_base::context::GpuConfig::Nvidia(nv) = &base.gpu_config;
        let index_bytes = base.index_bitwidth / 8;
        let args = ir.args(ir.entry());
        let word = usize::from(ir.kernel.grid_barrier);
        let declared = kernel.params.len();
        let assumed: Vec<_> = ir
            .all_ops()
            .into_iter()
            .filter(|&op| matches!(ir.kind(op), OpKind::AssumeAlign))
            .map(|op| ir.operand(op, 0))
            .collect();

        let mut params = Vec::with_capacity(args.len());
        let mut slots = Vec::new();
        for (i, &arg) in args.iter().enumerate() {
            let role = if i < declared {
                Role::Declared
            } else if i + word == args.len() {
                Role::BarrierWord
            } else {
                Role::Scratch
            };
            let source = kernel.params.get(i);
            let name = match (source, role) {
                (Some(p), _) => p.name.clone(),
                (None, Role::BarrierWord) => "grid_barrier".to_string(),
                (None, _) => ir.name(arg).unwrap_or_default().to_string(),
            };
            let slot = slots.len();
            let param = match ir.ty(arg) {
                Type::Tensor(t) => {
                    let shape = match source.map(|p| &p.ty) {
                        Some(AstType::Tensor(_, dims)) => dims
                            .iter()
                            .zip(&t.shape)
                            .map(|(dim, extent)| match (extent.fixed(), dim) {
                                (Some(n), _) => Extent::Fixed(n),
                                (None, Dim::Sym(s)) => Extent::Sym(s.clone()),
                                (None, Dim::Int(n)) => Extent::Fixed(*n),
                            })
                            .collect(),
                        _ => t
                            .shape
                            .iter()
                            .map(|e| e.fixed().map(Extent::Fixed).ok_or_else(|| anyhow::anyhow!("scratch `{name}` of dynamic shape")))
                            .collect::<anyhow::Result<_>>()?,
                    };
                    let elem_bytes = t.elem.bytes().unwrap_or(1);
                    let align_bytes = if assumed.contains(&arg) { ASSUMED_ALIGN_BYTES } else { elem_bytes };
                    let rank = t.shape.len();
                    let int = |part| Slot { param: i, part, class: Class::Int, bytes: index_bytes };
                    let ptr = |part| Slot { param: i, part, class: Class::Ptr, bytes: 8 };
                    slots.extend([ptr(Part::Allocated), ptr(Part::Aligned), int(Part::Offset)]);
                    slots.extend((0..rank).map(|d| int(Part::Size(d))));
                    slots.extend((0..rank).map(|d| int(Part::Stride(d))));
                    Param { name, role, elem: t.elem.name().to_string(), shape: Some(shape), align_bytes, slot }
                }
                Type::Scalar(s) => {
                    let (class, bytes) = scalar_slot(*s)?;
                    slots.push(Slot { param: i, part: Part::Value, class, bytes });
                    Param { name, role, elem: s.name().to_string(), shape: None, align_bytes: i64::from(bytes), slot }
                }
                other => anyhow::bail!("kernel `{}`: a {other} parameter", kernel.name),
            };
            params.push(param);
        }

        Ok(Contract {
            kernel: kernel.name.clone(),
            chip: nv.chip().to_string(),
            index_bits: base.index_bitwidth,
            cta_threads: ir.kernel.cta_threads,
            static_shared_bytes,
            dynamic_shared_bytes,
            persistent: kernel.attr("persistent").map(|_| Persistent { blocks: ir.kernel.persistent_blocks }),
            dims: sym_dims(kernel, ir, &params),
            params,
            slots,
            autotune: ir.kernel.shape_env.clone(),
        })
    }

    /// The contract as one JSON object.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("a contract always serializes")
    }
}

/// How a scalar parameter is passed: its class and its width in bytes.
fn scalar_slot(s: Scalar) -> anyhow::Result<(Class, u32)> {
    let class = if s.is_float() { Class::Float } else { Class::Int };
    let bytes = match s {
        Scalar::Bool => 1,
        other => other.bytes().ok_or_else(|| anyhow::anyhow!("a {} parameter", other.name()))? as u32,
    };
    Ok((class, bytes))
}

/// The symbols the kernel binds from its parameters at entry, in the order
/// `ir::build::params` binds them: each from the first parameter that is a
/// scalar of its name or a tensor with it as an axis.
fn sym_dims(kernel: &Kernel, ir: &Ir, params: &[Param]) -> Vec<SymDim> {
    let args = ir.args(ir.entry());
    let mut out: Vec<SymDim> = Vec::new();
    let bound = |out: &[SymDim], name: &str| out.iter().any(|d| d.name == name);
    for (i, param) in kernel.params.iter().enumerate() {
        match &param.ty {
            AstType::Scalar(_) => {
                // A scalar shadows a later tensor's axis of its name. Only
                // one some tensor axis names counts as an extent.
                let names_an_axis = params.iter().any(|p| {
                    p.shape.as_ref().is_some_and(|s| s.iter().any(|e| *e == Extent::Sym(param.name.clone())))
                });
                if names_an_axis && !bound(&out, &param.name) {
                    out.push(SymDim { name: param.name.clone(), param: i, axis: None, multiple_of: 1 });
                }
            }
            AstType::Tensor(_, dims) => {
                let Type::Tensor(t) = ir.ty(args[i]) else { continue };
                for (axis, dim) in dims.iter().enumerate() {
                    if let Dim::Sym(name) = dim
                        && t.shape[axis].is_dyn()
                        && !bound(&out, name)
                    {
                        let promised = t.div[axis].max(1);
                        let multiple_of = EXTENT_MULTIPLE * promised / gcd(EXTENT_MULTIPLE, promised);
                        out.push(SymDim { name: name.clone(), param: i, axis: Some(axis), multiple_of });
                    }
                }
            }
            AstType::Tile(..) => {}
        }
    }
    out
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

mod entry;
pub use entry::check_entry;

#[cfg(test)]
mod tests;
