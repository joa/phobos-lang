use std::fmt;

/// An element or scalar type. `Index` is the in-kernel integer: every
/// literal, loop variable and program id is one, and integer params and
/// tensor elements convert to it on the way in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    F16,
    BF16,
    F32,
    F64,
    I8,
    I32,
    I64,
    /// Unsigned storage: loaded, stored and converted, zero-extending, but
    /// computed with only after a conversion.
    U8,
    U16,
    U32,
    Bool,
    Index,
}

impl Scalar {
    pub fn from_ast(scalar: crate::ast::Scalar) -> Scalar {
        use crate::ast::Scalar as A;
        match scalar {
            A::F16 => Scalar::F16,
            A::BF16 => Scalar::BF16,
            A::F32 => Scalar::F32,
            A::F64 => Scalar::F64,
            A::I8 => Scalar::I8,
            A::I32 => Scalar::I32,
            A::I64 => Scalar::I64,
            A::U8 => Scalar::U8,
            A::U16 => Scalar::U16,
            A::U32 => Scalar::U32,
            A::Bool => Scalar::Bool,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Scalar::F16 => "f16",
            Scalar::BF16 => "bf16",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::I8 => "i8",
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::U8 => "u8",
            Scalar::U16 => "u16",
            Scalar::U32 => "u32",
            Scalar::Bool => "bool",
            Scalar::Index => "index",
        }
    }

    /// The name MLIR prints for the type: `i1` for a bool, the rest as is.
    pub fn mlir_name(self) -> &'static str {
        match self {
            Scalar::Bool => "i1",
            other => other.name(),
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::BF16 | Scalar::F32 | Scalar::F64)
    }

    /// The sized integers: what a tensor or tile element can be. `Index` and
    /// `Bool` are integers to arithmetic but never elements.
    pub fn is_int(self) -> bool {
        matches!(self, Scalar::I8 | Scalar::I32 | Scalar::I64)
    }

    /// The unsigned storage types.
    pub fn is_unsigned(self) -> bool {
        matches!(self, Scalar::U8 | Scalar::U16 | Scalar::U32)
    }

    pub fn is_numeric(self) -> bool {
        self.is_float() || self.is_int() || self == Scalar::Index
    }

    /// Width of an element in bytes; None for the unsized `Bool` and `Index`.
    pub fn bytes(self) -> Option<i64> {
        Some(match self {
            Scalar::I8 | Scalar::U8 => 1,
            Scalar::F16 | Scalar::BF16 | Scalar::U16 => 2,
            Scalar::F32 | Scalar::I32 | Scalar::U32 => 4,
            Scalar::F64 | Scalar::I64 => 8,
            Scalar::Bool | Scalar::Index => return None,
        })
    }

    pub fn float_bits(self) -> Option<u32> {
        match self {
            Scalar::F16 | Scalar::BF16 => Some(16),
            Scalar::F32 => Some(32),
            Scalar::F64 => Some(64),
            _ => None,
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One dimension of a shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extent {
    Fixed(i64),
    Dyn,
}

impl Extent {
    pub fn fixed(self) -> Option<i64> {
        match self {
            Extent::Fixed(n) => Some(n),
            Extent::Dyn => None,
        }
    }

    pub fn is_dyn(self) -> bool {
        self == Extent::Dyn
    }
}

impl fmt::Display for Extent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Extent::Fixed(n) => write!(f, "{n}"),
            Extent::Dyn => f.write_str("?"),
        }
    }
}

fn fmt_shape(f: &mut fmt::Formatter<'_>, shape: &[Extent]) -> fmt::Result {
    f.write_str("[")?;
    for (i, e) in shape.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{e}")?;
    }
    f.write_str("]")
}

/// Where a tile's bytes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Space {
    /// The CTA's shared memory: a tile buffer, or a view of one.
    Shared,
    /// Global memory: a slice of a tensor parameter.
    Global,
    /// One thread's own storage: the slot a `var` scalar lives in.
    Private,
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Space::Shared => "shared",
            Space::Global => "global",
            Space::Private => "private",
        })
    }
}

/// XOR column swizzle of a staging buffer:
/// `col' = col ^ (((row >> shift) & ((1 << bits) - 1)) << elem_log)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Swizzle {
    pub bits: u32,
    pub shift: u32,
    pub elem_log: u32,
}

/// How a tile's elements sit in its bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    /// None for a contiguous tile; otherwise the padded row pitch in
    /// elements, wider than the logical row.
    pub row_stride: Option<i64>,
    pub swizzle: Option<Swizzle>,
    /// Proven divisor, in elements, of the base offset and of every stride
    /// above the innermost. Zero means no offset at all.
    pub align_div: i64,
}

impl Layout {
    pub const CONTIGUOUS: Layout = Layout {
        row_stride: None,
        swizzle: None,
        align_div: 0,
    };

    pub fn is_contiguous(&self) -> bool {
        self.row_stride.is_none() && self.swizzle.is_none()
    }
}

/// A tensor parameter: global memory, identity layout, sliceable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub elem: Scalar,
    pub shape: Vec<Extent>,
    /// Known divisor of each dynamic extent, 1 when nothing was promised.
    /// `@aligned` is the only thing that raises it.
    pub div: Vec<i64>,
}

/// A tile: a shared-memory buffer, a view of one, or a slice of a tensor.
/// Layout, alignment and where the bytes live travel with the value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileType {
    pub elem: Scalar,
    pub shape: Vec<Extent>,
    pub layout: Layout,
    pub space: Space,
}

impl TileType {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|e| !e.is_dyn())
    }

    /// The static shape, when every extent is fixed.
    pub fn static_shape(&self) -> Option<Vec<i64>> {
        self.shape.iter().map(|e| e.fixed()).collect()
    }

    /// Elements in the buffer backing this tile, padding included.
    pub fn physical_elems(&self) -> Option<i64> {
        let shape = self.static_shape()?;
        let (last, lead) = shape.split_last()?;
        let last = self.layout.row_stride.unwrap_or(*last);
        Some(lead.iter().product::<i64>() * last)
    }

    pub fn bytes(&self) -> Option<i64> {
        Some(self.physical_elems()? * self.elem.bytes()?)
    }
}

/// How a register-resident tile's elements are spread over the lanes that
/// hold them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegLayout {
    /// The D fragments of `mma.sync` across a `wm x wn` warp grid, in the
    /// 8x8 tiles of the integer tensor cores with `imma`, else the 16x16
    /// ones of the f16 path.
    MmaAcc { wm: i64, wn: i64, imma: bool },
    /// Elements `t`, `t + threads`, ... of the row-major order on thread
    /// `t`, the mapping a plain sweep uses, so a sweep reads it without
    /// moving data between threads.
    Blocked,
    /// A register matmul's accumulator, summed over `k` a step and spread
    /// by whichever path the target runs it on.
    Gemm { k: i64 },
    /// Nowhere: each read computes the elements it needs from a
    /// `Generate` op's block.
    Generated,
}

/// A tile held in registers rather than in a buffer: a value, carried and
/// updated whole, never sliced.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DistType {
    pub elem: Scalar,
    pub shape: Vec<i64>,
    pub layout: RegLayout,
}

impl DistType {
    /// The tile as `mma.sync` accumulator fragments.
    pub fn mma_acc(&self) -> Option<MmaAcc> {
        let (RegLayout::MmaAcc { wm, wn, imma }, &[m, n]) = (self.layout, &self.shape[..]) else {
            return None;
        };
        Some(MmaAcc { m, n, wm, wn, imma })
    }

    /// The tile as a register matmul's accumulator.
    pub fn gemm(&self) -> Option<GemmType> {
        let (RegLayout::Gemm { k }, &[m, n]) = (self.layout, &self.shape[..]) else {
            return None;
        };
        Some(GemmType { acc: self.elem, m, n, k })
    }
}

/// An `[m, n]` f32 accumulator in `mma.sync` fragments across a `wm x wn`
/// warp grid: what an `MmaAcc` [`DistType`] holds, unpacked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MmaAcc {
    pub m: i64,
    pub n: i64,
    pub wm: i64,
    pub wn: i64,
    /// Held in the 8x8 tiles of the integer tensor cores' `m8n8k16` rather
    /// than the 16x16 ones of the f16 path.
    pub imma: bool,
}

impl MmaAcc {
    /// The side of one fragment tile.
    pub fn tile(&self) -> i64 {
        if self.imma { 8 } else { 16 }
    }

    /// The register-resident tile these fragments hold.
    pub fn dist(&self) -> DistType {
        DistType {
            elem: Scalar::F32,
            shape: vec![self.m, self.n],
            layout: RegLayout::MmaAcc {
                wm: self.wm,
                wn: self.wn,
                imma: self.imma,
            },
        }
    }
}

/// A register matmul's accumulator: an [m, n] tile of `acc` summed over a
/// k extent of `k` per step, held in registers by whichever path the
/// target runs it on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmType {
    pub acc: Scalar,
    pub m: i64,
    pub n: i64,
    pub k: i64,
}

impl GemmType {
    /// The register-resident tile this accumulator is.
    pub fn dist(&self) -> DistType {
        DistType {
            elem: self.acc,
            shape: vec![self.m, self.n],
            layout: RegLayout::Gemm { k: self.k },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(Scalar),
    Tensor(TensorType),
    Tile(TileType),
    Dist(DistType),
}

impl Type {
    pub const INDEX: Type = Type::Scalar(Scalar::Index);
    pub const BOOL: Type = Type::Scalar(Scalar::Bool);

    pub fn scalar(self) -> Option<Scalar> {
        match self {
            Type::Scalar(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_scalar(&self) -> Option<Scalar> {
        match self {
            Type::Scalar(s) => Some(*s),
            _ => None,
        }
    }

    pub fn as_tile(&self) -> Option<&TileType> {
        match self {
            Type::Tile(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_tensor(&self) -> Option<&TensorType> {
        match self {
            Type::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// The element type of anything addressable.
    pub fn elem(&self) -> Option<Scalar> {
        match self {
            Type::Scalar(_) | Type::Dist(_) => None,
            Type::Tensor(t) => Some(t.elem),
            Type::Tile(t) => Some(t.elem),
        }
    }

    /// The shape of anything addressable.
    pub fn shape(&self) -> Option<&[Extent]> {
        match self {
            Type::Scalar(_) | Type::Dist(_) => None,
            Type::Tensor(t) => Some(&t.shape),
            Type::Tile(t) => Some(&t.shape),
        }
    }

    /// A `Blocked` register tile, as a sweep holds it.
    pub fn regs(&self) -> Option<&DistType> {
        match self {
            Type::Dist(d) if d.layout == RegLayout::Blocked => Some(d),
            _ => None,
        }
    }

    /// A register matmul's accumulator.
    pub fn gemm(&self) -> Option<GemmType> {
        match self {
            Type::Dist(d) => d.gemm(),
            _ => None,
        }
    }

    /// Whether this is an accumulator, fragments or a register matmul's,
    /// which only its own ops may read.
    pub fn is_accumulator(&self) -> bool {
        matches!(self, Type::Dist(d) if matches!(d.layout, RegLayout::MmaAcc { .. } | RegLayout::Gemm { .. }))
    }

    /// A contiguous shared tile of a static shape, the common allocation.
    pub fn shared_tile(elem: Scalar, shape: &[i64]) -> Type {
        Type::Tile(TileType {
            elem,
            shape: shape.iter().map(|&n| Extent::Fixed(n)).collect(),
            layout: Layout::CONTIGUOUS,
            space: Space::Shared,
        })
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Scalar(s) => write!(f, "{s}"),
            Type::Tensor(t) => {
                write!(f, "tensor<{}>", t.elem)?;
                fmt_shape(f, &t.shape)?;
                if t.div.iter().any(|&d| d != 1) {
                    write!(f, "@div{:?}", t.div)?;
                }
                Ok(())
            }
            Type::Tile(t) => {
                write!(f, "tile<{}>", t.elem)?;
                fmt_shape(f, &t.shape)?;
                write!(f, "@{}", t.space)?;
                let mut extras = Vec::new();
                if let Some(s) = t.layout.row_stride {
                    extras.push(format!("stride {s}"));
                }
                if let Some(sw) = t.layout.swizzle {
                    extras.push(format!("swizzle {}/{}/{}", sw.bits, sw.shift, sw.elem_log));
                }
                if t.layout.align_div != 0 {
                    extras.push(format!("align {}", t.layout.align_div));
                }
                if !extras.is_empty() {
                    write!(f, "{{{}}}", extras.join(", "))?;
                }
                Ok(())
            }
            Type::Dist(d) => {
                let dims: Vec<String> = d.shape.iter().map(i64::to_string).collect();
                match d.layout {
                    RegLayout::MmaAcc { wm, wn, imma } => {
                        write!(f, "frags[{}]/{wm}x{wn}", dims.join(", "))?;
                        if imma {
                            write!(f, "/i8")?;
                        }
                    }
                    RegLayout::Blocked => write!(f, "regs<{}>[{}]", d.elem, dims.join(", "))?,
                    RegLayout::Gemm { k } => write!(f, "gemm<{}>[{}]/{k}", d.elem, dims.join(", "))?,
                    RegLayout::Generated => write!(f, "gen<{}>[{}]", d.elem, dims.join(", "))?,
                }
                Ok(())
            }
        }
    }
}
