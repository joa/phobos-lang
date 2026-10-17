// What each GEMM benchmark computes: storage, multiply and accumulate types,
// and the cuBLAS call that computes the same thing.

use crate::report::Precision;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Elem {
    F32,
    F16,
}

impl Elem {
    pub fn bytes(self) -> usize {
        match self {
            Elem::F32 => 4,
            Elem::F16 => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Elem::F32 => "f32",
            Elem::F16 => "f16",
        }
    }

    /// Unit roundoff.
    pub fn unit_roundoff(self) -> f64 {
        match self {
            Elem::F32 => f64::from(f32::EPSILON) / 2.0,
            Elem::F16 => 2f64.powi(-11),
        }
    }

    /// Rounds `x` to this type and back.
    pub fn round(self, x: f32) -> f32 {
        match self {
            Elem::F32 => x,
            Elem::F16 => half::f16::from_f32(x).to_f32(),
        }
    }
}

/// The cuBLAS call a kind is compared against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Baseline {
    /// cublasSgemm: f32 storage, f32 multiply and accumulate on CUDA cores.
    Sgemm,
    /// cublasGemmEx: f16 A and B, f32 C, CUBLAS_COMPUTE_32F.
    GemmExF16F32,
    /// cublasGemmEx: f16 A, B and C, CUBLAS_COMPUTE_16F.
    GemmExF16F16,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// f32 storage, f32 multiply, f32 accumulate.
    F32,
    /// [`Kind::F32`] without `@aligned`/`@pipeline`: any M, N, K.
    F32Unaligned,
    /// f32 storage, operands rounded to f16 while staging, tensor-core
    /// multiply, f32 accumulate.
    F32StoreF16Mul,
    /// f16 A and B, f32 C, tensor-core multiply, f32 accumulate.
    F16StoreF32Acc,
    /// f16 A, B and C, tensor-core multiply, f16 accumulate.
    F16Acc,
}

pub(crate) const KINDS: &[Kind] = &[
    Kind::F32,
    Kind::F32Unaligned,
    Kind::F32StoreF16Mul,
    Kind::F16StoreF32Acc,
    Kind::F16Acc,
];

impl Kind {
    /// The `--bench` name, and the CSV's benchmark column.
    pub fn name(self) -> &'static str {
        match self {
            Kind::F32 => "gemm_fp32",
            Kind::F32Unaligned => "gemm_fp32_unaligned",
            Kind::F32StoreF16Mul => "gemm_fp16tc_fp32acc",
            Kind::F16StoreF32Acc => "gemm_fp16_fp32acc",
            Kind::F16Acc => "gemm_fp16",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Kind::F32 => include_str!("../../../examples/gemm_fp32.ph"),
            Kind::F32Unaligned => include_str!("../../../examples/gemm_fp32_unaligned.ph"),
            Kind::F32StoreF16Mul => include_str!("../../../examples/gemm_fp16tc_fp32acc.ph"),
            Kind::F16StoreF32Acc => include_str!("../../../examples/gemm_fp16_fp32acc.ph"),
            Kind::F16Acc => include_str!("../../../examples/gemm_fp16.ph"),
        }
    }

    /// Storage of A and B.
    pub fn ab(self) -> Elem {
        match self {
            Kind::F32 | Kind::F32Unaligned | Kind::F32StoreF16Mul => Elem::F32,
            Kind::F16StoreF32Acc | Kind::F16Acc => Elem::F16,
        }
    }

    /// Storage of C.
    pub fn c(self) -> Elem {
        match self {
            Kind::F16Acc => Elem::F16,
            _ => Elem::F32,
        }
    }

    /// The type the products are formed in.
    pub fn mul(self) -> Elem {
        match self {
            Kind::F32 | Kind::F32Unaligned => Elem::F32,
            _ => Elem::F16,
        }
    }

    pub fn acc(self) -> Elem {
        match self {
            Kind::F16Acc => Elem::F16,
            _ => Elem::F32,
        }
    }

    pub fn tensor_cores(self) -> bool {
        self.mul() == Elem::F16
    }

    /// Whether every dimension must be a multiple of the tile.
    pub fn aligned(self) -> bool {
        self != Kind::F32Unaligned
    }

    pub fn baseline(self) -> Baseline {
        match self {
            Kind::F32 | Kind::F32Unaligned => Baseline::Sgemm,
            Kind::F32StoreF16Mul | Kind::F16StoreF32Acc => Baseline::GemmExF16F32,
            Kind::F16Acc => Baseline::GemmExF16F16,
        }
    }

    /// Whether cuBLAS needs A and B converted to f16 first: the Phobos kernel
    /// reads f32 and rounds while staging, which no cuBLAS call does.
    pub fn baseline_converts(self) -> bool {
        self == Kind::F32StoreF16Mul
    }

    /// The scalars the timed runs use, as the CSV has always recorded them.
    pub fn timed_scalars(self) -> (f32, f32) {
        match self {
            Kind::F32 | Kind::F32Unaligned => (1.5, 2.5),
            _ => (1.0, 1.0),
        }
    }

    pub fn precision(self) -> Precision {
        match (self.mul(), self.acc()) {
            (Elem::F32, _) => Precision::F32,
            (Elem::F16, Elem::F32) => Precision::F16TcF32,
            (Elem::F16, Elem::F16) => Precision::F16Tc,
        }
    }

    /// The numerical contract, spelled out for every report row.
    pub fn contract(self) -> String {
        format!(
            "A,B {} C {} mul {}{} acc {}",
            self.ab().label(),
            self.c().label(),
            self.mul().label(),
            if self.tensor_cores() { " (tensor cores)" } else { "" },
            self.acc().label()
        )
    }

    pub fn contract_json(self) -> serde_json::Value {
        serde_json::json!({
            "a_b": self.ab().label(),
            "c": self.c().label(),
            "mul": self.mul().label(),
            "acc": self.acc().label(),
            "tensor_cores": self.tensor_cores(),
        })
    }
}

impl Baseline {
    pub fn label(self) -> &'static str {
        match self {
            Baseline::Sgemm => "cublasSgemm",
            Baseline::GemmExF16F32 => "cublasGemmEx(16F,16F,32F,COMPUTE_32F)",
            Baseline::GemmExF16F16 => "cublasGemmEx(16F,16F,16F,COMPUTE_16F)",
        }
    }
}
