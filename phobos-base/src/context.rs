use std::collections::HashMap;

#[derive(Debug)]
pub struct Context {
    /// Whether to print the output of different compiler phases.
    pub print_phases: bool,

    /// The GPU config.
    pub gpu_config: GpuConfig,

    /// Values for @autotune search dims.
    pub shape_overrides: HashMap<String, i64>,

    /// Bit width index values lower to.
    pub index_bitwidth: u32,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            print_phases: false,
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::default()),
            shape_overrides: HashMap::new(),
            index_bitwidth: 32,
        }
    }
}

#[derive(Debug)]
pub enum GpuConfig {
    Nvidia(NvidiaGpuConfig),
}

impl GpuConfig {
    pub fn mlir_target_pass(&self) -> String {
        match self {
            GpuConfig::Nvidia(nv) => format!(
                "nvvm-attach-target{{chip={} features={} O=3}}",
                nv.chip, nv.features
            ),
        }
    }

    pub fn llvm_cpu(&self) -> &str {
        match self {
            GpuConfig::Nvidia(nv) => &nv.chip,
        }
    }

    pub fn llvm_features(&self) -> &str {
        match self {
            GpuConfig::Nvidia(nv) => &nv.features,
        }
    }

    pub fn llvm_target_triple(&self) -> &str {
        match self {
            GpuConfig::Nvidia(nv) => &nv.target_triple,
        }
    }

    /// Chip's compute capability as a number
    ///
    /// Example: sm_75 is 75, sm_90a is 90.
    ///
    /// TODO(joa): how to map this across vendors
    pub fn compute_capability(&self) -> u32 {
        match self {
            GpuConfig::Nvidia(nv) => nv
                .chip
                .trim_start_matches("sm_")
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0),
        }
    }

    /// Whether the chip supports cp.async
    pub fn supports_cp_async(&self) -> bool {
        // 80 is Ampere
        self.compute_capability() >= 80
    }

    pub fn supports_bf16_native(&self) -> bool {
        self.compute_capability() >= 80
    }

    pub fn supports_dp4a(&self) -> bool {
        self.compute_capability() >= 61
    }

    pub fn supports_int8_mma(&self) -> bool {
        self.compute_capability() >= 75
    }

    /// The k-dimension of the native f16 mma.sync
    pub fn mma_sync_k(&self) -> Option<u32> {
        match self.compute_capability() {
            cc if cc >= 80 => Some(16), // Ampere and later -> m16n8k16
            cc if cc >= 75 => Some(8),  // Turing's op is m16n8k8
            _ => None,
        }
    }

    /// Bytes of shared memory an SM can hand out across its resident CTAs.
    pub fn smem_per_sm(&self) -> u32 {
        (match self.compute_capability() {
            cc if cc >= 90 => 228, // Hopper
            87 => 164,             // Orin
            cc if cc >= 86 => 100, // Ada / GA10x
            cc if cc >= 80 => 164, // A100
            cc if cc >= 75 => 64,  // Turing
            cc if cc >= 70 => 96,  // Volta
            _ => 48,
        }) * 1024
    }

    /// 32-bit registers in an SM.
    pub fn regs_per_sm(&self) -> u32 {
        64 * 1024 // 64K on every CUDA arch since Kepler
    }

    pub fn max_warps_per_sm(&self) -> u32 {
        match self.compute_capability() {
            75 => 32,           // Turing
            86 | 87 | 89 => 48, // Ada / GA10x / Orin
            _ => 64,            // Volta, A100, Hopper
        }
    }
}

#[derive(Debug)]
pub struct NvidiaGpuConfig {
    chip: String,
    features: String,
    target_triple: String,
}

impl NvidiaGpuConfig {
    pub fn with_chip(chip: impl Into<String>) -> Self {
        NvidiaGpuConfig {
            chip: chip.into(),
            ..Default::default()
        }
    }
}

impl Default for NvidiaGpuConfig {
    fn default() -> Self {
        NvidiaGpuConfig {
            chip: "sm_75".to_string(),
            features: "+ptx90".to_string(),
            target_triple: "nvptx64-nvidia-cuda".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::context::{GpuConfig, NvidiaGpuConfig};

    #[test]
    fn nvgpuconfig_mlir_target_pass() {
        let cfg = GpuConfig::Nvidia(NvidiaGpuConfig::default());
        assert_eq!(
            cfg.mlir_target_pass(),
            "nvvm-attach-target{chip=sm_75 features=+ptx90 O=3}"
        )
    }

    #[test]
    fn nvgpuconfig_supports_cp_async() {
        let cfg = GpuConfig::Nvidia(NvidiaGpuConfig::default());
        assert!(
            !cfg.supports_cp_async(),
            "the default config does not support cp.async (Turing)"
        )
    }
}
