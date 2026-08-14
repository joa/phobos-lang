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
    /// Chip's compute capability as a number
    ///
    /// Example: sm_75 is 75, sm_90a is 90.
    ///
    /// This is what selects a target's instruction vocabulary; everything the
    /// number then decides lives behind that vocabulary rather than here.
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

    pub fn chip(&self) -> &str {
        &self.chip
    }

    pub fn features(&self) -> &str {
        &self.features
    }

    pub fn target_triple(&self) -> &str {
        &self.target_triple
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
    fn a_chip_name_reads_as_a_compute_capability() {
        for (chip, cc) in [("sm_75", 75), ("sm_90a", 90), ("sm_61", 61)] {
            let cfg = GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip));
            assert_eq!(cfg.compute_capability(), cc, "{chip}");
        }
    }
}
