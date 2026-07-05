use cust::device::{Device, DeviceAttribute};
use phobos_base::phinfo;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy)]
pub enum Precision {
    /// f32 CUDA cores.
    F32,
    /// f16 tensor cores,
    F16Tc,
    /// f16 tensor cores; f32 accumulate.
    F16TcF32,
}

impl Precision {
    fn label(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16Tc => "f16-tc",
            Precision::F16TcF32 => "f16-tc-f32-acc",
        }
    }

    /// The theoretical peak (GFLOP/s) to normalize this kernel against.
    fn peak_gflops(self, peaks: &Peaks) -> f64 {
        match self {
            Precision::F32 => peaks.fp32_gflops,
            Precision::F16Tc => peaks.fp16_tc_gflops,
            Precision::F16TcF32 => peaks.fp16_tc_f32_acc_gflops,
        }
    }
}

/// One row of the CSV: a single implementation's measured throughput on a
/// single benchmark.
struct Record {
    benchmark: String,
    implementation: &'static str,
    precision: Precision,
    gflops: f64,
}

/// Collects benchmark results across a run for CSV export.
#[derive(Default)]
pub struct Results {
    records: Vec<Record>,
}

impl Results {
    /// Record one implementation's throughput. gflops is the achieved rate
    /// (total GFLOP / seconds), not the operation count.
    pub fn push(
        &mut self,
        benchmark: &str,
        implementation: &'static str,
        precision: Precision,
        gflops: f64,
    ) {
        self.records.push(Record {
            benchmark: benchmark.to_string(),
            implementation,
            precision,
            gflops,
        });
    }

    /// Render the collected results as CSV text. Columns:
    /// benchmark,impl,precision,gflops,peak_gflops,pct_of_peak.
    fn to_csv(&self, peaks: &Peaks) -> String {
        let mut out = String::from("benchmark,impl,precision,gflops,peak_gflops,pct_of_peak\n");
        for r in &self.records {
            let peak = r.precision.peak_gflops(peaks);
            let pct = 100.0 * r.gflops / peak;
            let _ = writeln!(
                out,
                "{},{},{},{:.1},{:.1},{:.1}",
                r.benchmark,
                r.implementation,
                r.precision.label(),
                r.gflops,
                peak,
                pct,
            );
        }
        out
    }

    /// Write the CSV to path, using peaks for the reference columns.
    pub fn write_csv(&self, path: &Path, peaks: &Peaks) -> anyhow::Result<()> {
        fs::write(path, self.to_csv(peaks))?;
        phinfo!("wrote {} rows to {}", self.records.len(), path.display());
        Ok(())
    }
}

/// The GPU's theoretical peak throughput, used as the denominator for the
/// "% of peak" column. Derived from device attributes (SM count, clock,
/// compute capability) unless overridden on the command line.
pub struct Peaks {
    fp32_gflops: f64,
    fp16_tc_gflops: f64,
    fp16_tc_f32_acc_gflops: f64,
}

impl Peaks {
    /// Detect peaks from device 0, applying any command-line overrides (given
    /// in TFLOP/s). The per-architecture throughput figures are vendor-spec
    /// dense rates and best-effort across products (consumer parts in
    /// particular vary); the override flags exist to correct them. The chosen
    /// values are logged so the assumption is visible.
    pub fn detect(
        peak_fp32_tflops: Option<f64>,
        peak_fp16_tc_tflops: Option<f64>,
        peak_fp16_tc_f32_acc_tflops: Option<f64>,
    ) -> anyhow::Result<Peaks> {
        let dev = Device::get_device(0)?;
        let name = dev.name()?;
        let sms = dev.get_attribute(DeviceAttribute::MultiprocessorCount)? as f64;
        // ClockRate is in kHz.
        let clock_hz = dev.get_attribute(DeviceAttribute::ClockRate)? as f64 * 1e3;
        let cc = (dev.get_attribute(DeviceAttribute::ComputeCapabilityMajor)? * 10
            + dev.get_attribute(DeviceAttribute::ComputeCapabilityMinor)?) as u32;

        // FP32 FMA lanes per SM. 64 on Volta/Turing/GA100; 128 on consumer
        // Ampere, Ada, Hopper and assumed for anything newer.
        let fp32_lanes = match cc {
            70 | 72 | 75 | 80 => 64.0,
            _ => 128.0,
        };
        // FP16 tensor-core FLOPs per SM per clock (2 * FMA lanes). Consumer Ada
        // (sm_89) runs f32-accumulate at half rate; A100/GA10x sit at 2048;
        // Hopper at 4096; Volta/Turing at 1024.
        let tc_fp16_per_sm = match cc {
            cc if cc >= 90 => 4096.0,
            89 => 1024.0,
            cc if cc >= 80 => 2048.0,
            cc if cc >= 70 => 1024.0,
            _ => 0.0,
        };

        let derived_fp32 = sms * fp32_lanes * 2.0 * clock_hz / 1e9;
        let derived_fp16_tc = sms * tc_fp16_per_sm * clock_hz / 1e9;
        let derived_fp16_tc_f32_acc = sms * tc_fp16_per_sm * clock_hz / 2.0 / 1e9;

        // Overrides are given in TFLOP/s; store GFLOP/s.
        let fp32_gflops = peak_fp32_tflops.map_or(derived_fp32, |t| t * 1e3);
        let fp16_tc_gflops = peak_fp16_tc_tflops.map_or(derived_fp16_tc, |t| t * 1e3);
        let fp16_tc_f32_acc_gflops = peak_fp16_tc_f32_acc_tflops.map_or(derived_fp16_tc_f32_acc, |t| t * 1e3);

        phinfo!(
            "gpu: {} ({:.0} SMs @ {:.2} GHz, sm_{cc})",
            name,
            sms,
            clock_hz / 1e9,
        );
        phinfo!(
            "theoretical peak: fp32 {:.1} TFLOP/s{}, fp16-tc {:.1} TFLOP/s{}, fp16-tc-f32-acc {:.1} TFLOP/s{}",
            fp32_gflops / 1e3,
            if peak_fp32_tflops.is_some() {
                " (override)"
            } else {
                ""
            },
            fp16_tc_gflops / 1e3,
            if peak_fp16_tc_tflops.is_some() {
                " (override)"
            } else {
                ""
            },
            fp16_tc_f32_acc_gflops / 1e3,
            if peak_fp16_tc_f32_acc_tflops.is_some() {
                " (override)"
            } else {
                ""
            },
        );
        Ok(Peaks {
            fp32_gflops,
            fp16_tc_gflops,
            fp16_tc_f32_acc_gflops,
        })
    }
}
