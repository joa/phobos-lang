// The raw formats' kernels, compiled only for the formats a model holds.
// `spec` says which kernels each format has, and `build` compiles every
// requested format's set in one parallel batch.

use super::*;

/// One compiled kernel, the function it launches, and its output tile.
pub(super) struct Kernel {
    pub(super) module: Module,
    pub(super) name: &'static str,
    pub(super) tn: usize,
}

/// What one raw format decodes with. A field is `None` where the format has
/// no such kernel.
#[derive(Default)]
pub(super) struct FormatKernels {
    /// The masked matvec, for any `m` and a ragged `n`.
    pub(super) matvec: Option<Kernel>,
    /// The `m == 1` decode as one `*_qdot_t` call, with no shared-memory
    /// staging. Needs `n` to be a multiple of its tile.
    pub(super) qdot: Option<Kernel>,
    /// The dp4a decode matvec, wide tile then narrow. See `I8_NARROW_TN`.
    pub(super) qdot_i8: [Option<Kernel>; 2],
    /// The decode without the reduction. Writes a `[K, N]` strip of
    /// dequantized weight for [`DeviceBackend::project_raw_dense`].
    pub(super) dequant: Option<Kernel>,
    /// The warp-collective form of `dequant`, one `*_qdecode_t` call writing
    /// an f32 strip, then the same writing an f16 one. Needs a strip that is
    /// a multiple of its tile.
    pub(super) qdecode: [Option<Kernel>; 2],
    /// The prompt projection, decoding inside an integer tensor-core
    /// contraction. See `qmma_raw.rs`.
    pub(super) qmma: Option<Kernel>,
}

type Src = fn(usize) -> String;
type QmmaSrc = fn(usize, usize, usize) -> String;

/// A format's kernel sources: each builder with the function it defines.
/// The dp4a matvec also carries its wide and narrow tiles.
struct Spec {
    tn: usize,
    matvec: Option<(Src, &'static str)>,
    qdot: Option<(Src, &'static str)>,
    qdot_i8: Option<(Src, usize, usize, &'static str)>,
    dequant: Option<(Src, &'static str)>,
    qdecode: Option<(Src, &'static str)>,
    qmma: Option<(QmmaSrc, &'static str)>,
}

const NONE: Spec = Spec {
    tn: 0,
    matvec: None,
    qdot: None,
    qdot_i8: None,
    dequant: None,
    qdecode: None,
    qmma: None,
};

/// The kernels a raw format decodes with, or `None` for a format with none,
/// such as Q8_0, which [`Backend::constant_quant`] unpacks instead.
fn spec(quant: Quant) -> Option<Spec> {
    Some(match quant {
        Quant::Q2_K => Spec {
            tn: Q2K_TN,
            matvec: Some((q2k_matvec_src, "q2k_matvec")),
            qdot: Some((q2k_qdot_matvec_src, "q2k_qdot_matvec")),
            dequant: Some((q2k_dequant_src, "q2k_dequant")),
            ..NONE
        },
        Quant::Q3_K => Spec {
            tn: Q3K_TN,
            matvec: Some((q3k_matvec_src, "q3k_matvec")),
            qdot: Some((q3k_qdot_matvec_src, "q3k_qdot_matvec")),
            ..NONE
        },
        Quant::IQ1_S => Spec {
            tn: IQ1S_TN,
            matvec: Some((iq1s_matvec_src, "iq1s_matvec")),
            qdot: Some((iq1s_qdot_matvec_src, "iq1s_qdot_matvec")),
            qdot_i8: Some((
                iq1s_qdot_i8_matvec_src,
                IQ1S_I8_TN,
                IQ1S_I8_NARROW_TN,
                "iq1s_qdot_i8_matvec",
            )),
            dequant: Some((iq1s_dequant_src, "iq1s_dequant")),
            qdecode: Some((iq1s_qdecode_src, "iq1s_qdecode")),
            qmma: Some((iq1s_qmma_src, "iq1s_qmma")),
        },
        Quant::IQ1_M => Spec {
            tn: IQ1M_TN,
            matvec: Some((iq1m_matvec_src, "iq1m_matvec")),
            qdot: Some((iq1m_qdot_matvec_src, "iq1m_qdot_matvec")),
            qdot_i8: Some((
                iq1m_qdot_i8_matvec_src,
                IQ1M_I8_TN,
                IQ1M_I8_NARROW_TN,
                "iq1m_qdot_i8_matvec",
            )),
            dequant: Some((iq1m_dequant_src, "iq1m_dequant")),
            qdecode: Some((iq1m_qdecode_src, "iq1m_qdecode")),
            qmma: None,
        },
        Quant::IQ2_XXS => Spec {
            tn: IQ2XXS_TN,
            matvec: Some((iq2xxs_matvec_src, "iq2xxs_matvec")),
            qdot: Some((iq2xxs_qdot_matvec_src, "iq2xxs_qdot_matvec")),
            qdot_i8: Some((
                iq2xxs_qdot_i8_matvec_src,
                IQ2XXS_I8_TN,
                IQ2XXS_I8_NARROW_TN,
                "iq2xxs_qdot_i8_matvec",
            )),
            dequant: Some((iq2xxs_dequant_src, "iq2xxs_dequant")),
            qdecode: Some((iq2xxs_qdecode_src, "iq2xxs_qdecode")),
            qmma: Some((iq2xxs_qmma_src, "iq2xxs_qmma")),
        },
        Quant::IQ2_XS => Spec {
            tn: IQ2XS_TN,
            matvec: Some((iq2xs_matvec_src, "iq2xs_matvec")),
            qdot: Some((iq2xs_qdot_matvec_src, "iq2xs_qdot_matvec")),
            qdot_i8: Some((
                iq2xs_qdot_i8_matvec_src,
                IQ2XS_I8_TN,
                IQ2XS_I8_NARROW_TN,
                "iq2xs_qdot_i8_matvec",
            )),
            dequant: Some((iq2xs_dequant_src, "iq2xs_dequant")),
            qdecode: Some((iq2xs_qdecode_src, "iq2xs_qdecode")),
            qmma: Some((iq2xs_qmma_src, "iq2xs_qmma")),
        },
        Quant::IQ2_S => Spec {
            tn: IQ2S_TN,
            matvec: Some((iq2s_matvec_src, "iq2s_matvec")),
            qdot: Some((iq2s_qdot_matvec_src, "iq2s_qdot_matvec")),
            qdot_i8: Some((
                iq2s_qdot_i8_matvec_src,
                IQ2S_I8_TN,
                IQ2S_I8_NARROW_TN,
                "iq2s_qdot_i8_matvec",
            )),
            dequant: Some((iq2s_dequant_src, "iq2s_dequant")),
            qdecode: Some((iq2s_qdecode_src, "iq2s_qdecode")),
            qmma: Some((iq2s_qmma_src, "iq2s_qmma")),
        },
        Quant::IQ3_XXS => Spec {
            tn: IQ3XXS_TN,
            matvec: Some((iq3xxs_matvec_src, "iq3xxs_matvec")),
            qdot: Some((iq3xxs_qdot_matvec_src, "iq3xxs_qdot_matvec")),
            qdot_i8: Some((
                iq3xxs_qdot_i8_matvec_src,
                IQ3XXS_I8_TN,
                IQ3XXS_I8_NARROW_TN,
                "iq3xxs_qdot_i8_matvec",
            )),
            dequant: Some((iq3xxs_dequant_src, "iq3xxs_dequant")),
            qdecode: Some((iq3xxs_qdecode_src, "iq3xxs_qdecode")),
            qmma: Some((iq3xxs_qmma_src, "iq3xxs_qmma")),
        },
        Quant::IQ3_S => Spec {
            tn: IQ3S_TN,
            matvec: Some((iq3s_matvec_src, "iq3s_matvec")),
            qdot: Some((iq3s_qdot_matvec_src, "iq3s_qdot_matvec")),
            qdot_i8: Some((
                iq3s_qdot_i8_matvec_src,
                IQ3S_I8_TN,
                IQ3S_I8_NARROW_TN,
                "iq3s_qdot_i8_matvec",
            )),
            dequant: Some((iq3s_dequant_src, "iq3s_dequant")),
            qdecode: Some((iq3s_qdecode_src, "iq3s_qdecode")),
            qmma: Some((iq3s_qmma_src, "iq3s_qmma")),
        },
        Quant::IQ4_XS => Spec {
            tn: IQ4XS_TN,
            matvec: Some((iq4xs_matvec_src, "iq4xs_matvec")),
            qdot: Some((iq4xs_qdot_matvec_src, "iq4xs_qdot_matvec")),
            dequant: Some((iq4xs_dequant_src, "iq4xs_dequant")),
            ..NONE
        },
        Quant::Q4_K => i8_only(
            q4k_qdot_i8_matvec_src,
            Q4K_I8_TN,
            Q4K_I8_NARROW_TN,
            "q4k_qdot_i8_matvec",
        ),
        Quant::Q5_K => i8_only(
            q5k_qdot_i8_matvec_src,
            Q5K_I8_TN,
            Q5K_I8_NARROW_TN,
            "q5k_qdot_i8_matvec",
        ),
        Quant::Q6_K => i8_only(
            q6k_qdot_i8_matvec_src,
            Q6K_I8_TN,
            Q6K_I8_NARROW_TN,
            "q6k_qdot_i8_matvec",
        ),
        Quant::PTQ1_0 => i8_only(
            ptq1_qdot_i8_matvec_src,
            PTQ1_I8_TN,
            PTQ1_I8_NARROW_TN,
            "ptq1_qdot_i8_matvec",
        ),
        _ => return None,
    })
}

/// A format whose only kernel is the dp4a decode matvec.
fn i8_only(src: Src, wide: usize, narrow: usize, name: &'static str) -> Spec {
    Spec {
        qdot_i8: Some((src, wide, narrow, name)),
        ..NONE
    }
}

#[derive(Clone, Copy)]
enum Slot {
    Matvec,
    Qdot,
    QdotI8(usize),
    Dequant,
    Qdecode(usize),
    Qmma,
}

/// One kernel to compile, and where in its format's set it goes.
struct Job {
    quant: Quant,
    slot: Slot,
    source: String,
    shapes: Vec<(&'static str, usize)>,
    name: &'static str,
    tn: usize,
}

/// The compile jobs for one format's whole set.
fn jobs(quant: Quant, spec: &Spec, into: &mut Vec<Job>) {
    let mut push = |slot, source, shapes: Vec<_>, name, tn| {
        into.push(Job {
            quant,
            slot,
            source,
            shapes,
            name,
            tn,
        })
    };
    let tn = spec.tn;
    let one = |tn| vec![("TN", tn)];
    if let Some((src, name)) = spec.matvec {
        push(Slot::Matvec, src(tn), one(tn), name, tn);
    }
    if let Some((src, name)) = spec.qdot {
        push(Slot::Qdot, src(tn), one(tn), name, tn);
    }
    if let Some((src, wide, narrow, name)) = spec.qdot_i8 {
        let wide = qdot_i8_tn(wide);
        push(Slot::QdotI8(0), src(wide), one(wide), name, wide);
        push(Slot::QdotI8(1), src(narrow), one(narrow), name, narrow);
    }
    if let Some((src, name)) = spec.dequant {
        push(Slot::Dequant, src(tn), one(tn), name, tn);
    }
    if let Some((src, name)) = spec.qdecode {
        let body = src(tn);
        // The f16 strip is the same kernel with a narrower destination.
        let f16 = body.replace("SCRATCH: tensor<f32>[K, N]", "SCRATCH: tensor<f16>[K, N]");
        push(Slot::Qdecode(0), body, one(tn), name, tn);
        push(Slot::Qdecode(1), f16, one(tn), name, tn);
    }
    if let Some((src, name)) = spec.qmma {
        let (qtm, qtn, qcta) = qmma_tile();
        push(
            Slot::Qmma,
            src(qcta, qtm, qtn),
            vec![("TM", qtm), ("TN", qtn)],
            name,
            qtn,
        );
    }
}

/// Compiles the kernels of every format in `formats` in one parallel batch.
/// Each format in `formats` gets an entry, empty for one with no kernels.
pub(super) fn build(formats: &[Quant]) -> Result<HashMap<Quant, FormatKernels>> {
    let mut all = Vec::new();
    for &quant in formats {
        if let Some(spec) = spec(quant) {
            jobs(quant, &spec, &mut all);
        }
    }
    let entries: Vec<Entry> = all
        .iter()
        .map(|job| (job.source.as_str(), job.shapes.as_slice(), job.name))
        .collect();
    let modules = compile_parallel(&entries)?;

    let mut built: HashMap<Quant, FormatKernels> = formats
        .iter()
        .map(|&quant| (quant, FormatKernels::default()))
        .collect();
    for (job, module) in all.into_iter().zip(modules) {
        let kernel = Some(Kernel {
            module,
            name: job.name,
            tn: job.tn,
        });
        let set = built
            .get_mut(&job.quant)
            .expect("every job's format has an entry");
        match job.slot {
            Slot::Matvec => set.matvec = kernel,
            Slot::Qdot => set.qdot = kernel,
            Slot::QdotI8(i) => set.qdot_i8[i] = kernel,
            Slot::Dequant => set.dequant = kernel,
            Slot::Qdecode(i) => set.qdecode[i] = kernel,
            Slot::Qmma => set.qmma = kernel,
        }
    }
    Ok(built)
}

/// The formats a backend for `wanted` compiles: `wanted`, or every format
/// while `PHOBOS_KERNEL_MANIFEST` records one, since a release's cache is
/// warmed from what the recorded runs ask for.
pub(super) fn to_build(wanted: &[Quant]) -> Vec<Quant> {
    let recording = std::env::var_os("PHOBOS_KERNEL_MANIFEST").is_some_and(|v| !v.is_empty());
    Quant::ALL
        .into_iter()
        .filter(|q| recording || wanted.contains(q))
        .collect()
}

impl DeviceBackend {
    /// The kernels of `quant`, which must be among the formats the backend
    /// was made for.
    pub(super) fn format_kernels(&self, quant: Quant) -> Result<&FormatKernels> {
        self.formats.get(&quant).with_context(|| {
            format!(
                "no kernels were built for {}; this backend was made for another model's formats",
                quant.name()
            )
        })
    }
}

/// `kernel`, or an error naming the format and the kind it lacks.
pub(super) fn need<'a>(kernel: &'a Option<Kernel>, quant: Quant, kind: &str) -> Result<&'a Kernel> {
    kernel
        .as_ref()
        .with_context(|| format!("{} has no {kind} kernel", quant.name()))
}
