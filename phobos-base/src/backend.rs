// The compilation back end: how a target's kernels get from MLIR to text a
// driver will load.
//
// This is the half of the target seam that lives below phobos-lang, because
// phobos-mlir runs the lowering and cannot depend on the language crate. The
// other half, the instruction vocabulary the emitter builds a module out of,
// is `codegen::target::Isa` in phobos-lang. A second target needs both, and
// they meet here only in the sense that the same [`GpuConfig`] selects them.

use crate::context::{GpuConfig, NvidiaGpuConfig};

/// The LLVM machine a back end compiles its module for.
pub struct LlvmTarget<'a> {
    pub triple: &'a str,
    pub cpu: &'a str,
    pub features: &'a str,
}

/// One target's lowering: the passes that take a `gpu.module` down to LLVM IR,
/// the machine that IR is compiled for, and a last look at the result.
pub trait Backend {
    /// What the generated text is, for logs and error messages.
    fn code_name(&self) -> &'static str;

    /// The MLIR pass pipeline lowering a `gpu.module` to LLVM IR. `index_bits`
    /// is the width index values lower to, which several passes take as an
    /// option rather than infer.
    fn pipeline(&self, index_bits: u32) -> String;

    fn llvm_target(&self) -> LlvmTarget<'_>;

    /// A last pass over the generated text, for what the back end could not fix
    /// earlier. `dynamic_shared` is whether any kernel wants a launch-time
    /// shared allocation, which is the one thing a text patch has so far had to
    /// know. Most targets want the default.
    fn post_process(&self, code: String, dynamic_shared: bool) -> String {
        let _ = dynamic_shared;
        code
    }
}

impl GpuConfig {
    pub fn backend(&self) -> &dyn Backend {
        match self {
            GpuConfig::Nvidia(nv) => nv,
        }
    }
}

impl Backend for NvidiaGpuConfig {
    fn code_name(&self) -> &'static str {
        "PTX"
    }

    fn pipeline(&self, index_bits: u32) -> String {
        format!(
            "builtin.module(     \
                gpu-kernel-outlining,                \
                nvvm-attach-target{{chip={} features={} O=3}}, \
                gpu.module(                          \
                    expand-strided-metadata,         \
                    lower-affine,                    \
                    convert-scf-to-cf,               \
                    convert-math-to-llvm,            \
                    convert-nvgpu-to-nvvm,           \
                    convert-gpu-to-nvvm{{index-bitwidth={index_bits}}}, \
                    convert-vector-to-llvm{{vector-contract-lowering=outerproduct}}, \
                    convert-arith-to-llvm{{index-bitwidth={index_bits}}}, \
                    cse,                             \
                    canonicalize,                    \
                    sccp,                            \
                    reconcile-unrealized-casts       \
                ),                                   \
                gpu-to-llvm                          \
            )",
            self.chip(),
            self.features(),
        )
    }

    fn llvm_target(&self) -> LlvmTarget<'_> {
        LlvmTarget {
            triple: self.target_triple(),
            cpu: self.chip(),
            features: self.features(),
        }
    }

    fn post_process(&self, code: String, dynamic_shared: bool) -> String {
        if dynamic_shared {
            extern_dynamic_shared(&code)
        } else {
            code
        }
    }
}

/// Patch the demoted dynamic-shared declaration into an external one.
///
/// The back end demotes the allocation into the kernel as a one-byte static
/// object; a launch-time size attaches only to the module-scope external form.
/// This is an issue with MLIR and the NVPTX back end we don't really care
/// about.
fn extern_dynamic_shared(ptx: &str) -> String {
    const NEWLINE: char = '\n';

    let mut declarations = Vec::new();
    let mut body = String::with_capacity(ptx.len());

    for line in ptx.lines() {
        let trimmed = line.trim();
        let demoted = trimmed
            .strip_prefix(".shared .align ")
            .filter(|rest| rest.contains("__dynamic_shmem__"));

        match demoted {
            Some(rest) => {
                let (align, name) = rest.split_once(" .b8 ").unwrap_or(("16", rest));
                let name = name.trim_end_matches(';');
                let declaration = format!(".extern .shared .align {align} .b8 {name}[];");

                if !declarations.contains(&declaration) {
                    declarations.push(declaration);
                }
            }
            None => {
                body.push_str(line);
                body.push(NEWLINE);
            }
        }
    }

    if declarations.is_empty() {
        return body;
    }

    let at = body
        .find(".address_size")
        .and_then(|i| body[i..].find(NEWLINE).map(|j| i + j + 1))
        .unwrap_or(0);

    let mut out = String::with_capacity(body.len() + 64);

    out.push_str(&body[..at]);

    for declaration in &declarations {
        out.push_str(declaration);
        out.push(NEWLINE);
    }

    out.push_str(&body[at..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pipeline_carries_the_chip_and_the_index_width() {
        let pipeline = NvidiaGpuConfig::default().pipeline(64);
        assert!(pipeline.contains("nvvm-attach-target{chip=sm_75 features=+ptx90 O=3}"));
        assert_eq!(pipeline.matches("index-bitwidth=64").count(), 2);
    }

    #[test]
    fn dynamic_shared_is_declared_external_at_module_scope() {
        let ptx = concat!(
            ".version 9.0\n",
            ".target sm_75\n",
            ".address_size 64\n",
            "// __dynamic_shmem__0 has been demoted\n",
            ".visible .entry k(\n",
            ")\n",
            "{\n",
            "\t// demoted variable\n",
            "\t.shared .align 16 .b8 __dynamic_shmem__0;\n",
            "\tret;\n",
            "}\n"
        );
        let out = extern_dynamic_shared(ptx);
        assert!(
            out.contains(".extern .shared .align 16 .b8 __dynamic_shmem__0[];"),
            "no external declaration in:\n{out}"
        );
        assert!(
            !out.contains("\t.shared .align 16 .b8 __dynamic_shmem__0;"),
            "the demoted definition survived:\n{out}"
        );
        // Module scope: before the kernel that uses it.
        let declared = out.find(".extern .shared").expect("declaration");
        let entry = out.find(".visible .entry").expect("kernel");
        assert!(declared < entry, "declared after the kernel:\n{out}");
    }

    #[test]
    fn a_module_without_a_dynamic_allocation_is_left_alone() {
        let ptx = ".version 9.0\n.target sm_75\n";
        let cfg = GpuConfig::Nvidia(NvidiaGpuConfig::default());
        assert_eq!(cfg.backend().post_process(ptx.to_string(), false), ptx);
    }
}
