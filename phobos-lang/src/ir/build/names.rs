// The built-ins a call may name, which an unknown one is compared against.

use std::sync::OnceLock;

use crate::ast::Scalar;
use crate::ir::{Intrinsic, RawFmt};

/// Built-ins with a name of their own; the conversions and the quantized
/// families are added from their tables.
const NAMED: &[&str] = &[
    "barrier", "grid_barrier", "atomic_add", "atomic_cas", "load_acquire", "store_release",
    "exp", "log", "pow", "sqrt", "tanh", "round", "floor", "ceil", "abs", "rsqrt", "sin", "cos",
    "min", "max", "where", "iota", "tmax", "argsel", "rowmax", "rowsum", "rowargmax", "cumsum",
    "tril", "transpose", "flat", "gather", "dot", "dot_t", "qdot_t", "qmma_t", "q50_qdot_t",
    "q50_qmma_t", "q8_qgemm_t", "iq1s_qmma_t", "sum", "elem", "store", "word", "dp4a", "popcount", "byteperm",
    "f16bits", "extent",
];

/// Every name a call may take.
pub(crate) fn builtins() -> &'static [String] {
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let conversions = Scalar::NAMES.iter().filter(|&&n| n != "bool").map(|n| n.to_string());
        let quantized = RawFmt::ALL
            .iter()
            .flat_map(|f| ["qmma_staged_t", "qgemm_t"].map(|suffix| format!("{}_{suffix}", f.name())))
            .filter(|name| Intrinsic::from_name(name).is_some());
        NAMED.iter().map(|n| n.to_string()).chain(conversions).chain(quantized).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::builtins;

    #[test]
    fn every_listed_builtin_is_one() {
        let base = phobos_base::context::Context::default();
        let unknown: Vec<_> = builtins()
            .iter()
            .filter(|name| {
                let src = format!("kernel k(A: tensor<f32>[N]) {{\n    let x = {name}()\n}}\n");
                crate::compile_raw(&base, &src).is_err_and(|e| e.to_string().contains("unknown function"))
            })
            .collect();
        assert!(unknown.is_empty(), "listed, but the build does not know them: {unknown:?}");
    }

    #[test]
    fn a_format_without_the_family_is_not_a_builtin() {
        // SPEC.md's format table leaves these out; the build used to take
        // the name and fail in codegen.
        let base = phobos_base::context::Context::default();
        for name in ["q2k_qgemm_t", "iq4xs_qgemm_t", "q4k_qmma_staged_t", "iq1m_qmma_staged_t", "q6k_qmma_t"] {
            assert!(!builtins().iter().any(|b| b == name), "{name} is listed");
            let src = format!("kernel k(A: tensor<f32>[N]) {{\n    let x = {name}()\n}}\n");
            let Err(err) = crate::compile_raw(&base, &src) else { panic!("{name} compiled") };
            assert!(err.to_string().contains("unknown function"), "{name}: {err}");
        }
    }

    #[test]
    fn every_builtin_the_spec_names_is_listed() {
        // Calls the spec writes that are not built-ins: a user `fn f`, and
        // the standard library's functions.
        const PROSE: [&str; 1] = ["f"];
        let library = crate::stdlib::builtin();
        let spec = include_str!("../../../../SPEC.md");
        let missing: Vec<_> = spec
            .split('`')
            .skip(1)
            .step_by(2)
            .filter_map(|code| code.split_once('(').map(|(name, _)| name))
            .filter(|name| !name.is_empty() && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'))
            .filter(|name| !PROSE.contains(name) && !builtins().iter().any(|b| b == name))
            .filter(|name| !library.iter().any(|f| f.name == *name))
            .collect();
        assert!(missing.is_empty(), "SPEC.md calls these, but the builtin list lacks them: {missing:?}");
    }
}
