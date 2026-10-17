// The built-ins a call may name, which an unknown one is compared against.

use std::sync::OnceLock;

use crate::ast::Scalar;
use crate::ir::RawFmt;

/// Built-ins with a name of their own; the conversions and the quantized
/// families are added from their tables.
const NAMED: &[&str] = &[
    "barrier", "grid_barrier", "atomic_add", "atomic_cas", "load_acquire", "store_release",
    "exp", "log", "pow", "sqrt", "tanh", "round", "floor", "ceil", "abs", "rsqrt", "sin", "cos",
    "min", "max", "where", "iota", "tmax", "argsel", "rowmax", "rowsum", "rowargmax", "cumsum",
    "tril", "transpose", "flat", "gather", "dot", "dot_t", "qdot_t", "qmma_t", "q50_qdot_t",
    "q50_qmma_t", "iq1s_qmma_t", "sum", "elem", "store", "word", "dp4a", "popcount", "byteperm",
    "f16bits", "extent",
];

/// Every name a call may take.
pub(crate) fn builtins() -> &'static [String] {
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let conversions = Scalar::NAMES.iter().filter(|&&n| n != "bool").map(|n| n.to_string());
        let quantized = RawFmt::ALL
            .iter()
            .flat_map(|f| ["qmma_staged_t", "qgemm_t"].map(|suffix| format!("{}_{suffix}", f.name())));
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
    fn every_builtin_the_spec_names_is_listed() {
        // Calls the spec writes that are not built-ins: math in the prose,
        // and a user `fn f`.
        const PROSE: [&str; 2] = ["log2", "f"];
        let spec = include_str!("../../../../SPEC.md");
        let missing: Vec<_> = spec
            .split('`')
            .skip(1)
            .step_by(2)
            .filter_map(|code| code.split_once('(').map(|(name, _)| name))
            .filter(|name| !name.is_empty() && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'))
            .filter(|name| !PROSE.contains(name) && !builtins().iter().any(|b| b == name))
            .collect();
        assert!(missing.is_empty(), "SPEC.md calls these, but the builtin list lacks them: {missing:?}");
    }
}
