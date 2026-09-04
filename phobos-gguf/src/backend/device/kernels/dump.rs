// Writes the generated sources of the raw-format kernels to a directory, so
// `phobos-lang`'s `emit` example can turn each into MLIR under every target
// and a codegen change can be diffed over them; these live in `format!`
// strings, not `.ph` files, so the emit sweep is otherwise blind to them.
//
//   PHOBOS_DUMP_DIR=/some/dir cargo test -p phobos-gguf --features cuda \
//       dump_raw_kernel_sources -- --ignored

use super::*;
use crate::quant::Quant;

#[test]
#[ignore = "writes files; run by hand with PHOBOS_DUMP_DIR set"]
fn dump_raw_kernel_sources() {
    let Ok(dir) = std::env::var("PHOBOS_DUMP_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut sources: Vec<(String, String)> = Vec::new();
    for quant in Quant::ALL {
        if let (Some(name), Some(src)) = (qgemm_kernel(quant), qgemm_src(quant)) {
            sources.push((name.to_string(), src));
        }
    }
    let i8: [I8Row; 10] = [
        (
            iq1s_qdot_i8_matvec_src,
            IQ1S_I8_TN,
            IQ1S_I8_NARROW_TN,
            "iq1s",
        ),
        (
            iq1m_qdot_i8_matvec_src,
            IQ1M_I8_TN,
            IQ1M_I8_NARROW_TN,
            "iq1m",
        ),
        (
            iq2xxs_qdot_i8_matvec_src,
            IQ2XXS_I8_TN,
            IQ2XXS_I8_NARROW_TN,
            "iq2xxs",
        ),
        (
            iq2xs_qdot_i8_matvec_src,
            IQ2XS_I8_TN,
            IQ2XS_I8_NARROW_TN,
            "iq2xs",
        ),
        (
            iq2s_qdot_i8_matvec_src,
            IQ2S_I8_TN,
            IQ2S_I8_NARROW_TN,
            "iq2s",
        ),
        (
            iq3xxs_qdot_i8_matvec_src,
            IQ3XXS_I8_TN,
            IQ3XXS_I8_NARROW_TN,
            "iq3xxs",
        ),
        (
            iq3s_qdot_i8_matvec_src,
            IQ3S_I8_TN,
            IQ3S_I8_NARROW_TN,
            "iq3s",
        ),
        (q4k_qdot_i8_matvec_src, Q4K_I8_TN, Q4K_I8_NARROW_TN, "q4k"),
        (q5k_qdot_i8_matvec_src, Q5K_I8_TN, Q5K_I8_NARROW_TN, "q5k"),
        (q6k_qdot_i8_matvec_src, Q6K_I8_TN, Q6K_I8_NARROW_TN, "q6k"),
    ];
    for (src, wide, narrow, name) in i8 {
        sources.push((format!("{name}_qdot_i8_matvec_{wide}"), src(wide)));
        sources.push((format!("{name}_qdot_i8_matvec_{narrow}"), src(narrow)));
    }
    for (name, src) in sources {
        std::fs::write(dir.join(format!("{name}.ph")), src).unwrap();
    }
}
