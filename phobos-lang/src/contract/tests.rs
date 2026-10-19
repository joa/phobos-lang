use std::path::{Path, PathBuf};

use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};

use super::{Class, Contract, Extent, Part, Role, check_entry};

/// The targets the emit sweep diffs under.
const TARGETS: &[(&str, u32)] = &[("sm_75", 32), ("sm_75", 64), ("sm_80", 64), ("sm_89", 64), ("sm_120", 64)];

fn context(chip: &str, bits: u32) -> Context {
    Context {
        gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip.to_string())),
        index_bitwidth: bits,
        ..Default::default()
    }
}

fn compile(src: &str, chip: &str, bits: u32) -> (String, Vec<Contract>) {
    let out = crate::compile_raw(&context(chip, bits), src).unwrap_or_else(|e| panic!("{e:#}"));
    (out.code, out.contracts)
}

#[test]
fn saxpy_at_32_bits_is_three_descriptors_and_a_float() {
    let (ptx, contracts) = compile(include_str!("../../../examples/saxpy_fp32.ph"), "sm_75", 32);
    let [c] = &contracts[..] else { panic!("{contracts:?}") };
    assert_eq!(check_entry(&ptx, c), Vec::<String>::new());
    assert_eq!((c.index_bits, c.cta_threads, c.slots.len()), (32, 256, 16));
    assert_eq!(c.params[0].shape, Some(vec![Extent::Sym("N".into())]));
    assert_eq!((c.params[0].align_bytes, c.params[3].elem.as_str()), (16, "f32"));
    assert_eq!(c.slots[15].class, Class::Float);
    assert_eq!(c.slots[3].part, Part::Size(0));
    assert_eq!((c.dims[0].name.as_str(), c.dims[0].param, c.dims[0].axis, c.dims[0].multiple_of), ("N", 0, Some(0), 4));
    assert_eq!(c.autotune.get("BLOCK"), Some(&16));
}

#[test]
fn the_tensor_core_gemm_at_sm_80_widens_every_index_slot() {
    let (ptx, contracts) = compile(include_str!("../../../examples/gemm_fp16tc_fp32acc.ph"), "sm_80", 32);
    let [c] = &contracts[..] else { panic!("{contracts:?}") };
    assert_eq!(check_entry(&ptx, c), Vec::<String>::new());
    assert_eq!(c.index_bits, 64);
    assert!(c.slots.iter().filter(|s| s.class == Class::Int).all(|s| s.bytes == 8));
    // `@aligned(M = TILE_M)` at TILE_M's first choice.
    let m = c.dims.iter().find(|d| d.name == "M").unwrap();
    assert_eq!(m.multiple_of, 32);
}

#[test]
fn a_persistent_kernel_publishes_scratch_and_takes_the_barrier_word_last() {
    let src = "@persistent(4)
        kernel k(X: tensor<f32>[64, 64], Y: tensor<f32>[64, 64]) {
            var S: tensor<f32>[64, 64]
            if ctx.block.x == 0 {
                S[0 :+ 64, 0 :+ 64] = X[0 :+ 64, 0 :+ 64]
            }
            grid_barrier()
            Y[0 :+ 64, 0 :+ 64] = S[0 :+ 64, 0 :+ 64] * 2.0
        }";
    let (ptx, contracts) = compile(src, "sm_75", 32);
    let [c] = &contracts[..] else { panic!("{contracts:?}") };
    assert_eq!(check_entry(&ptx, c), Vec::<String>::new());
    let roles: Vec<_> = c.params.iter().map(|p| (p.name.as_str(), p.role)).collect();
    assert_eq!(roles, [("X", Role::Declared), ("Y", Role::Declared), ("S", Role::Scratch), ("grid_barrier", Role::BarrierWord)]);
    assert_eq!(c.persistent.as_ref().and_then(|p| p.blocks), Some(4));
    assert_eq!(c.params[3].shape, Some(vec![Extent::Fixed(2), Extent::Fixed(1)]));
}

#[test]
fn a_dynamic_shared_kernel_reports_the_bytes_a_launch_passes() {
    let src = "@dynshared
        kernel k(A: tensor<f32>[64, 64], C: tensor<f32>[64, 64]) {
            var t = A[0 :+ 64, 0 :+ 64]
            C[0 :+ 64, 0 :+ 64] = t * 2.0
        }";
    let (ptx, contracts) = compile(src, "sm_75", 32);
    let [c] = &contracts[..] else { panic!("{contracts:?}") };
    assert_eq!(check_entry(&ptx, c), Vec::<String>::new());
    assert_eq!((c.static_shared_bytes, c.dynamic_shared_bytes), (0, 64 * 64 * 4));
}

#[test]
fn a_half_scalar_is_two_bytes_the_entry_passes_as_an_array() {
    let src = "kernel k(out: tensor<f32>[N], a: f16, b: f32) {
            out[0] = a + b
        }";
    for (chip, bits) in TARGETS {
        let (ptx, contracts) = compile(src, chip, *bits);
        assert!(ptx.contains(".align 2 .b8 k_param_5[2]"), "{ptx}");
        assert_eq!(check_entry(&ptx, &contracts[0]), Vec::<String>::new());
        assert_eq!((contracts[0].slots[5].class, contracts[0].slots[5].bytes), (Class::Float, 2));
    }
}

#[test]
fn a_contract_round_trips_through_json() {
    let (_, contracts) = compile(include_str!("../../../examples/flash_attention_fp32.ph"), "sm_75", 32);
    for c in contracts {
        let text = serde_json::to_string(&c.to_json()).unwrap();
        assert_eq!(serde_json::from_str::<Contract>(&text).unwrap(), c);
    }
}

/// The kernel sources in the tree that compile on their own: the examples,
/// and the fused chains against the stage library they are written in.
pub(crate) fn tree_sources() -> Vec<(PathBuf, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let read = |p: &Path| std::fs::read_to_string(p).unwrap().replace('\r', "");
    let mut out = Vec::new();
    for dir in ["examples", "phobos-lang/examples"] {
        let mut paths: Vec<_> = std::fs::read_dir(root.join(dir)).unwrap().flatten().map(|e| e.path()).collect();
        paths.sort();
        out.extend(paths.into_iter().filter(|p| p.extension().is_some_and(|e| e == "ph")).map(|p| {
            let text = read(&p);
            (p, text)
        }));
    }
    let fuse = root.join("phobos-gguf/src/backend/fuse");
    let lib = read(&fuse.join("lib.ph"));
    let mut chains: Vec<_> = std::fs::read_dir(fuse.join("chains")).unwrap().flatten().map(|e| e.path()).collect();
    chains.sort();
    for p in chains {
        let text = format!("{lib}\n{}", read(&p));
        out.push((p, text));
    }
    out
}

/// Every `.entry` the tree's kernels compile to, under each target the
/// emit sweep diffs, matches the contract compiled beside it.
#[test]
fn every_entry_in_the_tree_matches_its_contract() {
    let sources = tree_sources();
    assert!(sources.len() >= 15, "found {} sources", sources.len());
    let jobs: Vec<_> = sources.iter().flat_map(|s| TARGETS.iter().map(move |t| (s, *t))).collect();
    let failures = std::sync::Mutex::new(Vec::new());
    let checked = std::sync::atomic::AtomicUsize::new(0);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(jobs.len());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn_scoped(scope, || {
                    while let Some(&((path, src), (chip, bits))) = jobs.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
                        let at = format!("{} on {chip}/{bits}", path.display());
                        let found = match crate::compile_raw(&context(chip, bits), src) {
                            Ok(out) if out.contracts.is_empty() => vec![format!("{at}: no contract")],
                            Ok(out) => {
                                checked.fetch_add(out.contracts.len(), std::sync::atomic::Ordering::Relaxed);
                                out
                                .contracts
                                .iter()
                                .flat_map(|c| check_entry(&out.code, c).into_iter().map(|e| format!("{at}, {}: {e}", c.kernel)).collect::<Vec<_>>())
                                .collect()
                            }
                            Err(e) => vec![format!("{at}: {e:#}")],
                        };
                        failures.lock().unwrap().extend(found);
                    }
                })
                .unwrap();
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
