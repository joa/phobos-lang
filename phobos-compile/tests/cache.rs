// What `phobos-compile` writes to the kernel cache is what the runtime looks
// up: the same key for the same source, chip and pins, with the runtime's
// context, which sets only the chip and the pins.

use std::collections::HashMap;
use std::process::Command;

use phobos_base::context::{Context, GpuConfig, NvidiaGpuConfig};

#[test]
fn the_runtime_finds_what_phobos_compile_cached() {
    let dir = std::env::temp_dir().join(format!("phobos-compile-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // SAFETY: the only test in this binary, so no other thread reads the
    // environment while it is written.
    unsafe { std::env::set_var("PHOBOS_KERNEL_CACHE_DIR", &dir) };
    let source = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/saxpy_fp32.ph");
    let text = std::fs::read_to_string(source).unwrap();
    for (chip, pins) in [("sm_75", vec![]), ("sm_86", vec![("BLOCK", 256)])] {
        let ptx = dir.join(format!("{chip}.ptx"));
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_phobos-compile"));
        cmd.args([source, "--chip", chip, "-o"]).arg(&ptx);
        for (name, value) in &pins {
            cmd.args(["--set", &format!("{name}={value}")]);
        }
        let out = cmd.output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

        let runtime = Context {
            gpu_config: GpuConfig::Nvidia(NvidiaGpuConfig::with_chip(chip)),
            shape_overrides: pins.iter().map(|&(n, v)| (n.to_string(), v)).collect::<HashMap<_, _>>(),
            ..Context::default()
        };
        let (cached, _) = phobos_kernels::cache::load(&runtime, &text).expect("the runtime's lookup misses");
        assert_eq!(cached, std::fs::read_to_string(&ptx).unwrap());
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(ptx.with_extension("json")).unwrap()).unwrap();
        assert_eq!(json["cache_key"], phobos_kernels::cache::key(&runtime, &text));
        assert_eq!(json["compiler_fingerprint"], phobos_kernels::COMPILER_FINGERPRINT);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
