fn main() {
    // SAFETY: single-threaded build script, set before any threads spawn.
    unsafe {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
    }

    let protos = [
        "proto/v1/phobos_v1_isa.proto",
        "proto/v1/phobos_v1_scheduler.proto",
        "proto/v1/phobos_v1_tileserver.proto",
    ];
    for p in &protos {
        println!("cargo:rerun-if-changed={p}");
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&protos, &["proto"])
        .expect("compiling phobos protos");
}
