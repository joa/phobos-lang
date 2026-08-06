fn main() {
    // SAFETY: single-threaded build script, set before any threads spawn.
    unsafe {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path().unwrap());
    }

    let proto = "proto/onnx.proto";
    println!("cargo:rerun-if-changed={proto}");

    // ONNX carries no RPC services, so we only need the prost message types.
    tonic_build::configure()
        .build_server(false)
        .build_client(false)
        .compile_protos(&[proto], &["proto"])
        .expect("compiling onnx.proto");
}
