fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    // cust links the CUDA driver API itself; we only add cuBLAS.
    if let Ok(cuda) = std::env::var("CUDA_PATH") {
        // TODO(joa): windows only atm
        println!("cargo:rustc-link-search=native={cuda}\\lib\\x64");
    }
    println!("cargo:rustc-link-lib=cublas");
}
