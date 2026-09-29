/// The build's compiler fingerprint. Every cache entry is keyed on it, so a
/// shipped cache is readable only by binaries with the same fingerprint.
pub const COMPILER_FINGERPRINT: &str = env!("PHOBOS_COMPILER_FINGERPRINT");

pub mod abi;
pub mod matmul;
pub mod util;

// Only `compile`, which needs the driver, reads and records through these.
// A build without it, such as `phobos-cache`, only writes.
#[cfg(feature = "compiler")]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod cache;
#[cfg(feature = "compiler")]
mod lower;
#[cfg(feature = "compiler")]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub mod manifest;

#[cfg(feature = "cuda")]
pub mod compile;
#[cfg(feature = "cuda")]
pub mod launch;
#[cfg(feature = "cuda")]
pub mod pool;

#[cfg(feature = "cuda")]
pub use compile::{Variants, compile, compile_in, compile_parallel, compile_shared};
#[cfg(feature = "cuda")]
pub use launch::{cuda_ok, push_descriptor};
#[cfg(feature = "cuda")]
pub use pool::Pool;
