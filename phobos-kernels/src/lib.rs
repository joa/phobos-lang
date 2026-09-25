/// The build's compiler fingerprint, which keys every cache entry: a release
/// ships a cache only binaries printing the same one can read.
pub const COMPILER_FINGERPRINT: &str = env!("PHOBOS_COMPILER_FINGERPRINT");

pub mod abi;
pub mod matmul;
pub mod util;

// The reading and recording halves of these serve `compile`, which needs the
// driver; a build without it, such as `phobos-cache`, only writes.
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
