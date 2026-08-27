// The Phobos source every device kernel is compiled from, and the tile
// sizes that shape it. One module per op family.
//
// Nothing here touches the driver: these are `&str` and `String` builders,
// so a change is visible with `cargo run -p phobos-lang --example emit`
// without a GPU in the machine.

mod argmax;
mod attn;
mod delta;
mod elem;
mod iq1m;
mod iq1s;
mod iq2s;
mod iq2xs;
mod iq2xxs;
mod iq3s;
mod iq3xxs;
mod iq4xs;
mod matmul;
mod norm;
mod q2k;
mod q3k;
mod quant;

pub(crate) use argmax::*;
pub(crate) use attn::*;
// The attention checks in `examples/` compile these themselves, so they stay
// reachable at `backend::device::` where they have always been.
pub use attn::{ATTN_GEMM_TILE, ATTN_SOFT_TILE, attn_gemm_src};
pub(crate) use delta::*;
pub(crate) use elem::*;
pub(crate) use iq1m::*;
pub(crate) use iq1s::*;
pub(crate) use iq2s::*;
pub(crate) use iq2xs::*;
pub(crate) use iq2xxs::*;
pub(crate) use iq3s::*;
pub(crate) use iq3xxs::*;
pub(crate) use iq4xs::*;
pub(crate) use matmul::*;
pub(crate) use norm::*;
pub(crate) use q2k::*;
pub(crate) use q3k::*;
pub(crate) use quant::*;

/// One dp4a decode matvec in the table that drives compilation: the source
/// builder, its wide and narrow output tiles, and the kernel's name.
pub(crate) type I8Row = (fn(usize) -> String, usize, usize, &'static str);

/// A built kernel source with the defines it was built for and its name, as
/// [`phobos_kernels::compile_parallel`] takes them.
pub(crate) type OwnedEntry = (String, [(&'static str, usize); 1], &'static str);

/// The borrowed form of the same thing.
pub(crate) type Entry<'a> = (&'a str, &'a [(&'static str, usize)], &'static str);
