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
mod matmul;
mod norm;
mod quant;

pub(crate) use argmax::*;
pub(crate) use attn::*;
// The attention checks in `examples/` compile these themselves, so they stay
// reachable at `backend::device::` where they have always been.
pub use attn::{ATTN_GEMM_TILE, ATTN_SOFT_TILE, attn_gemm_src};
pub(crate) use delta::*;
pub(crate) use elem::*;
pub(crate) use matmul::*;
pub(crate) use norm::*;
pub(crate) use quant::*;
