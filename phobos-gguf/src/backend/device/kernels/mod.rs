// The Phobos source every device kernel is compiled from, and the tile
// sizes that shape it. One module per op family.
//
// Nothing here touches the driver: these are `&str` and `String` builders,
// so a change is visible with `cargo run -p phobos-lang --example emit`
// without a GPU in the machine.

mod argmax;
mod attn;
mod delta;
#[cfg(test)]
mod dump;
mod elem;
mod f32_lanes;
mod flash;
mod hadamard;
mod iq1m;
mod iq_f32;
mod iq_lanes;
mod iq1s;
mod iq2s;
mod iq2xs;
mod iq2xxs;
mod iq3s;
mod iq3xxs;
mod iq4xs;
mod kquant;
mod matmul;
mod moe;
mod norm;
mod q2k;
mod q3k;
mod qgemm;
mod qgemm_ph;
mod quant;

pub(crate) use argmax::*;
pub(crate) use attn::*;
pub(crate) use flash::*;
// The attention checks in `examples/` compile these themselves, so they stay
// reachable at `backend::device::`.
pub use attn::{ATTN_GEMM_TILE, ATTN_SOFT_TILE, attn_gemm_src};
pub use kquant::q4k_qdot_lanes_src;
pub use qgemm::{QGEMM_TM, QGEMM_TN};
pub use moe::{moe_gateup_src, moe_qdot_src, moe_topk_src};
pub(crate) use delta::*;
pub(crate) use elem::*;
pub(crate) use hadamard::*;
pub(crate) use iq1m::*;
pub(crate) use f32_lanes::*;
pub(crate) use iq_lanes::*;
use iq_f32::{IqFmt, qdecode_src, qdot_matvec_src};
pub(crate) use qgemm::*;
pub(crate) use qgemm_ph::{Split, qgemm_ph_src};
pub(crate) use iq1s::*;
pub(crate) use iq2s::*;
pub(crate) use iq2xs::*;
pub(crate) use iq2xxs::*;
pub(crate) use iq3s::*;
pub(crate) use iq3xxs::*;
pub(crate) use iq4xs::*;
pub(crate) use kquant::*;
pub(crate) use matmul::*;
pub(crate) use moe::*;
pub(crate) use norm::*;
pub(crate) use q2k::*;
pub(crate) use q3k::*;
pub(crate) use quant::*;

/// A kernel source with the defines it is built for and its name, as
/// [`phobos_kernels::compile_parallel`] takes them.
pub(crate) type Entry<'a> = (&'a str, &'a [(&'static str, usize)], &'static str);
