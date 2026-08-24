// Tile operations, the bulk of the emitter. Every file here reopens `impl
// Codegen` and groups one kind of work; `contract.rs` dispatches across the
// quantized contraction files, which are separate hardware paths.

use super::*;

mod alloc;
mod check;
mod contract;
mod dp4a;
mod elem;
mod gather;
mod imma;
mod iq1m_qdot;
mod iq1s_qdot;
mod iq2s_qdot;
mod iq2xs_qdot;
mod iq2xxs_qdot;
mod iq3s_qdot;
mod iq3xxs_qdot;
mod iq4xs_qdot;
mod math;
mod q2k_qdot;
mod q3k_qdot;
mod qdot;
mod qmma;
mod reduce;
mod vector;
mod warp_attn;
