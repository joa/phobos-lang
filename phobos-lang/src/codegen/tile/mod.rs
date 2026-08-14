// Tile operations, the bulk of the emitter.
//
// Every file here reopens `impl Codegen` and groups one kind of work. The
// four quantized contractions are separate because they are separate
// hardware paths, not because they are separate ideas: `contract.rs`
// dispatches to them.

use super::*;

mod alloc;
mod check;
mod contract;
mod dp4a;
mod elem;
mod imma;
mod math;
mod qdot;
mod qmma;
mod reduce;
mod vector;
