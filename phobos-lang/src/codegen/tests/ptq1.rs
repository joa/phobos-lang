// PTQ1_0's decode matvec and staged projection: base-three bytes decoded
// by a multiply by three in 16-bit lanes and a byte permute of the tops.

use super::qgemm::{qdot_i8_src, qgemm_src};
use super::*;

/// `0x4140` and `0x4342` split a word into two lane pairs, `0x7531` picks
/// the four tops, and `0x00030001` and `0x001B0009` spread a tail byte over
/// four powers of three.
const DECODE: [&str; 5] = [
    "arith.constant 16704 : i32",
    "arith.constant 17218 : i32",
    "arith.constant 30001 : i32",
    "arith.constant 196609 : i32",
    "arith.constant 1769481 : i32",
];

#[test]
fn the_decode_matvec_contracts_trits_and_subtracts_the_activation_sum() {
    let mlir = emit_mlir(&qdot_i8_src("ptq1"));
    assert_contains(&mlir, &["nvvm.prmt", "nvvm.dot.accumulate.4way", "gpu.shuffle", "scf.for"]);
    assert_contains(&mlir, &DECODE);
    // The -1 bytes the second dp4a of each word runs against.
    assert_contains(&mlir, &["arith.constant -1 : i32"]);
    // No run-sum plane: the -1 dp4a folds the offset in the lane.
    assert!(!mlir.contains("xi32, 3>"), "{mlir}");
}

#[test]
fn the_staged_projection_writes_signed_trits_and_no_minimum() {
    let mlir = emit_mlir(&qgemm_src("ptq1", &[], "256, 2"));
    assert_contains(&mlir, &DECODE);
    // `(t + 0x7F7F7F7F) ^ 0x80808080` a word.
    assert_contains(
        &mlir,
        &["arith.constant 2139062143 : i32", "arith.constant -2139062144 : i32", "arith.xori"],
    );
    // No minimum: no row sums and so no shuffle joining their halves.
    assert!(!mlir.contains("gpu.shuffle"), "{mlir}");
    assert!(mlir.contains("memref<4x64xf32, 3>"), "{mlir}");
}
