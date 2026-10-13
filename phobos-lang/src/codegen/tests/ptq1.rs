// PTQ1_0's staged projection. Base-three bytes are decoded by multiplying by
// three in 16-bit lanes and byte-permuting the tops.

use super::qgemm::qgemm_src;
use super::*;

/// The decode constants. `0x4140` and `0x4342` split a word into two lane
/// pairs, and `0x7531` picks the four tops. `0x00030001` and `0x001B0009`
/// spread a tail byte over four powers of three.
const DECODE: [&str; 5] = [
    "arith.constant 16704 : i32",
    "arith.constant 17218 : i32",
    "arith.constant 30001 : i32",
    "arith.constant 196609 : i32",
    "arith.constant 1769481 : i32",
];

#[test]
fn the_staged_projection_writes_signed_trits_and_no_minimum() {
    let mlir = emit_mlir(&qgemm_src("ptq1", &[], "256, 2"));
    assert_contains(&mlir, &DECODE);
    // Each word is written as `(t + 0x7F7F7F7F) ^ 0x80808080`.
    assert_contains(
        &mlir,
        &["arith.constant 2139062143 : i32", "arith.constant -2139062144 : i32", "arith.xori"],
    );
    // No minimum means no row sums, so no shuffle joins their halves.
    assert!(!mlir.contains("gpu.shuffle"), "{mlir}");
    assert!(mlir.contains("memref<4x64xf32, 3>"), "{mlir}");
}
