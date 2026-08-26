// `<fmt>_qdecode_t`: see phobos-lang/src/codegen/tile/qdecode.rs.

use super::*;

const SRC: &str = "\
@launch(256)
@aligned(N = 8)
kernel iq1s_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                    GRID: tensor<i8>[1, 16384], SCRATCH: tensor<f32>[K, N]) {
  let pn = program_id(0)
  SCRATCH[:, pn * 8 :+ 8] = iq1s_qdecode_t(QB[pn * 8 :+ 8, :], D[pn * 8 :+ 8, :],
                                           GRID[0 :+ 1, :])
}";

const XXS_SRC: &str = "\
@launch(256)
@aligned(N = 8)
kernel iq2xxs_qdecode(QB: tensor<i8>[N, RB], D: tensor<f16>[N, NB],
                      GRID: tensor<i8>[1, 2048], SIGNS: tensor<i8>[1, 1024],
                      SCRATCH: tensor<f32>[K, N]) {
  let pn = program_id(0)
  SCRATCH[:, pn * 8 :+ 8] = iq2xxs_qdecode_t(QB[pn * 8 :+ 8, :], D[pn * 8 :+ 8, :],
                                             GRID[0 :+ 1, :], SIGNS[0 :+ 1, :])
}";

/// Every format with an expansion, by name and table count. `XXS_SRC`'s body
/// is the two-table shape and `SRC`'s the one-table shape, so a format's own
/// source is either of those with its name substituted in.
const FORMATS: [(&str, usize); 7] = [
    ("iq1s", 1),
    ("iq1m", 1),
    ("iq2xxs", 2),
    ("iq2s", 2),
    ("iq2xs", 2),
    ("iq3xxs", 2),
    ("iq3s", 2),
];

/// A format's kernel. The table lengths do not have to be the format's own:
/// the intrinsic indexes its tables itself, and only their element type is
/// checked.
fn src_for(name: &str, tables: usize) -> String {
    if tables == 1 {
        SRC.replace("iq1s", name)
    } else {
        XXS_SRC.replace("iq2xxs", name)
    }
}

/// The whole point of the intrinsic: no shared buffer, no barrier, and the
/// eight decoded weights of a lane going straight to the scratch.
#[test]
fn qdecode_t_writes_the_scratch_without_staging() {
    for (what, tables) in FORMATS {
        let mlir = emit_mlir(&src_for(what, tables));
        assert_eq!(
            mlir.matches("memref.global").count(),
            0,
            "{what} should stage nothing:\n{mlir}"
        );
        assert_eq!(
            mlir.matches("gpu.barrier").count(),
            0,
            "{what} threads are independent, so nothing should synchronize:\n{mlir}"
        );
        assert_eq!(
            mlir.matches("memref.store").count(),
            8,
            "{what} should store a lane's eight elements and nothing else:\n{mlir}"
        );
    }
}

/// The packed tables are read a whole entry at a time: one `vector.load` for
/// a one-table format, two for a format that keeps its signs separate. That
/// is the point of packing them to `i8` -- eight scalar loads an element
/// group become one, and the table is a quarter the size.
#[test]
fn qdecode_t_reads_a_whole_table_entry_at_once() {
    for (what, tables) in FORMATS {
        let mlir = emit_mlir(&src_for(what, tables));
        assert_eq!(
            mlir.matches("vector.load").count(),
            if what.starts_with("iq3") { tables + 1 } else { tables },
            "{what} should load each table entry once:
{mlir}"
        );
    }
}

/// The store has to coalesce, which is what picks the thread map: the column
/// is the fast axis, so a warp covers whole sectors of a row band.
#[test]
fn qdecode_t_makes_the_column_the_fast_axis() {
    let mlir = emit_mlir(SRC);
    let lane = mlir
        .lines()
        .find(|l| l.contains("arith.divui") && l.contains("index"))
        .expect("a lane index divided out of the thread id");
    let col = mlir
        .lines()
        .find(|l| l.contains("arith.remui") && l.contains("index"))
        .expect("a column index taken modulo the tile width");
    let width = "%c8";
    assert!(
        lane.contains(width) && col.contains(width),
        "both halves of the thread map should divide the tile width:\n{lane}\n{col}"
    );
}

/// The destination's element type picks the width: an f16 scratch halves the
/// traffic to the matmul that reads it, and costs nothing on the tensor-core
/// ladder, which truncates its weight operand to f16 anyway.
#[test]
fn qdecode_t_narrows_to_an_f16_destination() {
    let f32_mlir = emit_mlir(SRC);
    assert_eq!(
        f32_mlir.matches("arith.truncf").count(),
        0,
        "an f32 scratch should store the decoded value as it is:
{f32_mlir}"
    );
    let f16 = SRC.replace("SCRATCH: tensor<f32>[K, N]", "SCRATCH: tensor<f16>[K, N]");
    let mlir = emit_mlir(&f16);
    assert_eq!(
        mlir.matches("arith.truncf").count(),
        8,
        "an f16 scratch should narrow each of a lane's eight weights:
{mlir}"
    );
}

#[test]
fn qdecode_t_rejects_a_tile_destination() {
    let src = "\
@launch(256)
kernel iq1s_qdecode(QB: tensor<i8>[8, 50], D: tensor<f16>[8, 1],
                    GRID: tensor<i8>[1, 16384], SCRATCH: tensor<f32>[256, 8]) {
  var t: tile<f32>[256, 8] = 0.0
  t = iq1s_qdecode_t(QB[0 :+ 8, :], D[0 :+ 8, :], GRID[0 :+ 1, :])
  SCRATCH[0 :+ 256, 0 :+ 8] = t
}";
    let err = std::panic::catch_unwind(|| emit_mlir(src));
    assert!(err.is_err(), "a shared tile destination should be rejected");
}

#[test]
fn qdecode_t_rejects_use_as_a_value() {
    let src = SRC.replace(
        "SCRATCH[:, pn * 8 :+ 8] = iq1s_qdecode_t(",
        "SCRATCH[:, pn * 8 :+ 8] = 0.0 + iq1s_qdecode_t(",
    );
    let err = std::panic::catch_unwind(|| emit_mlir(&src));
    assert!(err.is_err(), "the intrinsic has no value form");
}

#[test]
fn qdecode_t_rejects_a_missing_table() {
    let src = XXS_SRC.replace(", SIGNS[0 :+ 1, :])", ")");
    let err = std::panic::catch_unwind(|| emit_mlir(&src));
    assert!(err.is_err(), "IQ2_XXS needs its sign table");
}
