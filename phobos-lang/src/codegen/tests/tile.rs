// Tile declaration, the buffer pool, and `flat` views.

use super::*;

#[test]
fn a_dead_named_tile_returns_its_buffer_to_the_pool() {
    // A named tile used to hold its shared buffer for the whole kernel,
    // because bind cannot see where the name stops being read. A block's
    // own declarations are the case where it can: the last statement
    // mentioning the name ends its life, so the next allocation in the
    // block reuses the buffer instead of minting another global.
    let mlir = emit_mlir(
        "@launch(256)
        kernel chain(X: tensor<f32>[R, N], O: tensor<f32>[R, N]) {
            var a: tile<f32>[16, 64] = X[0 :+ 16, 0 :+ 64]
            var b: tile<f32>[16, 64] = a + 1.0
            var c: tile<f32>[16, 64] = b * 2.0
            var d: tile<f32>[16, 64] = c - 3.0
            O[0 :+ 16, 0 :+ 64] = d
        }",
    );
    let buffers = mlir.matches("memref.global").count();
    assert!(
        buffers <= 2,
        "four dead-on-arrival tiles want at most two buffers, got {buffers}:
{mlir}"
    );
}

#[test]
fn dynamic_shared_resets_its_cursor_between_dead_phases() {
    // `@dynshared` tiles live at byte offsets in one allocation rather than
    // one `memref.global` apiece, so two distinct shapes used to just sum:
    // this is the bug behind attention_persist_src's 41KB combined
    // footprint before the fix, since its two barrier-separated phases
    // never share a shape. `a` ([16, 64], 4096 bytes) and `b` ([8, 4], 128
    // bytes) are both read and therefore released before `c` ([1, 8], a
    // shape neither of them is) is ever declared, so nothing is live when
    // `c` mints: the allocator should restart at offset 0 rather than
    // append past `a` and `b`'s combined 4224 bytes.
    let mlir = emit_mlir(
        "@dynshared
        kernel chain(X: tensor<f32>[16, 64], Y: tensor<f32>[8, 4],
                     O: tensor<f32>[16, 64], Q: tensor<f32>[8, 4],
                     P: tensor<f32>[1, 8]) {
            var a: tile<f32>[16, 64] = X[0 :+ 16, 0 :+ 64]
            O[0 :+ 16, 0 :+ 64] = a
            var b: tile<f32>[8, 4] = Y[0 :+ 8, 0 :+ 4]
            Q[0 :+ 8, 0 :+ 4] = b
            var c: tile<f32>[1, 8] = P[0 :+ 1, 0 :+ 8]
            P[0 :+ 1, 0 :+ 8] = c
        }",
    );
    assert!(
        !mlir.contains("c4224"),
        "c minted past a and b's combined footprint instead of reusing it, \
         the allocator did not reset when nothing was live:\n{mlir}"
    );
}

#[test]
fn a_tile_read_after_a_loop_keeps_its_buffer() {
    // The last mention is what ends a name's life, and a nested body is
    // part of the statement that contains it: `keep` is read inside the
    // loop and again after it, so neither point may release it.
    let mlir = emit_mlir(
        "@launch(256)
        kernel later(X: tensor<f32>[R, N], O: tensor<f32>[R, N]) {
            var keep: tile<f32>[16, 64] = X[0 :+ 16, 0 :+ 64]
            var acc: tile<f32>[16, 64] = 0.0
            for i in range(0, 4, 1) {
                acc = acc + keep
            }
            O[0 :+ 16, 0 :+ 64] = acc + keep
        }",
    );
    assert!(module_verifies(&mlir), "{mlir}");
}

/// A tile is shared memory, so a slice of one has to name that address
/// space, which is why the space travels on the [`MemVal`].
#[test]
fn a_tile_slice_stays_in_shared_memory() {
    let mlir = emit_mlir(
        "kernel fill(X: tensor<f32>[32, 32], O: tensor<f32>[32, 32]) {
            var A: tile<f32>[32, 32] = 0.0
            for b in range(0, 32, 16) {
                A[b :+ 16, 0 :+ 32] = X[b :+ 16, 0 :+ 32] * 2.0
            }
            O[0 :+ 32, 0 :+ 32] = A
        }",
    );
    assert_contains(
        &mlir,
        &[
            // the tile's own slice, in shared
            "memref<16x32xf32, strided<[32, 1], offset: ?>, 3>",
            // the tensor's, in global, from the same code path
            "memref<16x32xf32, strided<[32, 1], offset: ?>, 1>",
        ],
    );
}

/// Declaring a buffer a following loop fills entirely should not emit the
/// fill an initializer would, since every element of it is overwritten.
#[test]
fn an_uninitialized_tile_emits_no_fill() {
    let filled = emit_mlir(
        "kernel decl(X: tensor<f32>[16, 32], O: tensor<f32>[16, 32]) {
            var A: tile<f32>[16, 32] = 0.0
            A[0 :+ 16, 0 :+ 32] = X[0 :+ 16, 0 :+ 32]
            O[0 :+ 16, 0 :+ 32] = A
        }",
    );
    let bare = emit_mlir(
        "kernel decl(X: tensor<f32>[16, 32], O: tensor<f32>[16, 32]) {
            var A: tile<f32>[16, 32]
            A[0 :+ 16, 0 :+ 32] = X[0 :+ 16, 0 :+ 32]
            O[0 :+ 16, 0 :+ 32] = A
        }",
    );
    // Same buffers either way, one fewer sweep over one of them.
    assert_eq!(
        filled.matches("memref.global").count(),
        bare.matches("memref.global").count()
    );
    assert!(
        bare.matches("scf.for").count() < filled.matches("scf.for").count(),
        "the fill loop should be gone"
    );
}

/// `var` without an initializer needs the type, since nothing is left to
/// infer one from, and `let` cannot omit it at all: it would name nothing.
#[test]
fn an_uninitialized_declaration_needs_a_tile_type() {
    for src in [
        "kernel decl(O: tensor<f32>[4]) { var a\n O[0] = 1.0 }",
        "kernel decl(O: tensor<f32>[4]) { var a: f32\n O[0] = a }",
        "kernel decl(O: tensor<f32>[4]) { let a: f32\n O[0] = a }",
    ] {
        let err = std::panic::catch_unwind(|| emit_mlir(src));
        assert!(err.is_err(), "should not compile: {src}");
    }
}

/// The flat view is the same bytes under another type, still in shared
/// memory: what lets a value reduced per block of 32 be contracted over as
/// one row.
#[test]
fn flat_views_a_tile_as_one_row() {
    let mlir = emit_mlir(
        "kernel quant(X: tensor<f32>[32, 32],
                      Wq: tensor<i8>[32, 1024],
                      Ws: tensor<f32>[32, 32],
                      O: tensor<f32>[1, 32]) {
            var Aq: tile<i8>[32, 32]
            var As: tile<f32>[32, 1]
            for b in range(0, 32, 16) {
                var y: tile<f32>[16, 32] = X[b :+ 16, 0 :+ 32]
                var mx: tile<f32>[16, 1] = rowmax(tmax(y, -y))
                var q = y * (127.0 / (mx + 0.00000001))
                Aq[b :+ 16, 0 :+ 32] = i8(i32(round(q)))
                As[b :+ 16, 0 :+ 1] = mx / 127.0
            }
            O[0 :+ 1, 0 :+ 32] = qdot_t(flat(Aq), flat(As),
                                        Wq[0 :+ 32, :], Ws[0 :+ 32, :])
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref.reinterpret_cast",
            // the quantized row and its scales, both one row, both shared
            "memref<32x32xi8, 3> to memref<1x1024xi8, 3>",
            "memref<32x1xf32, 3> to memref<1x32xf32, 3>",
            // and the contraction reads them there rather than from global
            "vector<4xi8>",
        ],
    );
}

/// A flattened tile's buffer must leave the pool: the view is bound to a
/// name of its own, so the second declaration below would otherwise be
/// handed the bytes the view still reads, and compile to a wrong answer.
#[test]
fn a_flattened_tile_is_not_recycled() {
    let mlir = emit_mlir(
        "kernel alias(X: tensor<f32>[8, 4], O: tensor<f32>[1, 32], P: tensor<f32>[8, 4]) {
            var A: tile<f32>[8, 4] = 0.0
            A[0 :+ 8, 0 :+ 4] = X[0 :+ 8, 0 :+ 4]
            let flatA = flat(A)
            var B: tile<f32>[8, 4] = 1.0
            P[0 :+ 8, 0 :+ 4] = B
            O[0 :+ 1, 0 :+ 32] = flatA
        }",
    );
    // Two declarations of the same shape, and so two buffers rather than one
    // reused: the reuse is what would corrupt the view.
    assert_eq!(
        mlir.matches("memref<8x4xf32, 3> = uninitialized").count(),
        2,
        "the flattened buffer was handed out again"
    );
}

/// A view is not a buffer, so it cannot be flattened again, and neither can
/// the staging tiles whose rows are not where row-major says they are.
#[test]
fn flat_rejects_what_is_not_a_declared_tile() {
    for src in [
        // a slice of a tile, whose offset the cast would drop
        "kernel v(X: tensor<f32>[32, 32], O: tensor<f32>[1, 32]) {
            var A: tile<f32>[32, 32] = 0.0
            let s = A[0 :+ 16, 0 :+ 32]
            O[0 :+ 1, 0 :+ 32] = flat(s)
        }",
        // a tensor slice, which is not in shared memory at all
        "kernel v(X: tensor<f32>[32, 32], O: tensor<f32>[1, 32]) {
            O[0 :+ 1, 0 :+ 32] = flat(X[0 :+ 16, 0 :+ 32])
        }",
    ] {
        let err = std::panic::catch_unwind(|| emit_mlir(src));
        assert!(err.is_err(), "should not compile: {src}");
    }
}

/// `warp_partial`'s K/V loads at D = 128 (dpl = 4) come out as one 4xf16
/// vector load per key per lane, not four scalar ones.
#[test]
fn warp_partial_vectorizes_kv_loads_at_dpl_4() {
    let mlir = emit_mlir(&warp_partial_probe(128, 8, 8));
    assert_contains(&mlir, &["vector.load", "vector<4xf16>", "alignment = 8"]);
}

/// The same at D = 256 (dpl = 8): a whole lane's slice in one 16-byte load.
#[test]
fn warp_partial_vectorizes_kv_loads_at_dpl_8() {
    let mlir = emit_mlir(&warp_partial_probe(256, 8, 8));
    assert_contains(&mlir, &["vector.load", "vector<8xf16>", "alignment = 16"]);
}

/// `@padstage` routes a staging tile whose row pitch is a bank-period
/// multiple through `alloc_tile_padded`. See `Codegen::should_pad_stage`.
#[test]
fn padstage_pads_a_bank_period_pitch_tile() {
    let mlir = emit_mlir(
        "@padstage
        kernel stage(K: tensor<f16>[64, 128], O: tensor<f16>[8, 128]) {
            var k = K[0 :+ 8, 0 :+ 128]
            O[0 :+ 8, 0 :+ 128] = k
        }",
    );
    assert_contains(&mlir, &["memref<8x136xf16, 3>"]);
}

/// The same statement without `@padstage` stays unpadded: opt-in per kernel,
/// not a default.
#[test]
fn without_padstage_the_same_tile_stays_unpadded() {
    let mlir = emit_mlir(
        "kernel stage(K: tensor<f16>[64, 128], O: tensor<f16>[8, 128]) {
            var k = K[0 :+ 8, 0 :+ 128]
            O[0 :+ 8, 0 :+ 128] = k
        }",
    );
    assert_contains(&mlir, &["memref<8x128xf16, 3>"]);
    assert!(!mlir.contains("136"), "should not have padded:\n{mlir}");
}

/// A pitch under the bank period (32 f16 elements, 64 bytes/row) is left
/// alone even with the attribute on.
#[test]
fn padstage_leaves_a_sub_period_pitch_tile_alone() {
    let mlir = emit_mlir(
        "@padstage
        kernel stage(K: tensor<f16>[64, 32], O: tensor<f16>[8, 32]) {
            var k = K[0 :+ 8, 0 :+ 32]
            O[0 :+ 8, 0 :+ 32] = k
        }",
    );
    assert_contains(&mlir, &["memref<8x32xf16, 3>"]);
}

/// Two consecutive staging statements read only global memory, so neither's
/// write can race the other and the first's trailing barrier is redundant:
/// one barrier for the pair, plus the final store's own.
#[test]
fn consecutive_staged_slices_share_one_barrier() {
    let mlir = emit_mlir(
        "kernel stage2(A: tensor<f16>[8, 128], B: tensor<f16>[8, 128], O: tensor<f16>[8, 128]) {
            var a = A[0 :+ 8, 0 :+ 128]
            var b = B[0 :+ 8, 0 :+ 128]
            O[0 :+ 8, 0 :+ 128] = a + b
        }",
    );
    let barriers = mlir.matches("gpu.barrier").count();
    assert_eq!(
        barriers, 2,
        "expected one barrier for the merged a/b pair plus one for the \
         store, got {barriers}:\n{mlir}"
    );
    let a_pos = mlir.find("__stage2_tile0").expect("a's tile");
    let b_pos = mlir.find("__stage2_tile1").expect("b's tile");
    let first_barrier = mlir.find("gpu.barrier").expect("a barrier");
    assert!(
        a_pos < b_pos && b_pos < first_barrier,
        "the barrier must land after both copies, not between them:\n{mlir}"
    );
}

/// A lone staging statement keeps its own barrier: the merge needs a run of
/// two or more.
#[test]
fn a_lone_staged_slice_keeps_its_own_barrier() {
    let mlir = emit_mlir(
        "kernel stage1(A: tensor<f16>[8, 128], O: tensor<f16>[8, 128]) {
            var a = A[0 :+ 8, 0 :+ 128]
            O[0 :+ 8, 0 :+ 128] = a
        }",
    );
    let barriers = mlir.matches("gpu.barrier").count();
    assert_eq!(
        barriers, 2,
        "a lone staging copy plus the store should keep two barriers, got \
         {barriers}:\n{mlir}"
    );
}

/// A statement that reads a just-staged tile is not a bare tensor slice, so
/// `stage_run` stops before it rather than merging away a barrier it needs.
#[test]
fn staging_run_stops_before_a_non_slice_statement() {
    let mlir = emit_mlir(
        "kernel stagec(A: tensor<f16>[8, 128], B: tensor<f16>[8, 128], O: tensor<f16>[8, 128]) {
            var a = A[0 :+ 8, 0 :+ 128]
            var b = B[0 :+ 8, 0 :+ 128]
            var c = a
            O[0 :+ 8, 0 :+ 128] = c + b
        }",
    );
    let barriers = mlir.matches("gpu.barrier").count();
    assert_eq!(
        barriers, 3,
        "a and b still merge (1), but c is not a bare tensor slice and \
         starts a new run (1), plus the store (1), got {barriers}:\n{mlir}"
    );
}

/// A minimal kernel calling `warp_partial` the way `attention_split_src`
/// does: one program, one warp group of query rows, the whole cache as
/// `[lo, hi)`.
fn warp_partial_probe(d: i64, wct: i64, qw: i64) -> String {
    format!(
        "@autotune(D in [{d}], WCT in [{wct}], QG in [1], QW in [{qw}])
        @launch(256)
        @aligned(KW = D)
        kernel wp(Q: tensor<f32>[R, D], K: tensor<f16>[NK, KW], V: tensor<f16>[NK, KW],
                  O: tensor<f32>[QW, D]) {{
            var q = Q[0 :+ QG, 0 :+ D]
            var wm: tile<f32>[QG, WCT] = -300000000.0
            var wl: tile<f32>[QG, WCT] = 0.0
            var wacc: tile<f32>[QW, D] = 0.0
            warp_partial(q, K, V, 0, NK, 0, wm, wl, wacc, 0.088388347648)
            O[0 :+ QW, 0 :+ D] = wacc
        }}"
    )
}

