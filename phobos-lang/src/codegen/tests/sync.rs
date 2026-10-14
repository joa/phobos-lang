// `atomic_add` and `grid_barrier`.

use super::*;

#[test]
fn atomic_add_lowers_to_an_rmw() {
    let mlir = emit_mlir(
        "kernel tally(BAR: tensor<i32>[2], OUT: tensor<i32>[N]) {
            let i = ctx.block.x
            OUT[i] = atomic_add(BAR, 0, 1)
        }",
    );
    assert_contains(&mlir, &["memref.atomic_rmw", "addi"]);
}

#[test]
fn atomic_add_rejects_a_float_tensor() {
    let err = std::panic::catch_unwind(|| {
        emit_mlir(
            "kernel tally(BAR: tensor<f32>[2], OUT: tensor<i32>[N]) {
                let i = ctx.block.x
                OUT[i] = atomic_add(BAR, 0, 1)
            }",
        )
    });
    assert!(err.is_err(), "an f32 barrier tensor should not compile");
}

/// The two-phase barrier: a generation read, an arrival, a reset, a release
/// and a spin, between CTA barriers that pass the release to the rest of the
/// block. See codegen/sync.rs.
#[test]
fn grid_barrier_lowers_to_arrive_and_spin() {
    let mlir = emit_mlir(
        "kernel staged(X: tensor<f32>[N], BAR: tensor<i32>[2]) {
            let i = ctx.block.x
            X[i] = X[i] + 1.0
            grid_barrier(BAR)
            X[i] = X[i] * 2.0
        }",
    );
    assert_contains(
        &mlir,
        &[
            "memref.atomic_rmw",
            "scf.while",
            "gpu.barrier",
            "gpu.grid_dim",
        ],
    );
    // Five atomics: the generation read, the arrival, the counter reset, the
    // release, and the spin's read.
    assert_eq!(mlir.matches("memref.atomic_rmw").count(), 5);
    // A device-scope fence on each side of the arrival, so the barrier
    // orders the blocks' global writes as well as their progress.
    assert_eq!(mlir.matches("fence.acq_rel.gpu").count(), 2, "{mlir}");
}

/// The same barrier as a `[2, 1]` column, for a host whose launch ABI passes
/// rank-2 descriptors. The slot indexes the leading dimension, and the
/// atomic takes a second subscript that the verifier checks against the
/// memref rank.
#[test]
fn grid_barrier_takes_a_rank_two_column() {
    let mlir = emit_mlir(
        "kernel staged(X: tensor<f32>[N], BAR: tensor<i32>[2, 1]) {
            let i = ctx.block.x
            X[i] = X[i] + 1.0
            grid_barrier(BAR)
        }",
    );
    assert_contains(&mlir, &["memref.atomic_rmw", "scf.while"]);
    assert_eq!(mlir.matches("memref.atomic_rmw").count(), 5);
}

#[test]
fn grid_barrier_rejects_a_non_tensor() {
    let err = std::panic::catch_unwind(|| {
        emit_mlir(
            "kernel staged(X: tensor<f32>[N]) {
                var t: tile<i32>[1, 2] = 0
                grid_barrier(t)
            }",
        )
    });
    assert!(err.is_err(), "a tile barrier should not compile");
}

/// Two elementwise sweeps over one tile at one width touch the same
/// elements from the same thread, so they need no barrier between them.
#[test]
fn same_thread_sweeps_need_no_barrier_between_them() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N]) {
            var t = A[0 :+ 8, 0 :+ 8]
            t = t * 2.0
            t = t + 1.0
            C[0 :+ 8, 0 :+ 8] = t
        }",
    );
    // The staging copy moves 16 bytes per lane and the f32 sweeps four
    // elements, the same mapping. The store reads it the same way too.
    assert_eq!(mlir.matches("gpu.barrier").count(), 0, "{mlir}");
}

/// One barrier can order several earlier accesses at once, so a sweep that
/// clashes with more than one keeps only that barrier, whatever order it
/// names its operands in.
#[test]
fn one_kept_barrier_orders_every_clash() {
    let src = |add: &str| {
        format!(
            "@launch(256)
            @autotune(TN in [256], U in [8])
            @aligned(N = TN)
            kernel k(W: tensor<f32>[1, U], C: tensor<f32>[U, N], G: tensor<f32>[1, 1], S: tensor<f32>[1, N], X: tensor<f32>[1, N]) {{
                let pn = ctx.block.x
                var acc: tile<f32>[1, TN] = X[0 :+ 1, pn * TN :+ TN]
                for j in range(0, U, 1) {{
                    let w = W[0, j]
                    acc {add} C[j :+ 1, pn * TN :+ TN] * w
                }}
                var g: tile<f32>[1, 1] = G[0 :+ 1, 0 :+ 1]
                g = 1.0 / (1.0 + exp(0.0 - g))
                acc {add} S[0 :+ 1, pn * TN :+ TN] * g
                X[0 :+ 1, pn * TN :+ TN] = acc
            }}"
        )
    };
    let spelled = emit_mlir(&src("= acc +"));
    let compound = emit_mlir(&src("+="));
    // One barrier publishes `g` and the loop's sweep to the last update,
    // which reads at another width; one publishes that to the store.
    assert_eq!(spelled.matches("gpu.barrier").count(), 2, "{spelled}");
    assert_eq!(compound.matches("gpu.barrier").count(), 2, "{compound}");
}

/// A reduction reads every column of a row from one thread, so the sweep
/// that wrote the row has to publish it first.
#[test]
fn a_reduction_after_a_sweep_keeps_the_barrier() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, 1]) {
            var t = A[0 :+ 8, 0 :+ 8]
            t = t * 2.0
            var s: tile<f32>[8, 1] = rowsum(t)
            C[0 :+ 8, 0 :+ 1] = s
        }",
    );
    let barriers: Vec<usize> = mlir.match_indices("gpu.barrier").map(|(at, _)| at).collect();
    let sweep = mlir.find("arith.mulf").expect("the scaling sweep");
    let reduce = mlir.find("arith.addf").expect("the reduction");
    assert!(
        barriers.iter().any(|&b| sweep < b && b < reduce),
        "the scaling sweep must publish before the reduction reads:\n{mlir}"
    );
}

/// A sweep carried to the next iteration is read by the same threads again,
/// so the loop needs no barrier at all.
#[test]
fn a_loop_carried_sweep_needs_no_barrier() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N], n: i32) {
            var t = A[0 :+ 8, 0 :+ 8]
            for i in range(0, n) {
                t = t * 2.0
            }
            C[0 :+ 8, 0 :+ 8] = t
        }",
    );
    assert_eq!(mlir.matches("gpu.barrier").count(), 0, "{mlir}");
}

/// Each loop re-adds what was pending at its entry, in case it runs no
/// turn. Thirty in a row once doubled the pending list thirty times.
#[test]
fn many_loops_in_a_row_stay_cheap() {
    let loops = "for i in range(0, n) {
 t = t * 2.0
 }
".repeat(30);
    let mlir = emit_mlir(&format!(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N], n: i32) {{
            var t = A[0 :+ 8, 0 :+ 8]
            {loops}
            C[0 :+ 8, 0 :+ 8] = t
        }}"
    ));
    assert_eq!(mlir.matches("gpu.barrier").count(), 0, "{mlir}");
}

/// A loop body that reads its tile with a different mapping than it wrote
/// it must keep a barrier between iterations.
#[test]
fn a_loop_carried_hazard_keeps_a_barrier_in_the_body() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, 1], n: i32) {
            var t = A[0 :+ 8, 0 :+ 8]
            var s: tile<f32>[8, 1] = 0.0
            for i in range(0, n) {
                t = t * 2.0
                s = rowsum(t)
            }
            C[0 :+ 8, 0 :+ 1] = s
        }",
    );
    let body = mlir.find("scf.for").expect("the loop");
    let barriers = mlir.match_indices("gpu.barrier").filter(|(at, _)| *at > body).count();
    assert!(barriers >= 1, "{mlir}");
}

/// A per-thread element read of a tile that another thread's sweep wrote
/// needs the sweep's barrier. The per-thread store has no barrier of its
/// own, so the pass keeps the sweep's.
#[test]
fn an_element_read_after_a_sweep_keeps_the_barrier() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N]) {
            var t = A[0 :+ 8, 0 :+ 8]
            t = t * 2.0
            let x = t[0, 1]
            C[0, 0] = x
        }",
    );
    let sweep = mlir.find("arith.mulf").expect("the scaling sweep");
    let load = mlir.rfind("memref.load").expect("the element read");
    assert!(
        mlir.match_indices("gpu.barrier").any(|(b, _)| sweep < b && b < load),
        "the sweep must publish before another thread reads an element:\n{mlir}"
    );
}

/// A hazard that only crosses the iteration boundary. The body's last op
/// writes the tile with one mapping and its first op reads it with another.
/// The write's own trailing barrier is the only site between them, and the
/// walk only sees the conflict on its second pass over the body.
#[test]
fn a_hazard_across_the_iteration_boundary_keeps_the_last_ops_barrier() {
    let mlir = emit_mlir(
        "@aligned(M = 8, N = 8)
        kernel k(A: tensor<f32>[M, N], C: tensor<f32>[M, N], n: i32) {
            var t = A[0 :+ 8, 0 :+ 8]
            var u: tile<f32>[8, 8] = 0.0
            for i in range(0, n) {
                u = exp(t)
                t = t * 2.0
            }
            C[0 :+ 8, 0 :+ 8] = u
        }",
    );
    // From the body's first op on, expect two barriers. One after the read,
    // for the write later in the same iteration. One after the write, for
    // the read that opens the next. None for the store after the loop.
    let read = mlir.find("ex2.approx").expect("the exp sweep");
    let barriers: Vec<usize> = mlir
        .match_indices("gpu.barrier")
        .map(|(at, _)| at)
        .filter(|&at| at > read)
        .collect();
    assert_eq!(barriers.len(), 2, "{mlir}");
    let write = mlir.rfind("arith.mulf").expect("the scaling sweep");
    assert!(barriers[1] > write, "the write's own barrier must stay:\n{mlir}");
}

/// A tile staged each turn and read by lane loads. The reads come from other
/// threads than the copy that wrote them, and no tile op between the last
/// read and the next turn's copy ends in a barrier, so one goes ahead of the
/// copy.
#[test]
fn a_tile_restaged_after_lane_reads_waits_for_them() {
    let mlir = emit_mlir(
        "@autotune(K in [256])
        @launch(128)
        kernel k(W: tensor<f32>[N, K], x: tensor<f32>[1, K], y: tensor<f32>[1, N]) {
            let row = ctx.block.x * ctx.warps + ctx.warp
            var acc: f32 = 0.0
            for kt in range(0, K, ctx.lanes * 4) {
                var xs = x[0 :+ 1, kt :+ ctx.lanes * 4]
                let at = ctx.lane * 4
                if row < N {
                    acc += sum(W[row, kt + at :+ 4] * xs[0, at :+ 4])
                }
            }
            let total = sum<over ctx.lane>(acc)
            if ctx.lane == 0 && row < N {
                y[0, row] = total
            }
        }",
    );
    let body = mlir.find("affine.for").expect("the loop");
    let copy = body + mlir[body..].find("vector.store").expect("the copy into xs");
    assert!(
        mlir.match_indices("gpu.barrier").any(|(b, _)| body < b && b < copy),
        "the next turn's copy must wait for this turn's lane reads:\n{mlir}"
    );
}

/// A register tile reads its staged operand in the threads a one-wide sweep
/// gives each element, so restaging it next turn needs no barrier. The
/// reduction after the loop writes scratch over the operand's bytes from
/// other threads, so it waits for the last turn's reads.
#[test]
fn a_register_tile_orders_only_what_other_threads_touch() {
    let mlir = emit_mlir(
        "@launch(256)
        @registers(acc)
        kernel k(X: tensor<f32>[64, K], S: tensor<f32>[64, 1]) {
            var acc: tile<f32>[64, 16] = 0.0
            for kt in range(0, K, 16) {
                var x = X[0 :+ 64, kt :+ 16]
                acc += x * x
            }
            var s: tile<f32>[64, 1] = rowsum(acc)
            S[0 :+ 64, 0 :+ 1] = s
        }",
    );
    let body = mlir.find("iter_args").expect("the loop");
    let copy = body + mlir[body..].find("memref<64x16xf32, 3>").expect("the copy into x");
    assert!(
        !mlir.match_indices("gpu.barrier").any(|(b, _)| body < b && b < copy),
        "restaging for the same threads needs no barrier:\n{mlir}"
    );
    let end = body + mlir[body..].find("scf.yield").expect("the loop's end");
    let reduce = mlir.find("gpu.shuffle").expect("the reduction");
    assert!(
        mlir.match_indices("gpu.barrier").any(|(b, _)| end < b && b < reduce),
        "the reduction's scratch must wait for the last turn's reads:\n{mlir}"
    );
}

#[test]
fn an_atomic_runs_once_per_cta_and_broadcasts() {
    let mlir = emit_mlir(
        "kernel claim(L: tensor<i32>[1], OUT: tensor<i32>[N]) {
            let i = ctx.block.x
            OUT[i] = atomic_cas(L, 0, 0, i)
            OUT[0] = atomic_add(L, 0, 0)
        }",
    );
    assert_contains(&mlir, &["memref.generic_atomic_rmw", "memref.atomic_yield", "gpu.thread_id"]);
    // A barrier on each side of each atomic.
    assert_eq!(mlir.matches("gpu.barrier").count(), 4, "\n{mlir}");
    assert_eq!(mlir.matches("scf.if").count(), 2, "\n{mlir}");
}

#[test]
fn atomic_cas_takes_only_an_i32_tensor() {
    let e = emit_err(
        "kernel k(L: tensor<f32>[1]) {
            let old = atomic_cas(L, 0, 0, 1)
        }",
    );
    assert!(e.contains("atomic_cas expects an i32 tensor"), "got: {e}");
}

#[test]
fn a_claimed_slot_steers_uniform_control_flow() {
    // The CAS result is the same on every thread, so it may decide an `if`
    // that holds a barrier.
    let mlir = emit_mlir(
        "kernel k(L: tensor<i32>[1], X: tensor<f32>[16, 64]) {
            if atomic_cas(L, 0, 0, 1) == 0 {
                var x = X[0 :+ 16, 0 :+ 64]
                var m: tile<f32>[16, 1] = rowmax(x)
                X[0 :+ 16, 0 :+ 1] = m
            }
        }",
    );
    assert_contains(&mlir, &["memref.generic_atomic_rmw", "gpu.shuffle"]);
    // Every thread reads the claimed value from a shared slot after the
    // atomic's last barrier. The staged tile reuses those bytes, so a
    // barrier sits between the read and the copy.
    let read = mlir
        .match_indices("memref.load")
        .map(|(at, _)| at)
        .find(|&at| mlir[at..].lines().next().is_some_and(|l| l.contains("memref<1xi32, 3>")))
        .expect("the slot read");
    let copy = read + mlir[read..].find("memref<16x64xf32, 3>").expect("the staging copy");
    assert!(
        mlir.match_indices("gpu.barrier").any(|(b, _)| read < b && b < copy),
        "the copy must wait for every thread's read of the slot:\n{mlir}"
    );
}

/// A device-side wait on a flag the host raises, then a release that lowers
/// it: the expert hand-off's shape.
const HANDOFF: &str = "kernel wait(F: tensor<i32>[1, 1], Y: tensor<f32>[1, 64], D: tensor<f32>[1, 64]) {
    while load_acquire(F, 0) == 0 {
    }
    var y = Y[0 :+ 1, 0 :+ 64]
    var d = D[0 :+ 1, 0 :+ 64]
    D[0 :+ 1, 0 :+ 64] = d + y
    store_release(F, 0, 0)
}";

#[test]
fn system_scope_access_is_one_thread_and_a_fence() {
    let mlir = emit_mlir_sync(HANDOFF, "sm_75");
    assert_contains(
        &mlir,
        &["ld.acquire.sys.global.u32", "fence.acq_rel.sys", "st.release.sys.global.u32", "scf.while"],
    );
}

#[test]
fn system_scope_access_widens_the_index() {
    let kernels = crate::parse(HANDOFF).unwrap();
    assert!(crate::needs_wide_index(&phobos_base::context::Context::default(), &kernels));
    let e = emit_err(HANDOFF);
    assert!(e.contains("needs 64-bit indices"), "got: {e}");
}
