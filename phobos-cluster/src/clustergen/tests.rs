use phobos_lang::ast::{AssignOp, BinOp, Dim, Expr, Kernel, Stmt};

use super::compile;
use crate::ir::{ClusterStmt, Coord, SuperTile};
use crate::tile::AccessMode;

const MATMUL: &str = r#"
@cluster(TILE_M in [4096, 16384], TILE_N in [4096, 16384], TILE_K in [4096, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N]) {
let pm = program_id(0)
let pn = program_id(1)
var acc: tile<f32>[TILE_M, TILE_N] = 0.0
for kt in range(0, K, TILE_K) {
    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
}
C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}"#;

const ADD: &str = r#"
@cluster(BLOCK in [1048576, 16777216])
@autotune(BLOCK in [16, 4096])
kernel add(a: tensor<f32>[N], b: tensor<f32>[N], c: tensor<f32>[N]) {
let base = program_id(0) * BLOCK
c[base :+ BLOCK] = a[base :+ BLOCK] + b[base :+ BLOCK]
}"#;

fn first(src: &str) -> Kernel {
    phobos_lang::parse(src).unwrap().remove(0)
}

/// Mirrors phobos-lang's emit_mlir test harness: device-compile a leaf
/// and run the MLIR verifier on the emitted module.
fn verify_leaf(k: &Kernel) -> String {
    use melior::{
        Context,
        dialect::DialectRegistry,
        ir::{Location, Module, operation::OperationLike},
        utility::register_all_dialects,
    };
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    let module = Module::new(Location::unknown(&context));
    let base = phobos_base::context::Context::default();
    phobos_lang::codegen::emit(&base, std::slice::from_ref(k), &context, &module).unwrap();
    let text = module.as_operation().to_string();
    assert!(module.as_operation().verify(), "invalid module:\n{text}");
    text
}

#[test]
fn matmul_cluster_ir() {
    let p = compile(&first(MATMUL)).unwrap();

    assert_eq!(p.name, "matmul");
    assert_eq!(p.super_dims.len(), 3);
    // two-value search dims expand to doubling choices
    assert_eq!(p.super_dims[0].choices, vec![4096, 8192, 16384]);

    // grid: (M / TILE_M, N / TILE_N)
    assert_eq!(p.grid.len(), 2);
    assert_eq!(p.grid[0].dim, Dim::Sym("M".into()));
    assert_eq!(p.grid[0].super_sym, "TILE_M");
    assert_eq!(p.grid[1].dim, Dim::Sym("N".into()));
    assert_eq!(p.grid[1].super_sym, "TILE_N");

    // tensors: A read, B read, C pure output (init leaf overwrites)
    let modes: Vec<_> = p.tensors.iter().map(|t| t.mode).collect();
    assert_eq!(
        modes,
        vec![AccessMode::Read, AccessMode::Read, AccessMode::Write]
    );
    assert_eq!(p.tensors[0].super_syms, vec!["TILE_M", "TILE_K"]);
    assert_eq!(p.tensors[1].super_syms, vec!["TILE_K", "TILE_N"]);
    assert_eq!(p.tensors[2].super_syms, vec!["TILE_M", "TILE_N"]);

    // body: init compute, then the kt chain
    assert_eq!(p.body.len(), 2);
    let ClusterStmt::Compute { leaf: 1, args, .. } = &p.body[0] else {
        panic!("expected init compute, got {:?}", p.body[0]);
    };
    assert_eq!(
        args[0],
        (
            SuperTile {
                tensor: 2,
                coords: vec![Coord::Grid(0), Coord::Grid(1)]
            },
            AccessMode::Write
        )
    );
    let ClusterStmt::Loop {
        var,
        dim,
        super_sym,
        body,
    } = &p.body[1]
    else {
        panic!("expected cluster loop, got {:?}", p.body[1]);
    };
    assert_eq!(var, "kt");
    assert_eq!(*dim, Dim::Sym("K".into()));
    assert_eq!(super_sym, "TILE_K");

    // chain step: A(pm, kt) read, B(kt, pn) read, C(pm, pn) rmw (acc elided)
    let ClusterStmt::Compute { leaf: 0, args, .. } = &body[0] else {
        panic!("expected step compute, got {:?}", body[0]);
    };
    assert_eq!(
        args[0],
        (
            SuperTile {
                tensor: 0,
                coords: vec![Coord::Grid(0), Coord::Loop("kt".into())]
            },
            AccessMode::Read
        )
    );
    assert_eq!(
        args[1],
        (
            SuperTile {
                tensor: 1,
                coords: vec![Coord::Loop("kt".into()), Coord::Grid(1)]
            },
            AccessMode::Read
        )
    );
    assert_eq!(
        args[2],
        (
            SuperTile {
                tensor: 2,
                coords: vec![Coord::Grid(0), Coord::Grid(1)]
            },
            AccessMode::RMW
        )
    );
}

#[test]
fn matmul_leaves() {
    let p = compile(&first(MATMUL)).unwrap();
    assert_eq!(p.leaves.len(), 2);

    // step leaf: original kernel with the final store flipped to +=
    let step = &p.leaves[0];
    assert_eq!(step.kernel.name, "matmul");
    assert!(step.kernel.attrs.iter().all(|a| a.name != "cluster"));
    let Some(Stmt::Assign { op, .. }) = step.kernel.body.last() else {
        panic!("step leaf does not end in the accumulator store");
    };
    assert_eq!(*op, AssignOp::Add, "chain step must accumulate into C");
    assert_eq!(
        step.modes,
        vec![AccessMode::Read, AccessMode::Read, AccessMode::RMW]
    );

    // init leaf: zero-fill of the output supertile
    let init = &p.leaves[1];
    assert_eq!(init.kernel.name, "matmul_init");
    assert_eq!(init.kernel.params.len(), 1);
    assert_eq!(init.kernel.params[0].name, "C");
    assert_eq!(init.modes, vec![AccessMode::Write]);
}

#[test]
fn matmul_leaves_compile_at_device_scale() {
    let p = compile(&first(MATMUL)).unwrap();
    let step = verify_leaf(&p.leaves[0].kernel);
    assert!(step.contains("gpu.func"), "step leaf missing gpu.func");
    let init = verify_leaf(&p.leaves[1].kernel);
    assert!(init.contains("gpu.func"), "init leaf missing gpu.func");
}

/// A full GEMM: C = alpha*acc + beta*C_old. The accumulator epilogue reads
/// C back, so copy-elision folds beta into the init leaf (C = beta*C_old,
/// an rmw) and keeps alpha on the step chain (C = alpha*acc + c_old).
const GEMM: &str = r#"
@cluster(TILE_M in [4096, 16384], TILE_N in [4096, 16384], TILE_K in [4096, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel matmul(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N], alpha: f32, beta: f32) {
let pm = program_id(0)
let pn = program_id(1)
var acc: tile<f32>[TILE_M, TILE_N] = 0.0
for kt in range(0, K, TILE_K) {
    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
}
let c_old = C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N]
C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = alpha * acc + beta * c_old
}"#;

#[test]
fn gemm_cluster_ir() {
    let p = compile(&first(GEMM)).unwrap();

    // C is now read-modify-write: the init leaf seeds it with beta*C_old,
    // so its original value must be loaded rather than zeroed.
    let modes: Vec<_> = p.tensors.iter().map(|t| t.mode).collect();
    assert_eq!(
        modes,
        vec![AccessMode::Read, AccessMode::Read, AccessMode::RMW]
    );

    // both scalars stay out of the cluster IR's dataflow; the step compute
    // carries the kernel's own [alpha, beta], the init carries its local beta.
    assert_eq!(p.scalars[0].name, "alpha");
    assert_eq!(p.scalars[1].name, "beta");

    // body: init (beta*C_old) reads C rmw and carries a beta scalar
    let ClusterStmt::Compute {
        leaf: 1,
        args,
        scalars,
    } = &p.body[0]
    else {
        panic!("expected init compute, got {:?}", p.body[0]);
    };
    assert_eq!(
        args[0],
        (
            SuperTile {
                tensor: 2,
                coords: vec![Coord::Grid(0), Coord::Grid(1)]
            },
            AccessMode::RMW
        )
    );
    assert_eq!(scalars.len(), 1, "init leaf carries its beta coefficient");
    assert_eq!(p.scalars[scalars[0]].name, "beta");
    assert_eq!(
        p.scalars[scalars[0]].param_pos, 1,
        "beta sits at the init leaf's local position (after C)"
    );

    // the chain step still targets C(pm,pn) rmw (acc copy-elided) and carries
    // both kernel scalars
    let ClusterStmt::Loop { body, .. } = &p.body[1] else {
        panic!("expected the k-loop, got {:?}", p.body[1]);
    };
    let ClusterStmt::Compute {
        leaf: 0,
        args,
        scalars,
    } = &body[0]
    else {
        panic!("expected step compute, got {:?}", body[0]);
    };
    assert_eq!(scalars, &vec![0, 1]);
    assert_eq!(args[2].1, AccessMode::RMW);
}

#[test]
fn gemm_leaves() {
    let p = compile(&first(GEMM)).unwrap();
    assert_eq!(p.leaves.len(), 2);

    // step leaf: the store keeps the fused epilogue shape alpha*acc + c_old
    // (an implicit beta of 1); the real beta is applied once by the init.
    let step = &p.leaves[0];
    let Some(Stmt::Assign {
        op: AssignOp::Set,
        value:
            Expr::Binary {
                op: BinOp::Add,
                lhs,
                rhs,
            },
        ..
    }) = step.kernel.body.last()
    else {
        panic!("step leaf must end in `C[..] = alpha*acc + c_old`");
    };
    assert!(
        matches!(lhs.as_ref(), Expr::Binary { op: BinOp::Mul, .. }),
        "step store's acc term should be alpha*acc"
    );
    assert!(
        matches!(rhs.as_ref(), Expr::Var(n) if n == "c_old"),
        "step store's prior-C term should be c_old with an implicit beta of 1"
    );

    // init leaf: C = beta*c_old, an rmw over C plus the beta scalar param
    let init = &p.leaves[1];
    assert_eq!(init.kernel.name, "matmul_init");
    assert_eq!(init.kernel.params.len(), 2);
    assert_eq!(init.kernel.params[0].name, "C");
    assert_eq!(init.kernel.params[1].name, "beta");
    assert_eq!(init.modes, vec![AccessMode::RMW, AccessMode::Read]);
}

#[test]
fn gemm_leaves_compile_at_device_scale() {
    let p = compile(&first(GEMM)).unwrap();
    let step = verify_leaf(&p.leaves[0].kernel);
    assert!(step.contains("gpu.func"), "step leaf missing gpu.func");
    let init = verify_leaf(&p.leaves[1].kernel);
    assert!(init.contains("gpu.func"), "init leaf missing gpu.func");
}

#[test]
fn add_cluster_ir() {
    let p = compile(&first(ADD)).unwrap();
    assert_eq!(p.grid.len(), 1);
    assert_eq!(p.grid[0].dim, Dim::Sym("N".into()));
    assert_eq!(p.grid[0].super_sym, "BLOCK");
    assert_eq!(p.leaves.len(), 1, "elementwise kernels need no init leaf");
    assert_eq!(p.body.len(), 1);
    let ClusterStmt::Compute { leaf: 0, args, .. } = &p.body[0] else {
        panic!("expected one compute, got {:?}", p.body[0]);
    };
    let modes: Vec<_> = args.iter().map(|(_, m)| *m).collect();
    assert_eq!(
        modes,
        vec![AccessMode::Read, AccessMode::Read, AccessMode::Write]
    );
    for (i, (r, _)) in args.iter().enumerate() {
        assert_eq!(
            *r,
            SuperTile {
                tensor: i,
                coords: vec![Coord::Grid(0)]
            }
        );
    }
    verify_leaf(&p.leaves[0].kernel);
}

/// A flash-attention-shaped kernel: grid over query blocks, a device-scale
/// key loop kept inside the leaf, full : slices over the head dim, running
/// tile state, and a scalar parameter.
const FLASH: &str = r#"
@cluster(BR in [1024, 4096])
@autotune(D in [64], BR in [32, 128], BC in [32, 128])
kernel attn(Q: tensor<f32>[Nq, D],
        K: tensor<f32>[Nk, D],
        V: tensor<f32>[Nk, D],
        O: tensor<f32>[Nq, D],
        scale: f32) {
let pid = program_id(0)
let row = pid * BR
let q = Q[row :+ BR, :]
var acc: tile<f32>[BR, D] = 0.0
var l: tile<f32>[BR, 1] = 0.0
for kt in range(0, Nk, BC) {
    let k = K[kt :+ BC, :]
    let v = V[kt :+ BC, :]
    var s: tile<f32>[BR, BC] = dot_t(q, k)
    s = s * scale
    var p: tile<f32>[BR, BC] = exp(s)
    l += rowsum(p)
    acc += dot(p, v)
}
acc = acc / l
O[row :+ BR, :] = acc
}"#;

#[test]
fn accepts_scalar_params() {
    // matmul + an (unused) alpha scalar still clusters; the scalar is
    // recorded with its parameter position and the step compute carries it.
    let src = MATMUL.replace("C: tensor<f32>[M, N])", "C: tensor<f32>[M, N], alpha: f32)");
    let p = compile(&first(&src)).unwrap();
    assert_eq!(p.scalars.len(), 1);
    assert_eq!(p.scalars[0].name, "alpha");
    assert_eq!(p.scalars[0].data_type, crate::tile::DataType::F32);
    assert_eq!(p.scalars[0].param_pos, 3);
    // the step compute (leaf 0) inside the k-loop passes the scalar
    let ClusterStmt::Loop { body, .. } = &p.body[1] else {
        panic!("expected the k-loop");
    };
    let ClusterStmt::Compute {
        leaf: 0, scalars, ..
    } = &body[0]
    else {
        panic!("expected the step compute");
    };
    assert_eq!(scalars, &vec![0]);
}

#[test]
fn flash_single_leaf() {
    let p = compile(&first(FLASH)).unwrap();

    // one grid axis over query blocks, supertiled by BR
    assert_eq!(p.grid.len(), 1);
    assert_eq!(p.grid[0].dim, Dim::Sym("Nq".into()));
    assert_eq!(p.grid[0].super_sym, "BR");

    // the whole kernel is a single leaf (no cluster loop, no init leaf)
    assert_eq!(p.leaves.len(), 1);
    assert_eq!(p.leaves[0].kernel.name, "attn");

    // scalar recorded at its parameter position (after the four tensors)
    assert_eq!(p.scalars.len(), 1);
    assert_eq!(p.scalars[0].name, "scale");
    assert_eq!(p.scalars[0].param_pos, 4);

    // Q/O are query-tiled (grid, full head dim); K/V are read whole
    assert_eq!(p.tensors[0].super_syms, vec!["BR", "D"]); // Q
    assert_eq!(p.tensors[1].super_syms, vec!["Nk", "D"]); // K
    assert_eq!(p.tensors[2].super_syms, vec!["Nk", "D"]); // V
    assert_eq!(p.tensors[3].super_syms, vec!["BR", "D"]); // O
    let modes: Vec<_> = p.tensors.iter().map(|t| t.mode).collect();
    assert_eq!(
        modes,
        vec![
            AccessMode::Read,
            AccessMode::Read,
            AccessMode::Read,
            AccessMode::Write
        ]
    );

    // exactly one compute: Q(p0,:), K(:,:), V(:,:) read, O(p0,:) written,
    // carrying the scalar
    assert_eq!(p.body.len(), 1);
    let ClusterStmt::Compute {
        leaf: 0,
        args,
        scalars,
    } = &p.body[0]
    else {
        panic!("expected one leaf compute, got {:?}", p.body[0]);
    };
    assert_eq!(scalars, &vec![0]);
    assert_eq!(
        args[0],
        (
            SuperTile {
                tensor: 0,
                coords: vec![Coord::Grid(0), Coord::Full]
            },
            AccessMode::Read
        )
    );
    assert_eq!(
        args[1],
        (
            SuperTile {
                tensor: 1,
                coords: vec![Coord::Full, Coord::Full]
            },
            AccessMode::Read
        )
    );
    assert_eq!(
        args[3],
        (
            SuperTile {
                tensor: 3,
                coords: vec![Coord::Grid(0), Coord::Full]
            },
            AccessMode::Write
        )
    );

    // the leaf device-compiles (the f32 scalar lands as a value param)
    let text = verify_leaf(&p.leaves[0].kernel);
    assert!(text.contains("f32"), "leaf missing the scalar param");
}

#[test]
fn rejects_misaligned_offsets() {
    let src = MATMUL.replace("A[pm * TILE_M :+ TILE_M", "A[pm * TILE_M + 1 :+ TILE_M");
    let err = compile(&first(&src)).unwrap_err().to_string();
    assert!(err.contains("not supertile-aligned"), "got: {err}");
}

#[test]
fn rejects_extent_step_mismatch() {
    // slice along K uses TILE_M as extent while the loop steps by TILE_K
    let src = MATMUL.replace("kt :+ TILE_K]", "kt :+ TILE_M]");
    let err = compile(&first(&src)).unwrap_err().to_string();
    assert!(err.contains("steps by"), "got: {err}");
}

#[test]
fn rejects_cluster_dim_without_autotune() {
    let src = MATMUL.replace(
        "@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])",
        "@autotune(TILE_M in [32, 256], TILE_N in [32, 256])",
    );
    let err = compile(&first(&src)).unwrap_err().to_string();
    assert!(err.contains("must also be an @autotune dim"), "got: {err}");
}

#[test]
fn rejects_multiple_cluster_computes() {
    // Two rmw chains over the same cluster loop is still unsupported on the
    // accumulator path.
    let src = r#"
@cluster(TILE_M in [4096, 16384], TILE_N in [4096, 16384], TILE_K in [4096, 16384])
@autotune(TILE_M in [32, 256], TILE_N in [32, 256], TILE_K in [4, 32])
kernel two(A: tensor<f32>[M, K], B: tensor<f32>[K, N], C: tensor<f32>[M, N], D: tensor<f32>[M, N]) {
let pm = program_id(0)
let pn = program_id(1)
var acc: tile<f32>[TILE_M, TILE_N] = 0.0
for kt in range(0, K, TILE_K) {
    let a = A[pm * TILE_M :+ TILE_M, kt :+ TILE_K]
    let b = B[kt :+ TILE_K, pn * TILE_N :+ TILE_N]
    acc += dot(a, b)
    D[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] += dot(a, b)
}
C[pm * TILE_M :+ TILE_M, pn * TILE_N :+ TILE_N] = acc
}"#;
    let err = compile(&first(src)).unwrap_err().to_string();
    assert!(err.contains("multiple compute statements"), "got: {err}");
}
