use super::*;
use std::collections::HashMap;

fn node(op: &str, ins: &[&str], outs: &[&str]) -> Node {
    Node {
        name: format!("{op}_node"),
        op_type: op.into(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        outputs: outs.iter().map(|s| s.to_string()).collect(),
        attrs: HashMap::new(),
    }
}

fn dims_fn(map: &HashMap<String, Dims>) -> impl Fn(&str) -> Option<Dims> + '_ {
    move |e: &str| map.get(e).cloned()
}

#[test]
fn lowers_matmul_grid_and_params() {
    let dims = HashMap::from([
        ("X".to_string(), vec![128, 64]),
        ("W".to_string(), vec![64, 192]),
    ]);
    let n = node("MatMul", &["X", "W"], &["Y"]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();

    assert_eq!(plan.kernel_name, "matmul");
    // grid is (M/TILE_M, N/TILE_N), so (128/64, 192/64) is (2, 3).
    assert_eq!(plan.grid, (2, 3, 1));
    assert_eq!(plan.block, 256);
    assert_eq!(plan.output, "Y");
    // The A, B and C descriptors, in order.
    assert_eq!(
        plan.params[0],
        Param::Tensor {
            edge: "X".into(),
            view: vec![128, 64]
        }
    );
    assert_eq!(
        plan.params[2],
        Param::Tensor {
            edge: "Y".into(),
            view: vec![128, 192]
        }
    );
    assert!(plan.source.contains("kernel matmul"));
    assert!(plan.source.contains("acc += dot(a, b)"));
}

#[test]
fn matmul_rejects_mismatched_k() {
    let dims = HashMap::from([
        ("X".to_string(), vec![128, 64]),
        ("W".to_string(), vec![32, 32]),
    ]);
    let n = node("MatMul", &["X", "W"], &["Y"]);
    assert!(lower_node(&n, &dims_fn(&dims)).is_err());
}

#[test]
fn lowers_add_as_flattened_elementwise() {
    let dims = HashMap::from([
        ("A".to_string(), vec![64, 64]),
        ("B".to_string(), vec![64, 64]),
    ]);
    let n = node("Add", &["A", "B"], &["Y"]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();

    assert_eq!(plan.kernel_name, "add");
    // 4096 elements over a 256 tile is 16 blocks.
    assert_eq!(plan.grid, (16, 1, 1));
    // Tensors are viewed as rank-1.
    assert_eq!(
        plan.params[0],
        Param::Tensor {
            edge: "A".into(),
            view: vec![4096]
        }
    );
    assert!(plan.source.contains("A[base :+ BLOCK] + B[base :+ BLOCK]"));
    assert_eq!(plan.overrides, vec![("BLOCK".to_string(), 256)]);
}

#[test]
fn each_arith_op_uses_its_symbol() {
    let dims = HashMap::from([("A".to_string(), vec![256]), ("B".to_string(), vec![256])]);
    for (op, sym, name) in [
        ("Add", "+", "add"),
        ("Sub", "-", "sub"),
        ("Mul", "*", "mul"),
        ("Div", "/", "div"),
    ] {
        let n = node(op, &["A", "B"], &["Y"]);
        let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
        assert_eq!(plan.kernel_name, name);
        assert!(
            plan.source
                .contains(&format!("A[base :+ BLOCK] {sym} B[base :+ BLOCK]")),
            "op {op} should emit symbol {sym}"
        );
    }
}

#[test]
fn add_rejects_shape_mismatch() {
    let dims = HashMap::from([("A".to_string(), vec![256]), ("B".to_string(), vec![128])]);
    let n = node("Add", &["A", "B"], &["Y"]);
    assert!(lower_node(&n, &dims_fn(&dims)).is_err());
}

#[test]
fn lowers_relu_as_tmax_zero() {
    let dims = HashMap::from([("X".to_string(), vec![32, 32])]);
    let n = node("Relu", &["X"], &["Y"]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();

    assert_eq!(plan.kernel_name, "relu");
    assert_eq!(plan.grid, (4, 1, 1)); // 1024 / 256
    assert!(plan.source.contains("var zero: tile<f32>[BLOCK] = 0.0"));
    assert!(plan.source.contains("tmax(X[base :+ BLOCK], zero)"));
    assert_eq!(plan.params.len(), 2);
}

#[test]
fn non_divisible_extent_is_rejected() {
    // 300 elements is not a multiple of the 256 tile.
    let dims = HashMap::from([("A".to_string(), vec![300]), ("B".to_string(), vec![300])]);
    let n = node("Add", &["A", "B"], &["Y"]);
    assert!(lower_node(&n, &dims_fn(&dims)).is_err());
}

#[test]
fn unsupported_op_is_rejected() {
    let dims = HashMap::new();
    let n = node("Conv", &["X", "W"], &["Y"]);
    assert!(lower_node(&n, &dims_fn(&dims)).is_err());
}

#[test]
fn lowers_bias_add_as_row_broadcast() {
    let dims = HashMap::from([
        ("A".to_string(), vec![128, 64]),
        ("b".to_string(), vec![64]),
    ]);
    let n = node("Add", &["A", "b"], &["Y"]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();

    assert_eq!(plan.kernel_name, "add_bias");
    // 128 rows over a 16 row-tile is 8 blocks.
    assert_eq!(plan.grid, (8, 1, 1));
    // The bias goes in as a [1, W] descriptor for the broadcast.
    assert_eq!(
        plan.params[1],
        Param::Tensor {
            edge: "b".into(),
            view: vec![1, 64]
        }
    );
    assert!(plan.source.contains("var y: tile<f32>[16, 64] = a + bias"));
}

#[test]
fn lowers_gelu_as_logistic() {
    let dims = HashMap::from([("X".to_string(), vec![64, 64])]);
    let n = node("Gelu", &["X"], &["Y"]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
    assert_eq!(plan.kernel_name, "gelu");
    assert_eq!(plan.grid, (16, 1, 1)); // 4096 / 256
    assert!(plan.source.contains("1.0 + exp(-1.702 * x)"));
    assert!(plan.source.contains("Y[base :+ BLOCK] = x / d"));
}

#[test]
fn lowers_softmax_rowwise() {
    let dims = HashMap::from([("X".to_string(), vec![32, 64])]);
    let n = node("Softmax", &["X"], &["Y"]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
    assert_eq!(plan.kernel_name, "softmax");
    assert_eq!(plan.grid, (2, 1, 1)); // 32 rows / 16
    assert!(plan.source.contains("rowmax(x)"));
    assert!(plan.source.contains("exp(x - m)"));
    assert!(plan.source.contains("var y: tile<f32>[16, 64] = e / s"));
}

#[test]
fn lowers_fused_linear_with_gelu_epilogue() {
    let mut n = node("PhobosFusedLinear", &["A", "B", "bias"], &["Y"]);
    n.attrs.insert(
        "activation".into(),
        crate::ir::Attribute::String("gelu".into()),
    );
    let dims = HashMap::from([
        ("A".to_string(), vec![128, 64]),
        ("B".to_string(), vec![64, 128]),
    ]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
    assert_eq!(plan.kernel_name, "fused_linear");
    assert_eq!(plan.grid, (4, 4, 1)); // 128/32, 128/32
    assert!(plan.source.contains("acc = acc + bias"));
    assert!(plan.source.contains("1.0 + exp(-1.702 * acc)"));
    // A, B, the [1, N] bias, then C.
    assert_eq!(
        plan.params[2],
        Param::Tensor {
            edge: "bias".into(),
            view: vec![1, 128]
        }
    );
    assert_eq!(plan.params.len(), 4);
}

#[test]
fn lowers_flash_attention() {
    let mut n = node("PhobosFlashAttention", &["Q", "K", "V"], &["O"]);
    n.attrs
        .insert("scale".into(), crate::ir::Attribute::Float(0.125));
    let dims = HashMap::from([
        ("Q".to_string(), vec![64, 32]),
        ("K".to_string(), vec![64, 32]),
        ("V".to_string(), vec![64, 32]),
    ]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
    assert_eq!(plan.kernel_name, "flash_attention");
    assert_eq!(plan.grid, (2, 1, 1)); // Nq=64, BR=32
    assert!(plan.source.contains("dot_t(q, k)"));
    assert!(plan.source.contains("acc += dot(p, v)"));
    // Q, K, V, O, then the scale.
    assert_eq!(plan.params.len(), 5);
    assert_eq!(plan.params[4], Param::ScalarF32(0.125));
    assert_eq!(
        plan.overrides,
        vec![("D".into(), 32), ("BR".into(), 32), ("BC".into(), 32)]
    );
}

#[test]
fn lowers_layernorm_with_sqrt_and_eps() {
    let mut n = node("LayerNormalization", &["X", "G", "B"], &["Y"]);
    n.attrs
        .insert("epsilon".into(), crate::ir::Attribute::Float(1e-3));
    let dims = HashMap::from([
        ("X".to_string(), vec![48, 64]),
        ("G".to_string(), vec![64]),
        ("B".to_string(), vec![64]),
    ]);
    let plan = lower_node(&n, &dims_fn(&dims)).unwrap();
    assert_eq!(plan.kernel_name, "layernorm");
    assert_eq!(plan.grid, (3, 1, 1)); // 48 rows / 16
    assert!(plan.source.contains("sqrt(vv + 0.00100000)"));
    assert!(plan.source.contains("var mu: tile<f32>[16, 1] = s / 64.0"));
    // The scale and bias come in as [1, W] descriptors.
    assert_eq!(
        plan.params[1],
        Param::Tensor {
            edge: "G".into(),
            view: vec![1, 64]
        }
    );
    assert_eq!(plan.params.len(), 4);
}
