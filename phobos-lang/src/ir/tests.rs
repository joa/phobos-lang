use super::*;
use crate::ast::BinOp;

fn kernel(params: &[Type]) -> Ir {
    let info = KernelInfo {
        name: "k".into(),
        cta_threads: 256,
        ..Default::default()
    };
    Ir::new(info, params)
}

fn tensor(elem: Scalar, rank: usize) -> Type {
    Type::Tensor(TensorType {
        elem,
        shape: vec![Extent::Dyn; rank],
        div: vec![1; rank],
    })
}

fn plain() -> ForInfo {
    carried(0)
}

fn carried(n: usize) -> ForInfo {
    ForInfo {
        bounds: Bounds::Dynamic,
        ragged: false,
        carried: n,
        hoisted: 0,
        pipeline: None,
    }
}

fn index_const(b: &mut Builder<'_>, n: i64) -> ValueId {
    b.value(OpKind::Const(Literal::Int(n)), &[], Type::INDEX)
}

/// The errors the verifier reports, one per line, or empty.
fn errors(ir: &Ir) -> String {
    verify::verify(ir).err().map(|e| e.to_string()).unwrap_or_default()
}

#[test]
fn create_op_records_operands_results_and_uses() {
    let mut ir = kernel(&[Type::INDEX]);
    let n = ir.args(ir.entry())[0];
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let one = index_const(&mut b, 1);
    let sum = b.value(OpKind::Binary(BinOp::Add), &[n, one], Type::INDEX);
    let add = ir.def_op(sum).unwrap();

    assert_eq!(ir.operands(add), &[n, one]);
    assert_eq!(ir.results(add), &[sum]);
    assert_eq!(ir.def(sum), Def::Result { op: add, index: 0 });
    assert_eq!(ir.def(n), Def::Arg { block: entry, index: 0 });
    assert_eq!(ir.uses(n), &[Use { op: add, index: 0 }]);
    assert_eq!(ir.uses(one), &[Use { op: add, index: 1 }]);
    assert!(ir.uses(sum).is_empty());
    assert_eq!(ir.position(add), 1);
    assert_eq!(errors(&ir), "");
}

#[test]
fn replace_all_uses_moves_every_use() {
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let a = index_const(&mut b, 1);
    let c = index_const(&mut b, 2);
    let x = b.value(OpKind::Binary(BinOp::Add), &[a, a], Type::INDEX);
    let y = b.value(OpKind::Binary(BinOp::Mul), &[x, a], Type::INDEX);
    ir.replace_all_uses(a, c);

    let (xop, yop) = (ir.def_op(x).unwrap(), ir.def_op(y).unwrap());
    assert!(ir.uses(a).is_empty());
    assert_eq!(ir.operands(xop), &[c, c]);
    assert_eq!(ir.operands(yop), &[x, c]);
    assert_eq!(ir.uses(c).len(), 3);
    assert_eq!(errors(&ir), "");
}

#[test]
#[should_panic(expected = "while %1 is used by op2")]
fn erase_refuses_a_used_result() {
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let a = index_const(&mut b, 1);
    let one = index_const(&mut b, 1);
    b.value(OpKind::Binary(BinOp::Add), &[a, one], Type::INDEX);
    let def = ir.def_op(one).unwrap();
    ir.erase_op(def);
}

#[test]
fn erase_removes_nested_blocks_and_operand_uses() {
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let lo = index_const(&mut b, 0);
    let hi = index_const(&mut b, 8);
    let body = b.block_with(&[Type::INDEX]);
    let iv = b.ir.args(body)[0];
    b.in_block(body, |inner| {
        inner.value(OpKind::Binary(BinOp::Add), &[iv, lo], Type::INDEX);
        inner.stmt(OpKind::Yield, &[]);
    });
    let for_op = b.op(OpKind::For(plain()), &[lo, hi, lo], Vec::new(), vec![body]);
    assert_eq!(ir.uses(lo).len(), 3);
    assert_eq!(errors(&ir), "");

    ir.erase_op(for_op);
    assert!(!ir.is_alive(for_op));
    assert_eq!(ir.uses(lo).len(), 0);
    assert_eq!(ir.uses(hi).len(), 0);
    assert_eq!(ir.ops(entry).len(), 2);
    assert_eq!(errors(&ir), "");
}

#[test]
fn a_def_moved_past_its_use_fails_dominance() {
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let a = index_const(&mut b, 1);
    let x = b.value(OpKind::Unary(UnOp::Neg), &[a], Type::INDEX);
    let (aop, xop) = (ir.def_op(a).unwrap(), ir.def_op(x).unwrap());
    assert!(ir.dominates(a, xop));

    ir.move_op(aop, At::After(xop));
    assert!(!ir.dominates(a, xop));
    assert!(errors(&ir).contains("operand 0 (%0) is not visible here"));

    ir.move_op(aop, At::Start(entry));
    assert_eq!(errors(&ir), "");
}

#[test]
fn dominance_follows_the_block_nesting() {
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let outer = index_const(&mut b, 0);
    let cond = b.value(OpKind::Const(Literal::Bool(true)), &[], Type::BOOL);
    let then = b.block_with(&[]);
    let inner = b.in_block(then, |t| {
        let v = t.value(OpKind::Binary(BinOp::Add), &[outer, outer], Type::INDEX);
        t.stmt(OpKind::Yield, &[]);
        v
    });
    let if_op = b.op(OpKind::If, &[cond], Vec::new(), vec![then]);
    let after = b.value(OpKind::Unary(UnOp::Neg), &[outer], Type::INDEX);
    let after_op = ir.def_op(after).unwrap();
    let inner_op = ir.def_op(inner).unwrap();

    assert!(ir.dominates(outer, inner_op), "an outer value is visible inside");
    assert!(!ir.dominates(inner, after_op), "an inner value is not visible after");
    assert!(!ir.dominates(after, inner_op), "a later outer value is not visible inside");
    assert!(ir.contains(if_op, inner_op));
    assert!(!ir.contains(if_op, after_op));
    assert_eq!(errors(&ir), "");

    // An op's results are not visible inside its own blocks.
    ir.set_operand(inner_op, 1, cond);
    let ok = errors(&ir);
    assert!(ok.contains("operands differ"), "{ok}");
}

#[test]
fn the_verifier_names_each_broken_rule() {
    // A terminator that is not last.
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let lo = index_const(&mut b, 0);
    let body = b.block_with(&[Type::INDEX]);
    b.in_block(body, |inner| {
        inner.stmt(OpKind::Yield, &[]);
        index_const(inner, 1);
    });
    b.op(OpKind::For(plain()), &[lo, lo, lo], Vec::new(), vec![body]);
    let e = errors(&ir);
    assert!(e.contains("a terminator that is not the last op"), "{e}");
    assert!(e.contains("ends in const rather than yield"), "{e}");

    // A loop whose body does not yield what it carries.
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let lo = index_const(&mut b, 0);
    let body = b.block_with(&[Type::INDEX, Type::INDEX]);
    b.in_block(body, |inner| {
        inner.stmt(OpKind::Yield, &[]);
    });
    b.op(OpKind::For(carried(1)), &[lo, lo, lo, lo], vec![Type::INDEX], vec![body]);
    let e = errors(&ir);
    assert!(e.contains("yields 0 values, 1 expected"), "{e}");

    // Mismatched binary operands and a store of the wrong element type.
    let mut ir = kernel(&[tensor(Scalar::F32, 1)]);
    let t = ir.args(ir.entry())[0];
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let i = index_const(&mut b, 0);
    let f = b.value(OpKind::Const(Literal::Float(1.0)), &[], Type::Scalar(Scalar::F32));
    b.value(OpKind::Binary(BinOp::Add), &[i, f], Type::INDEX);
    b.stmt(OpKind::Store, &[i, t, i]);
    let e = errors(&ir);
    assert!(e.contains("operands differ: index vs f32"), "{e}");
    assert!(e.contains("stored value must be the element type"), "{e}");

    // A block created and never given to an op, and an if with results
    // but no else.
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    b.block_with(&[]);
    let cond = b.value(OpKind::Const(Literal::Bool(false)), &[], Type::BOOL);
    let then = b.block_with(&[]);
    b.in_block(then, |t| {
        let one = index_const(t, 1);
        t.stmt(OpKind::Yield, &[one]);
    });
    b.op(OpKind::If, &[cond], vec![Type::INDEX], vec![then]);
    let e = errors(&ir);
    assert!(e.contains("^1 belongs to no op"), "{e}");
    assert!(e.contains("yields results but has no else block"), "{e}");
}

#[test]
fn a_gemm_accumulator_is_carried_alone_and_folds_slices() {
    let gemm = Type::Gemm(GemmType {
        acc: Scalar::F32,
        m: 64,
        n: 64,
        k: 16,
    });
    let mut ir = kernel(&[]);
    let entry = ir.entry();
    let mut b = Builder::at_end(&mut ir, entry);
    let lo = index_const(&mut b, 0);
    let f = b.value(OpKind::Const(Literal::Float(0.0)), &[], Type::Scalar(Scalar::F32));
    let seed = b.value(OpKind::GemmInit, &[f], gemm.clone());
    let body = b.block_with(&[Type::INDEX, gemm.clone(), gemm.clone()]);
    b.in_block(body, |inner| {
        let acc = inner.ir.args(body)[1];
        let one = index_const(inner, 1);
        let next = inner.value(OpKind::GemmDot, &[acc, one, one], gemm.clone());
        inner.stmt(OpKind::Yield, &[next, next]);
    });
    b.op(
        OpKind::For(carried(2)),
        &[lo, lo, lo, seed, seed],
        vec![gemm.clone(), gemm],
        vec![body],
    );
    let e = errors(&ir);
    assert!(e.contains("a gemm accumulator is carried alone"), "{e}");
    assert!(e.contains("operand 1 must be an unmasked slice or a shared tile, is index"), "{e}");
}

#[test]
fn slice_operands_are_laid_out_by_dimension() {
    let s = Slice {
        sizes: vec![Extent::Fixed(64), Extent::Dyn, Extent::Fixed(4)],
        masked: vec![true, false, true],
        divs: vec![1, 1, 1],
    };
    assert_eq!(s.operand_count(), 1 + 3 + 1 + 2);
    assert_eq!(s.offset_operand(2), 3);
    assert_eq!(s.mask_operand(0), Some(5));
    assert_eq!(s.mask_operand(1), None);
    assert_eq!(s.mask_operand(2), Some(6));
}

#[test]
fn intrinsic_names_round_trip() {
    let mut all = vec![
        Intrinsic::QdotT,
        Intrinsic::QmmaT,
        Intrinsic::Gather,
        Intrinsic::ArgSel,
        Intrinsic::RmsNormQ,
        Intrinsic::WarpPartial,
    ];
    for fmt in RawFmt::ALL {
        all.extend([
            Intrinsic::RawQdot(fmt),
            Intrinsic::RawQdotI8(fmt),
            Intrinsic::RawQmma(fmt),
            Intrinsic::RawQmmaStaged(fmt),
            Intrinsic::RawQgemm(fmt),
            Intrinsic::RawQdecode(fmt),
        ]);
    }
    for i in all {
        assert_eq!(Intrinsic::from_name(&i.name()), Some(i), "{}", i.name());
    }
    assert_eq!(Intrinsic::from_name("wobble"), None);
    assert_eq!(Intrinsic::from_name("q9k_qdot_t"), None);
}

#[test]
fn the_printer_shows_names_types_and_nesting() {
    let mut ir = kernel(&[tensor(Scalar::F32, 2), Type::INDEX]);
    let entry = ir.entry();
    let (a, n) = (ir.args(entry)[0], ir.args(entry)[1]);
    ir.set_name(a, "A");
    ir.set_name(n, "n");
    let mut b = Builder::at_end(&mut ir, entry);
    let zero = index_const(&mut b, 0);
    let acc = b.value(OpKind::Alloc, &[], Type::shared_tile(Scalar::F32, &[64, 64]));
    b.ir.set_name(acc, "acc");
    let body = b.block_with(&[Type::INDEX]);
    let iv = b.ir.args(body)[0];
    b.ir.set_name(iv, "i");
    b.in_block(body, |inner| {
        let slice = Slice {
            sizes: vec![Extent::Fixed(64), Extent::Fixed(64)],
            masked: vec![false, true],
            divs: vec![1, 1],
        };
        let ty = Type::Tile(TileType {
            elem: Scalar::F32,
            shape: slice.sizes.clone(),
            layout: Layout::CONTIGUOUS,
            space: Space::Global,
        });
        let view = inner.value(OpKind::Slice(slice), &[a, iv, zero, n], ty);
        inner.stmt(
            OpKind::DotInto {
                transpose: false,
                accumulate: true,
                aliased: false,
            },
            &[view, view, acc],
        );
        inner.stmt(OpKind::Yield, &[]);
    });
    b.op(OpKind::For(plain()), &[zero, n, zero], Vec::new(), vec![body]);
    let cond = b.value(OpKind::Const(Literal::Bool(true)), &[], Type::BOOL);
    let then = b.block_with(&[]);
    b.in_block(then, |t| {
        t.stmt(OpKind::Yield, &[]);
    });
    let els = b.block_with(&[]);
    b.in_block(els, |t| {
        t.stmt(OpKind::Yield, &[]);
    });
    b.op(OpKind::If, &[cond], Vec::new(), vec![then, els]);
    assert_eq!(errors(&ir), "");

    let expected = "\
kernel k(%A.0: tensor<f32>[?, ?], %n.1: index) {
  %2: index = const 0
  %acc.3: tile<f32>[64, 64]@shared = alloc
  for %2, %n.1, %2 {
  ^(%i.4: index)
    %5: tile<f32>[64, 64]@global = slice [64, 64] mask [-, m] %A.0, %i.4, %2, %n.1
    dot_into acc %5, %5, %acc.3
    yield
  }
  %6: bool = const true
  if %6 {
    yield
  } else {
    yield
  }
}
";
    assert_eq!(ir.to_string(), expected);
}

#[test]
fn tile_types_know_their_bytes() {
    let t = TileType {
        elem: Scalar::F16,
        shape: vec![Extent::Fixed(64), Extent::Fixed(64)],
        layout: Layout {
            row_stride: Some(72),
            swizzle: None,
            align_div: 8,
        },
        space: Space::Shared,
    };
    assert_eq!(t.physical_elems(), Some(64 * 72));
    assert_eq!(t.bytes(), Some(64 * 72 * 2));
    assert_eq!(
        Type::Tile(t).to_string(),
        "tile<f16>[64, 64]@shared{stride 72, align 8}"
    );
    let dynamic = TileType {
        elem: Scalar::F32,
        shape: vec![Extent::Dyn],
        layout: Layout::CONTIGUOUS,
        space: Space::Global,
    };
    assert_eq!(dynamic.bytes(), None);
}
