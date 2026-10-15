// What the persistent-grid passes decide, read off the IR they leave.

use crate::ir::verify::verify;

/// The first kernel of `src` built and through the passes, printed.
fn passed(src: &str) -> String {
    let kernels = crate::parse(src).unwrap();
    let base = phobos_base::context::Context::default();
    let target = crate::codegen::target::build_target(&base);
    let (mut ir, mut report) = crate::ir::build::build(&base, target, &kernels[0]).unwrap_or_else(|e| panic!("{e:#}"));
    super::run(&mut ir, &mut report, &target, &kernels[0]).unwrap_or_else(|e| panic!("{e:#}"));
    verify(&ir).unwrap_or_else(|e| panic!("{e}\n{ir}"));
    ir.to_string()
}

/// The lines of `ir` from the one containing `from`.
fn after<'a>(ir: &'a str, from: &str) -> &'a str {
    &ir[ir.find(from).unwrap_or_else(|| panic!("no `{from}` in:\n{ir}"))..]
}

/// An MLP's shape: a row every block computes alike, units of it projected
/// and stored, then units of the stored row read whole and accumulated
/// into the residual the first stage read.
const MLP: &str = "@launch(256)
@persistent(BLOCKS)
@autotune(BLOCKS in [8])
kernel k(X: tensor<f32>[1, 256], W: tensor<f32>[256, 256]) {
  let p = ctx.block.x
  var H: tensor<f32>[1, 256]
  var V: tensor<f32>[1, 256]
  H[0, ctx.thread] = X[0, ctx.thread] * 2.0
  barrier()
  for i in range(0, 4) {
    let u = p + i * ctx.blocks.x
    if u < 32 {
      V[0 :+ 1, u * 8 :+ 8] = H[0 :+ 1, u * 8 :+ 8] + 1.0
    }
  }
  for i in range(0, 4) {
    let u = p + i * ctx.blocks.x
    if u < 32 {
      X[0 :+ 1, u * 8 :+ 8] += V[0 :+ 1, u * 8 :+ 8] * 3.0 + V[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]
    }
  }
}";

#[test]
fn one_barrier_orders_the_stored_row_and_the_residual() {
    let ir = passed(MLP);
    // Every block writes H alike, so it is a shared tile of its own; V is
    // written per unit, so it is a parameter, before the barrier word.
    assert!(ir.contains("tile<f32>[1, 256]@shared{align 256} = alloc"), "{ir}");
    let signature = ir.lines().next().expect("the kernel line");
    let (v, word) = (signature.find("%V.").expect("V is a parameter"), signature.find("%grid_barrier.").expect("the word"));
    assert!(v < word && !signature.contains("%H."), "{ir}");
    assert_eq!(ir.matches("grid_barrier %").count(), 1, "{ir}");
    // Between the two unit loops, not inside either.
    let second = after(&ir, "grid_barrier %");
    assert!(second.contains("+= ") || second.contains("fused"), "{ir}");
    assert_eq!(after(&ir, "grid_barrier %").matches("for ").count(), 1, "{ir}");
}

#[test]
fn a_unit_reading_only_its_own_needs_no_barrier() {
    // The second loop reads exactly the elements its unit wrote, and the
    // residual is not read before it.
    let src = MLP
        .replace(" + V[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]", "")
        .replace("H[0, ctx.thread] = X[0, ctx.thread] * 2.0", "H[0, ctx.thread] = 2.0");
    let ir = passed(&src);
    assert!(!ir.contains("grid_barrier"), "{ir}");
}

#[test]
fn a_rotated_loop_is_another_owner() {
    // The second loop starts its units on other blocks, so a read of its
    // own unit's elements is another block's write.
    let src = MLP
        .replace(" + V[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]", "")
        .replace("H[0, ctx.thread] = X[0, ctx.thread] * 2.0", "H[0, ctx.thread] = 2.0")
        .replace(
            "  for i in range(0, 4) {\n    let u = p + i * ctx.blocks.x\n    if u < 32 {\n      X",
            "  for i in range(0, 4) {\n    let u = (p + 3) % ctx.blocks.x + i * ctx.blocks.x\n    if u < 32 {\n      X",
        );
    let ir = passed(&src);
    assert_eq!(ir.matches("grid_barrier %").count(), 1, "{ir}");
}

#[test]
fn a_layer_loop_orders_its_back_edge() {
    // Two layers: the bottom of one writes the row the top of the next reads
    // whole, so a barrier opens the body as well as the one inside it.
    let src = "@launch(256)
@persistent(BLOCKS)
@autotune(BLOCKS in [8])
kernel k(X: tensor<f32>[1, 256], Y: tensor<f32>[1, 256]) {
  let p = ctx.block.x
  for l in range(0, 2) {
    for i in range(0, 4) {
      let u = p + i * ctx.blocks.x
      if u < 32 {
        Y[0 :+ 1, u * 8 :+ 8] = X[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]
      }
    }
    for i in range(0, 4) {
      let u = p + i * ctx.blocks.x
      if u < 32 {
        X[0 :+ 1, u * 8 :+ 8] = Y[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]
      }
    }
  }
}";
    let ir = passed(src);
    assert_eq!(ir.matches("grid_barrier %").count(), 2, "{ir}");
    let body = after(&ir, "^(%l.");
    let first = body.find("grid_barrier %").expect("a barrier");
    assert!(first < body.find("for ").expect("the first unit loop"), "the back edge's barrier opens the body:\n{ir}");
}

/// Why the passes refuse `src`'s first kernel.
fn refused(src: &str) -> String {
    let kernels = crate::parse(src).unwrap();
    let base = phobos_base::context::Context::default();
    let target = crate::codegen::target::build_target(&base);
    let (mut ir, mut report) = crate::ir::build::build(&base, target, &kernels[0]).unwrap_or_else(|e| panic!("{e:#}"));
    match super::run(&mut ir, &mut report, &target, &kernels[0]) {
        Ok(()) => panic!("expected the passes to refuse:\n{ir}"),
        Err(e) => format!("{e:#}"),
    }
}

#[test]
fn grid_barriers_pins_the_count() {
    let ok = MLP.replace("@persistent(BLOCKS)", "@persistent(BLOCKS)\n@grid_barriers(1)");
    assert_eq!(passed(&ok).matches("grid_barrier %").count(), 1);
    let err = refused(&MLP.replace("@persistent(BLOCKS)", "@persistent(BLOCKS)\n@grid_barriers(0)"));
    assert!(err.contains("`@grid_barriers(0)`: the kernel has 1, 0 written and 1 placed"), "{err}");
}

#[test]
fn shared_asserts_and_global_vetoes_holding() {
    let ok = MLP.replace("@persistent(BLOCKS)", "@persistent(BLOCKS)\n@shared(H)");
    assert!(passed(&ok).contains("@shared{align 256} = alloc"));
    let err = refused(&MLP.replace("@persistent(BLOCKS)", "@persistent(BLOCKS)\n@shared(V)"));
    assert!(err.contains("`@shared(V)`: `V` cannot live in shared memory"), "{err}");
    let global = passed(&MLP.replace("@persistent(BLOCKS)", "@persistent(BLOCKS)\n@global(H)"));
    let signature = global.lines().next().expect("the kernel line");
    assert!(signature.contains("%H.") && !global.contains("= alloc"), "{global}");
}

#[test]
fn a_dynamic_buffer_still_orders_its_blocks() {
    // The same exchange through a buffer of dynamic rows.
    let src = MLP.replace("var V: tensor<f32>[1, 256]\n", "").replace(
        "W: tensor<f32>[256, 256]",
        "W: tensor<f32>[256, 256], V: tensor<f32>[N, 256]",
    );
    let ir = passed(&src);
    assert_eq!(ir.matches("grid_barrier %").count(), 1, "{ir}");
}

#[test]
fn a_span_past_its_unit_is_another_blocks() {
    // Unit `u` reads its own eight elements and the next unit's.
    let src = MLP
        .replace(" + V[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]", "")
        .replace("H[0, ctx.thread] = X[0, ctx.thread] * 2.0", "H[0, ctx.thread] = 2.0")
        .replace("X[0 :+ 1, u * 8 :+ 8] += V[0 :+ 1, u * 8 :+ 8] * 3.0", "X[0 :+ 1, u * 4 :+ 4] += rowsum(V[0 :+ 1, u * 8 :+ 16])");
    assert_eq!(passed(&src).matches("grid_barrier %").count(), 1);
}

#[test]
fn grid_barriers_counts_the_back_edge() {
    // The layer loop's two barriers, each counted once.
    let ir = passed(
        "@launch(256)
@persistent(BLOCKS)
@grid_barriers(2)
@autotune(BLOCKS in [8])
kernel k(X: tensor<f32>[1, 256], Y: tensor<f32>[1, 256]) {
  for l in range(0, 2) {
    for i in range(0, 4) {
      let u = ctx.block.x + i * ctx.blocks.x
      if u < 32 {
        Y[0 :+ 1, u * 8 :+ 8] = X[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]
      }
    }
    for i in range(0, 4) {
      let u = ctx.block.x + i * ctx.blocks.x
      if u < 32 {
        X[0 :+ 1, u * 8 :+ 8] = Y[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]
      }
    }
  }
}",
    );
    assert_eq!(ir.matches("grid_barrier %").count(), 2);
}

#[test]
fn the_launch_report_lists_what_the_launch_supplies() {
    let report = crate::launch_report(&phobos_base::context::Context::default(), MLP, "k").unwrap();
    let names: Vec<_> = report.scratch.iter().map(|s| (s.name.as_str(), s.dims.clone())).collect();
    assert_eq!(names, [("V", vec![1, 256])]);
    assert!(report.barrier_word);
    // On the line of the second unit loop.
    let line = MLP.lines().position(|l| l.contains("X[0 :+ 1, u * 8 :+ 8] +=")).unwrap() as u32 + 1;
    assert_eq!(report.grid_barriers, [line - 3]);
}

#[test]
fn a_placed_barrier_takes_the_place_of_a_written_cta_barrier() {
    let src = MLP.replace("  }\n  for i in range(0, 4) {\n    let u = p + i * ctx.blocks.x\n    if u < 32 {\n      X", "  }\n  barrier()\n  for i in range(0, 4) {\n    let u = p + i * ctx.blocks.x\n    if u < 32 {\n      X");
    assert!(src.matches("barrier()").count() == 2, "{src}");
    let ir = passed(&src);
    assert_eq!(ir.matches("grid_barrier %").count(), 1, "{ir}");
    assert_eq!(ir.lines().filter(|l| l.trim() == "barrier").count(), 1, "only the first CTA barrier is left:\n{ir}");
}

/// [`MLP`] with grid-stride loops in place of the guarded ones.
fn strided(src: &str) -> String {
    src.replace(
        "  for i in range(0, 4) {\n    let u = p + i * ctx.blocks.x\n    if u < 32 {\n",
        "  for u in range(p, 32, ctx.blocks.x) {\n    if true {\n",
    )
}

#[test]
fn a_grid_stride_loop_hands_out_units() {
    let src = strided(MLP);
    assert!(!src.contains("let u"), "{src}");
    let ir = passed(&src);
    assert_eq!(ir.matches("grid_barrier %").count(), 1, "{ir}");
    // Reading only its own unit, the second loop needs none.
    let own = strided(
        &MLP.replace(" + V[0 :+ 1, ((u + 1) % 32) * 8 :+ 8]", "")
            .replace("H[0, ctx.thread] = X[0, ctx.thread] * 2.0", "H[0, ctx.thread] = 2.0"),
    );
    assert!(!passed(&own).contains("grid_barrier"));
}
