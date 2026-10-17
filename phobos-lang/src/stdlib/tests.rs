use super::*;
use phobos_base::context::Context;
use phobos_base::term::{ColorChoice, stderr_colors};

const CALLS_SILU: &str = "kernel k(X: tensor<f32>[16, 64], Y: tensor<f32>[16, 64]) {
    Y[0 :+ 16, :] = silu(X[0 :+ 16, :])
}";

fn compile(stdlib: Stdlib, src: &str) -> anyhow::Result<String> {
    let context = Context { stdlib, ..Context::default() };
    crate::compile(&context, src)
}

fn custom(text: &str) -> Stdlib {
    Stdlib::Custom {
        name: "mine.ph".into(),
        text: text.into(),
    }
}

#[test]
fn the_shipped_library_defines_something_in_every_file() {
    let fns = builtin();
    for (name, _) in FILES {
        assert!(fns.iter().any(|f| f.file == *name), "nothing defined in {name}");
    }
}

#[test]
fn no_library_function_takes_a_built_in_name() {
    let taken: Vec<_> = builtin()
        .into_iter()
        .filter(|f| crate::ir::build::names::builtins().contains(&f.name))
        .map(|f| f.name)
        .collect();
    assert!(taken.is_empty(), "{taken:?}");
}

#[test]
fn a_program_function_shadows_the_library_one() {
    let own = format!("fn silu(x) = x * 2.0\n{CALLS_SILU}");
    let doubled = CALLS_SILU.replace("silu(X[0 :+ 16, :])", "X[0 :+ 16, :] * 2.0");
    assert_eq!(compile(Stdlib::Builtin, &own).unwrap(), compile(Stdlib::Builtin, &doubled).unwrap());
}

#[test]
fn off_leaves_only_the_program_and_custom_replaces_the_library() {
    let err = compile(Stdlib::Off, CALLS_SILU).unwrap_err();
    assert!(err.to_string().contains("unknown function `silu`"), "{err}");

    let doubled = CALLS_SILU.replace("silu(X[0 :+ 16, :])", "X[0 :+ 16, :] * 2.0");
    assert_eq!(
        compile(custom("fn silu(x) = x * 2.0"), CALLS_SILU).unwrap(),
        compile(Stdlib::Builtin, &doubled).unwrap()
    );
    let gelu = CALLS_SILU.replace("silu", "gelu");
    assert!(compile(custom("fn silu(x) = x"), &gelu).is_err(), "a custom library replaces the shipped one");
}

#[test]
fn a_custom_library_reports_its_own_errors() {
    let err = compile(custom("fn silu(x) = x *"), CALLS_SILU).unwrap_err();
    let shown = crate::diag::render_error(&err, stderr_colors(ColorChoice::Never));
    assert!(shown.contains("mine.ph:1:"), "{shown}");

    let err = compile(custom("fn f(x) = x\nfn f(y) = y"), CALLS_SILU).unwrap_err();
    assert!(err.to_string().contains("`f` is defined twice"), "{err}");

    let err = compile(custom("kernel k() {}"), CALLS_SILU).unwrap_err();
    assert!(err.to_string().contains("functions only"), "{err}");
}
