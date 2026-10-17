// The standard library: `fn` definitions every source can call without
// defining them, shipped as the Phobos sources in `stdlib/`. A function is
// inlined where it is called, so one no kernel calls costs nothing, and a
// program's own function of the same name shadows it. `Context::stdlib`
// swaps the library for another source or turns it off.

use std::sync::{Arc, OnceLock};

use phobos_base::context::Stdlib;

use crate::diag::{Diagnostic, Diagnostics, SourceFile};
use crate::parser::{Function, Parser};

/// The shipped library, one file a topic, in the order the docs list them.
pub const FILES: &[(&str, &str)] = &[
    ("std/math.ph", include_str!("stdlib/math.ph")),
    ("std/activation.ph", include_str!("stdlib/activation.ph")),
    ("std/reduce.ph", include_str!("stdlib/reduce.ph")),
    ("std/norm.ph", include_str!("stdlib/norm.ph")),
    ("std/attention.ph", include_str!("stdlib/attention.ph")),
    ("std/linalg.ph", include_str!("stdlib/linalg.ph")),
    ("std/quant.ph", include_str!("stdlib/quant.ph")),
    ("std/loss.ph", include_str!("stdlib/loss.ph")),
    ("std/random.ph", include_str!("stdlib/random.ph")),
];

/// One function of a library, as it is declared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub file: String,
    pub name: String,
    pub params: Vec<String>,
}

/// The functions `choice` puts in scope.
pub(crate) fn prelude(choice: &Stdlib) -> anyhow::Result<Arc<[Function]>> {
    static BUILTIN: OnceLock<Arc<[Function]>> = OnceLock::new();
    Ok(match choice {
        Stdlib::Builtin => BUILTIN
            .get_or_init(|| {
                let files = FILES.iter().map(|&(name, text)| (name, text));
                library(files).unwrap_or_else(|e| panic!("the shipped standard library does not parse: {e}"))
            })
            .clone(),
        Stdlib::Off => Arc::from([]),
        Stdlib::Custom { name, text } => library([(name.as_str(), &**text)])?,
    })
}

/// Every function the shipped library defines, file by file.
pub fn builtin() -> Vec<Signature> {
    signatures(&Stdlib::Builtin).expect("the shipped standard library parses")
}

/// Every function `choice` defines.
pub fn signatures(choice: &Stdlib) -> anyhow::Result<Vec<Signature>> {
    Ok(prelude(choice)?
        .iter()
        .map(|f| Signature {
            file: f.file.clone(),
            name: f.name.clone(),
            params: f.params.clone(),
        })
        .collect())
}

/// The functions of `files`, each a library. A name defined twice, in one
/// file or two, is an error.
fn library<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> anyhow::Result<Arc<[Function]>> {
    let mut out: Vec<Function> = Vec::new();
    for (name, text) in files {
        let file = SourceFile::new(name, text);
        let fail = |list: Vec<Diagnostic>| anyhow::Error::new(Diagnostics::new(file.clone(), list));
        let toks = crate::lexer::Lexer::new(file.text()).tokenize().map_err(fail)?;
        let fns = Parser::new(toks, file.clone()).parse_library().map_err(fail)?;
        for f in fns {
            if let Some(seen) = out.iter().find(|g| g.name == f.name) {
                let d = Diagnostic::error(format!("function `{}` is defined twice, here and in {}", f.name, seen.file))
                    .primary(f.span, "defined again");
                return Err(Diagnostics::new(file.clone(), vec![d]).into());
            }
            out.push(Function {
                file: name.to_string(),
                ..f
            });
        }
    }
    Ok(out.into())
}

#[cfg(test)]
mod tests;
