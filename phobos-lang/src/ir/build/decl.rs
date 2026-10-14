// Where names were declared, which an error about a use of one points at.

use anyhow::anyhow;

use super::Build;
use crate::ast::Type as AstType;
use crate::diag::Span;
use crate::ir::AtOp;

/// Where a name was declared: the statement or parameter, its type where
/// one is written and the binding has it, and its initializer where it has
/// one.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Decl {
    pub(crate) at: Span,
    pub(crate) ty: Span,
    pub(crate) value: Span,
}

impl Decl {
    /// `span`, where `ty` is written, when the binding takes that type: a
    /// tile's or a tensor's. A written scalar type is not what the build
    /// gives a scalar, which takes its initializer's.
    pub(crate) fn held(ty: Option<&AstType>, span: Span) -> Span {
        match ty {
            Some(AstType::Tile(..) | AstType::Tensor(..)) => span,
            _ => Span::default(),
        }
    }
}

impl Build {
    /// `e`, a store of the wrong type to `name`, with where `name`'s type
    /// came from.
    pub(crate) fn declared(&self, name: &str, e: anyhow::Error) -> anyhow::Error {
        if e.is::<crate::diag::Diagnostics>() || e.is::<AtOp>() {
            return e;
        }
        let Some(decl) = self.decl(name) else { return e };
        let at = AtOp::unplaced(format!("{e:#}"));
        if decl.ty != Span::default() {
            at.at(decl.ty, format!("`{name}` is declared with this type")).into()
        } else {
            at.at(decl.value, format!("`{name}` takes its type from this")).into()
        }
    }

    /// Where the binding `name` resolves to was declared.
    fn decl(&self, name: &str) -> Option<Decl> {
        let value = self.lookup(name)?.value()?;
        self.decls.get(&(name.to_string(), value)).copied()
    }

    /// The error for an assignment to `name`, which only a `var` takes.
    pub(crate) fn not_assignable(&self, name: &str) -> anyhow::Error {
        let decl = self.decl(name).unwrap_or_default();
        let keyword = Span::new(decl.at.lo, decl.at.lo + 3);
        let declared_let = self.source.text().get(keyword.lo as usize..keyword.hi as usize) == Some("let");
        if !declared_let {
            return anyhow!("`{name}` is not assignable");
        }
        AtOp::unplaced(format!("cannot assign to `{name}`, which is declared with `let`"))
            .label("cannot assign")
            .at(decl.at, "declared with `let` here")
            .suggest(format!("declare `{name}` with `var` to assign to it"), keyword, "var")
            .into()
    }
}
