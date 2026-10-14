// Errors about ops, which the build places at the ops' spans.

use std::sync::Arc;

use super::build::Poisoned;
use super::{Ir, OpId};
use crate::diag::{Diagnostic, Diagnostics, SourceFile, Span};

/// An error about one op, placed at its span; with no op, or one without a
/// span, where the caller places it. Other ops it involves are labelled at
/// their own spans.
#[derive(Debug, Default)]
pub struct AtOp {
    pub op: Option<OpId>,
    pub message: String,
    /// What is wrong at the primary span; may be empty.
    pub label: String,
    pub others: Vec<(OpId, String)>,
    pub notes: Vec<String>,
    pub help: Vec<String>,
    /// A name written, and a known one it may have meant.
    pub instead: Option<(String, String)>,
}

impl AtOp {
    pub fn new(op: OpId, message: impl Into<String>) -> Self {
        AtOp {
            op: Some(op),
            message: message.into(),
            ..AtOp::default()
        }
    }

    /// An error the caller places.
    pub fn unplaced(message: impl Into<String>) -> Self {
        AtOp {
            message: message.into(),
            ..AtOp::default()
        }
    }

    pub fn label(mut self, text: impl Into<String>) -> Self {
        self.label = text.into();
        self
    }

    /// Labels `op`, another op the error involves.
    pub fn with(mut self, op: Option<OpId>, text: impl Into<String>) -> Self {
        if let Some(op) = op {
            self.others.push((op, text.into()));
        }
        self
    }

    pub fn note(mut self, text: impl Into<String>) -> Self {
        self.notes.push(text.into());
        self
    }

    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help.push(text.into());
        self
    }

    /// Suggests `near`, when there is one, in place of `written`.
    pub fn instead(mut self, written: &str, near: Option<&str>) -> Self {
        self.instead = near.map(|n| (written.to_string(), n.to_string()));
        self
    }
}

impl std::fmt::Display for AtOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AtOp {}

/// [`AtOp`] as an error.
pub fn at_op(op: OpId, message: impl Into<String>) -> anyhow::Error {
    AtOp::new(op, message).into()
}

/// `e` placed at `span` in `source`: an [`AtOp`] at its op's span, any
/// other error at `span`. One placed already, a [`Poisoned`] one, and any at
/// an empty span pass through, so the innermost place wins.
pub(crate) fn locate(source: &Arc<SourceFile>, ir: &Ir, e: anyhow::Error, span: Span) -> anyhow::Error {
    if e.is::<Diagnostics>() || e.is::<Poisoned>() {
        return e;
    }
    let span_of = |op: OpId| if ir.is_alive(op) { ir.span(op) } else { Span::default() };
    let Some(at) = e.downcast_ref::<AtOp>() else {
        if span == Span::default() {
            return e;
        }
        return Diagnostics::new(source.clone(), vec![Diagnostic::error(format!("{e:#}")).primary(span, "")]).into();
    };
    let span = match at.op.map(span_of) {
        Some(s) if s != Span::default() => s,
        _ => span,
    };
    if span == Span::default() {
        return e;
    }
    let mut d = Diagnostic::error(at.message.clone()).primary(span, at.label.clone());
    for (op, text) in &at.others {
        let s = span_of(*op);
        if s != Span::default() && s != span {
            d = d.secondary(s, text.clone());
        }
    }
    for note in &at.notes {
        d = d.note(note.clone());
    }
    for help in &at.help {
        d = d.help(help.clone());
    }
    if let Some((written, near)) = &at.instead {
        let text = &source.text()[span.lo as usize..span.hi as usize];
        if let Some(offset) = text.find(written.as_str()) {
            let lo = span.lo + offset as u32;
            let edit = Span::new(lo, lo + written.len() as u32);
            d = d.suggest(format!("a similar name exists: `{near}`"), edit, near.clone());
        }
    }
    Diagnostics::new(source.clone(), vec![d]).into()
}
