//! What the compiler tells a kernel's author, pointed at the source.
//!
//! A [`Diagnostic`] names spans of one [`SourceFile`]; [`Diagnostics`] is the
//! error a compile returns, carrying the file, so it renders anywhere: rustc's
//! layout on a terminal ([`Diagnostics::render`]), one JSON object per line for
//! a tool ([`Diagnostics::to_json`]), or `file:line:col: message` through
//! `Display` for a caller that only logs it.

mod json;
mod paint;
mod render;
mod source;

use std::sync::Arc;

pub use paint::Class;
pub use phobos_base::term::Colors;
pub use source::SourceFile;

/// The most errors one run reports; past it the rest are dropped.
pub const MAX_ERRORS: usize = 20;

/// A range of a source's bytes, `hi` exclusive.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub fn new(lo: u32, hi: u32) -> Self {
        Span { lo, hi: hi.max(lo) }
    }

    /// From the start of `self` to the end of `other`.
    pub fn to(self, other: Span) -> Span {
        Span::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

impl Severity {
    fn word(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    pub span: Span,
    pub text: String,
    pub primary: bool,
}

/// A replacement of `span`'s bytes, what a help suggests typing.
#[derive(Debug, Clone, PartialEq)]
pub struct Edit {
    pub span: Span,
    pub replacement: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Help {
    pub text: String,
    pub edit: Option<Edit>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<&'static str>,
    pub message: String,
    pub labels: Vec<Label>,
    pub notes: Vec<String>,
    pub help: Vec<Help>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    fn new(severity: Severity, message: impl Into<String>) -> Self {
        Diagnostic {
            severity,
            code: None,
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
            help: Vec::new(),
        }
    }

    pub fn code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    /// The span that is wrong, and what is wrong there; `text` may be empty.
    pub fn primary(mut self, span: Span, text: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            text: text.into(),
            primary: true,
        });
        self
    }

    /// Another span the message involves.
    pub fn secondary(mut self, span: Span, text: impl Into<String>) -> Self {
        self.labels.push(Label {
            span,
            text: text.into(),
            primary: false,
        });
        self
    }

    pub fn note(mut self, text: impl Into<String>) -> Self {
        self.notes.push(text.into());
        self
    }

    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help.push(Help {
            text: text.into(),
            edit: None,
        });
        self
    }

    /// A help that shows the source with `span` replaced.
    pub fn suggest(mut self, text: impl Into<String>, span: Span, replacement: impl Into<String>) -> Self {
        self.help.push(Help {
            text: text.into(),
            edit: Some(Edit {
                span,
                replacement: replacement.into(),
            }),
        });
        self
    }

    /// Where the diagnostic points: its first primary label, else its first.
    pub fn span(&self) -> Option<Span> {
        self.labels
            .iter()
            .find(|l| l.primary)
            .or(self.labels.first())
            .map(|l| l.span)
    }
}

/// The diagnostics of one source, the error a compile fails with.
#[derive(Debug, Clone)]
pub struct Diagnostics {
    source: Arc<SourceFile>,
    list: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn new(source: Arc<SourceFile>, list: Vec<Diagnostic>) -> Self {
        Diagnostics { source, list }
    }

    pub fn source(&self) -> &Arc<SourceFile> {
        &self.source
    }

    pub fn list(&self) -> &[Diagnostic] {
        &self.list
    }

    pub fn errors(&self) -> usize {
        self.list.iter().filter(|d| d.severity == Severity::Error).count()
    }

    /// rustc's layout, every diagnostic and then a count, colored as `colors`
    /// allows.
    pub fn render(&self, colors: Colors) -> String {
        render::render_all(self, colors)
    }

    /// One JSON object per line, spans as byte ranges and as lines and
    /// columns.
    pub fn to_json(&self) -> String {
        json::to_json(self)
    }

    /// `name:line:col` for `span`, or `line:col` for a source with no name.
    pub fn location(&self, span: Span) -> String {
        let (line, col) = self.source.line_col(span.lo);
        match self.source.name() {
            "" => format!("{line}:{col}"),
            name => format!("{name}:{line}:{col}"),
        }
    }
}

impl std::fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, d) in self.list.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            if let Some(span) = d.span() {
                write!(f, "{}: ", self.location(span))?;
            }
            if d.severity != Severity::Error {
                write!(f, "{}: ", d.severity.word())?;
            }
            f.write_str(&d.message)?;
        }
        Ok(())
    }
}

impl std::error::Error for Diagnostics {}

/// `e` as rustc would show it: rendered if it is [`Diagnostics`], else its
/// message chain as one error with no location.
pub fn render_error(e: &anyhow::Error, colors: Colors) -> String {
    of_error(e).render(colors)
}

/// `e` as [`Diagnostics::to_json`] writes it, a plain error as one with no
/// location.
pub fn error_json(e: &anyhow::Error) -> String {
    of_error(e).to_json()
}

fn of_error(e: &anyhow::Error) -> std::borrow::Cow<'_, Diagnostics> {
    match e.downcast_ref::<Diagnostics>() {
        Some(ds) => std::borrow::Cow::Borrowed(ds),
        None => std::borrow::Cow::Owned(Diagnostics::new(
            SourceFile::new("", ""),
            vec![Diagnostic::error(format!("{e:#}"))],
        )),
    }
}

/// The candidate closest to `name` by edit distance, if it is close enough to
/// be what was meant: a third of the name's length, at least one edit.
pub fn similar<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let limit = (name.chars().count() / 3).max(1);
    candidates
        .into_iter()
        .filter(|c| *c != name)
        .map(|c| (edit_distance(name, c), c))
        .filter(|&(d, _)| d <= limit)
        .min_by_key(|&(d, c)| (d, c.len().abs_diff(name.len())))
        .map(|(_, c)| c)
}

/// Edits from `a` to `b`, a swap of two neighbours counting as one, since
/// that is the commonest typo.
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut d = vec![vec![0; b.len() + 1]; a.len() + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[a.len()][b.len()]
}

#[cfg(test)]
mod tests;
