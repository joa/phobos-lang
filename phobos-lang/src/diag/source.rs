use std::sync::Arc;

use super::Span;

/// A kernel source and where its lines start, so a byte offset turns into a
/// line and a column only when something is shown.
pub struct SourceFile {
    name: String,
    text: String,
    line_starts: Vec<u32>,
    origin: Option<Origin>,
}

/// The source a generated kernel source was written from, another
/// language's, and where each construct of the one came from in the other.
/// Whatever reports a location reports it there.
pub struct Origin {
    pub file: Arc<SourceFile>,
    /// Each emitted construct's span and the span it was written from.
    pub map: Vec<(Span, Span)>,
    /// Operators the source spelled in a way that matters to a check,
    /// each at the span of the operator it was emitted as.
    pub spellings: Vec<(Span, Spelling)>,
}

/// How a front end spelled an operator, where one operator stands for two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spelling {
    /// Division that a source expects to give a fraction of integers, as
    /// Python's `/` does.
    TrueDiv,
    /// Division that a source expects to give a whole number, as Python's
    /// `//` does, toward minus infinity.
    FloorDiv,
}

/// The name and size, not the text, which a kernel's `Debug` would repeat.
impl std::fmt::Debug for SourceFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SourceFile({:?}, {} bytes)", self.name, self.text.len())
    }
}

impl SourceFile {
    /// `name` is what a location shows: a path, a generated kernel's label,
    /// or empty for a bare `line:col`.
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Arc<Self> {
        let text = text.into();
        let line_starts = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| i as u32 + 1))
            .collect();
        Arc::new(SourceFile {
            name: name.into(),
            text,
            line_starts,
            origin: None,
        })
    }

    /// A source generated from `origin.file`, where every location is shown.
    pub fn with_origin(name: impl Into<String>, text: impl Into<String>, mut origin: Origin) -> Arc<Self> {
        let file = Arc::into_inner(Self::new(name, text)).expect("a new file has one owner");
        // Innermost first, so the first entry holding a span is the closest.
        origin.map.sort_by_key(|(at, _)| at.hi - at.lo);
        Arc::new(SourceFile { origin: Some(origin), ..file })
    }

    pub fn origin(&self) -> Option<&Origin> {
        self.origin.as_ref()
    }

    /// The file a location in this one is shown in: the one it was
    /// generated from, or itself.
    pub fn display(self: &Arc<Self>) -> Arc<SourceFile> {
        self.origin.as_ref().map_or_else(|| self.clone(), |o| o.file.clone())
    }

    /// [`Self::display`], borrowed.
    pub fn shown(&self) -> &SourceFile {
        self.origin.as_ref().map_or(self, |o| &o.file)
    }

    /// `span` in [`Self::display`]'s bytes. A span inside a construct
    /// copied verbatim keeps its place within it; any other takes the whole
    /// construct's.
    pub fn locate(&self, span: Span) -> Span {
        let Some(origin) = &self.origin else { return span };
        let holds = |at: &Span| at.lo <= span.lo && span.hi <= at.hi;
        let Some(&(at, from)) = origin.map.iter().find(|(at, _)| holds(at)) else {
            // Outside every construct: the first written after it, else the
            // file's start.
            let next = origin.map.iter().filter(|(at, _)| at.lo >= span.lo).min_by_key(|(at, _)| at.lo);
            return next.map_or(Span::default(), |&(_, from)| Span::new(from.lo, from.lo));
        };
        let emitted = self.text.get(at.lo as usize..at.hi as usize);
        let written = origin.file.text.get(from.lo as usize..from.hi as usize);
        if emitted.is_some() && emitted == written {
            Span::new(from.lo + (span.lo - at.lo), from.lo + (span.hi - at.lo))
        } else {
            from
        }
    }

    /// Whether `span` lies in one construct copied verbatim, so an edit of
    /// it is an edit of the source it was written from.
    pub fn verbatim(&self, span: Span) -> bool {
        let Some(origin) = &self.origin else { return true };
        origin.map.iter().find(|(at, _)| at.lo <= span.lo && span.hi <= at.hi).is_some_and(|&(at, from)| {
            let emitted = self.text.get(at.lo as usize..at.hi as usize);
            emitted.is_some() && emitted == origin.file.text.get(from.lo as usize..from.hi as usize)
        })
    }

    /// The spelling of an operator emitted between `lo` and `hi`, the end
    /// of one operand and the start of the other.
    pub fn spelling_between(&self, lo: u32, hi: u32) -> Option<Spelling> {
        let origin = self.origin.as_ref()?;
        origin.spellings.iter().find(|(at, _)| lo <= at.lo && at.hi <= hi).map(|&(_, s)| s)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The number of lines, a last line without a newline included.
    pub fn line_count(&self) -> u32 {
        let trailing = self.text.ends_with('\n') || self.text.is_empty();
        self.line_starts.len() as u32 - u32::from(trailing && self.line_starts.len() > 1)
    }

    /// The line, from 1, that holds `offset`. The end of a file that ends in
    /// a newline counts as its last line.
    pub fn line_of(&self, offset: u32) -> u32 {
        let offset = offset.min(self.text.len() as u32);
        let line = self.line_starts.partition_point(|&s| s <= offset) as u32;
        line.min(self.line_count()).max(1)
    }

    /// The line and column, both from 1, of `offset`; the column counts
    /// characters.
    pub fn line_col(&self, offset: u32) -> (u32, u32) {
        let line = self.line_of(offset);
        let start = self.line_start(line) as usize;
        let end = (offset as usize).clamp(start, start + self.line_text(line).len());
        let col = self.text.get(start..end).map_or(end - start, |s| s.chars().count());
        (line, col as u32 + 1)
    }

    /// The byte offset line `line` starts at.
    pub fn line_start(&self, line: u32) -> u32 {
        self.line_starts[(line.max(1) - 1) as usize]
    }

    /// Line `line`'s text, without its line ending.
    pub fn line_text(&self, line: u32) -> &str {
        let start = self.line_start(line) as usize;
        let end = self
            .line_starts
            .get(line as usize)
            .map_or(self.text.len(), |&s| s as usize);
        self.text[start..end].trim_end_matches(['\n', '\r'])
    }

    /// The byte offset of a line and a column, both from 1 and counted as
    /// [`Self::line_col`] counts them, clamped to the line.
    pub fn offset(&self, line: u32, col: u32) -> u32 {
        let line = line.clamp(1, self.line_starts.len() as u32);
        let start = self.line_start(line);
        let text = self.line_text(line);
        let within = text
            .char_indices()
            .nth(col.max(1) as usize - 1)
            .map_or(text.len(), |(i, _)| i);
        start + within as u32
    }
}
