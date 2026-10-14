use std::sync::Arc;

/// A kernel source and where its lines start, so a byte offset turns into a
/// line and a column only when something is shown.
#[derive(Debug)]
pub struct SourceFile {
    name: String,
    text: String,
    line_starts: Vec<u32>,
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
        })
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

    /// The byte offset of a line and a column from 1, as the lexer counts
    /// them, clamped to the line.
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
