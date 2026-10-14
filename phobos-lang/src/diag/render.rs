// rustc's layout: a headline, the location, the source lines the labels sit
// on with a gutter of line numbers, an underline per label (`^` for the
// primary, `-` for the rest), a margin of `/`, `|` and `_` for a label that
// spans lines, then the notes and helps, and a help's suggested source last.

use super::paint::{Class, Paint, Role};
use super::{Colors, Diagnostic, Diagnostics, Edit, Severity, SourceFile, Span};

const TAB_WIDTH: usize = 4;
/// The display columns of a source line shown whole; a longer one is cut
/// to a window around its labels.
const MAX_LINE_WIDTH: usize = 120;
/// Columns kept left of a cut line's first label.
const CUT_CONTEXT: usize = 20;
const ELLIPSIS: &str = "...";

/// Every diagnostic of `ds`, then rustc's closing count.
pub(super) fn render_all(ds: &Diagnostics, colors: Colors) -> String {
    let paint = Paint { colors };
    let classes = match colors {
        Colors::Off => Vec::new(),
        _ => crate::lexer::classify(ds.source.text()),
    };
    let mut out = String::new();
    for d in &ds.list {
        let r = Renderer {
            file: &ds.source,
            paint,
            classes: &classes,
            out: &mut out,
        };
        r.diagnostic(d, ds);
        out.push('\n');
    }
    let errors = ds.errors();
    let warnings = ds.list.len() - errors;
    let emitted = format!("{} emitted", count(warnings, "warning"));
    if errors > 0 {
        let mut text = format!("aborting due to {}", count(errors, "previous error"));
        if warnings > 0 {
            text = format!("{text}; {emitted}");
        }
        headline(&paint, Severity::Error, None, &text, &mut out);
    } else if warnings > 0 {
        headline(&paint, Severity::Warning, None, &emitted, &mut out);
    }
    out
}

fn count(n: usize, what: &str) -> String {
    format!("{n} {what}{}", if n == 1 { "" } else { "s" })
}

fn headline(paint: &Paint, severity: Severity, code: Option<&str>, message: &str, out: &mut String) {
    let role = severity_role(severity);
    paint.push(out, role, severity.word());
    if let Some(code) = code {
        paint.push(out, role, &format!("[{code}]"));
    }
    paint.push(out, Role::Message, &format!(": {message}"));
    out.push('\n');
}

fn severity_role(severity: Severity) -> Role {
    match severity {
        Severity::Error => Role::Error,
        Severity::Warning => Role::Warning,
    }
}

/// A label placed on the shown lines: lines from 1, display columns from 0,
/// `col1` exclusive.
struct Mark<'a> {
    line0: u32,
    col0: usize,
    line1: u32,
    col1: usize,
    text: &'a str,
    role: Role,
    primary: bool,
    /// The margin column of a label that spans lines.
    depth: usize,
}

impl Mark<'_> {
    fn multi(&self) -> bool {
        self.line1 > self.line0
    }
}

/// The display columns of a line that are drawn: `start..end`, each cut
/// end marked by an ellipsis.
#[derive(Clone, Copy)]
struct Window {
    start: usize,
    end: usize,
    width: usize,
}

impl Window {
    fn lead(self) -> usize {
        if self.start > 0 { ELLIPSIS.len() } else { 0 }
    }

    /// Where display column `col` of the line is drawn.
    fn at(self, col: usize) -> usize {
        col.saturating_sub(self.start) + self.lead()
    }
}

/// One drawn row after the gutter: characters with their roles.
#[derive(Default)]
struct Row(Vec<(char, Role)>);

impl Row {
    fn put(&mut self, at: usize, c: char, role: Role) {
        if self.0.len() <= at {
            self.0.resize(at + 1, (' ', Role::Plain));
        }
        self.0[at] = (c, role);
    }

    fn put_str(&mut self, at: usize, s: &str, role: Role) {
        for (i, c) in s.chars().enumerate() {
            self.put(at + i, c, role);
        }
    }
}

struct Renderer<'a> {
    file: &'a SourceFile,
    paint: Paint,
    classes: &'a [(Span, Class)],
    out: &'a mut String,
}

impl Renderer<'_> {
    fn diagnostic(mut self, d: &Diagnostic, ds: &Diagnostics) {
        headline(&self.paint, d.severity, d.code, &d.message, self.out);
        let mut marks = self.marks(d);
        let lines = shown_lines(&marks);
        let edit_lines = d
            .help
            .iter()
            .filter_map(|h| h.edit.as_ref())
            .map(|e| self.file.line_of(e.span.hi) + e.replacement.matches('\n').count() as u32);
        let last = lines.iter().copied().chain(edit_lines).max().unwrap_or(1);
        let width = last.to_string().len();
        if let Some(span) = d.span() {
            self.chrome(&format!("{}--> ", " ".repeat(width)));
            self.out.push_str(&ds.location(span));
            self.out.push('\n');
            self.gutter(width, "", '|', Row::default());
            self.snippet(&mut marks, &lines, width);
        }
        let plain_help = d.help.iter().filter(|h| h.edit.is_none());
        let extras: Vec<(&str, &str)> = d
            .notes
            .iter()
            .map(|n| ("note", n.as_str()))
            .chain(plain_help.map(|h| ("help", h.text.as_str())))
            .collect();
        if !extras.is_empty() && d.span().is_some() {
            self.gutter(width, "", '|', Row::default());
        }
        for (word, text) in extras {
            let role = if word == "note" { Role::Note } else { Role::Help };
            self.chrome(&format!("{} = ", " ".repeat(width)));
            self.paint.push(self.out, role, word);
            let indent = " ".repeat(width + 3 + word.len() + 2);
            for (i, line) in text.lines().enumerate() {
                self.out.push_str(if i == 0 { ": " } else { &indent });
                self.out.push_str(line);
                self.out.push('\n');
            }
        }
        for help in &d.help {
            if let Some(edit) = &help.edit {
                self.paint.push(self.out, Role::Help, "help");
                self.out.push_str(&format!(": {}\n", help.text));
                self.suggestion(edit, width);
            }
        }
    }

    fn chrome(&mut self, s: &str) {
        self.paint.push(self.out, Role::Chrome, s);
    }

    /// A line number, or blanks, then `bar` and the row.
    fn gutter(&mut self, width: usize, number: &str, bar: char, row: Row) {
        self.chrome(&format!("{number:>width$} {bar}"));
        let end = row.0.iter().rposition(|&(c, _)| c != ' ').map_or(0, |i| i + 1);
        if end > 0 {
            self.out.push(' ');
        }
        let mut cells = row.0[..end].iter().peekable();
        while let Some(&(c, role)) = cells.next() {
            let mut run = String::from(c);
            while let Some(&(c, _)) = cells.next_if(|&&(_, r)| r == role) {
                run.push(c);
            }
            self.paint.push(self.out, role, &run);
        }
        self.out.push('\n');
    }

    /// `d`'s labels placed on lines and display columns, those that span
    /// lines numbered by margin column.
    fn marks<'d>(&self, d: &'d Diagnostic) -> Vec<Mark<'d>> {
        let text = self.file.text().as_bytes();
        let len = text.len() as u32;
        let mut depth = 0;
        let mut marks: Vec<Mark<'d>> = d
            .labels
            .iter()
            .map(|l| {
                let lo = l.span.lo.min(len);
                let mut hi = l.span.hi.clamp(lo, len);
                while hi > lo && matches!(text[hi as usize - 1], b'\n' | b'\r') {
                    hi -= 1;
                }
                let (line0, col0) = self.at(lo);
                let (line1, mut col1) = self.at(hi);
                if line1 == line0 && col1 <= col0 {
                    col1 = col0 + 1;
                }
                let role = if l.primary { severity_role(d.severity) } else { Role::Secondary };
                Mark { line0, col0, line1, col1, text: &l.text, role, primary: l.primary, depth: 0 }
            })
            .collect();
        for m in marks.iter_mut().filter(|m| m.multi()) {
            m.depth = depth;
            depth += 1;
        }
        marks
    }

    /// The line and display column of `offset`.
    fn at(&self, offset: u32) -> (u32, usize) {
        let line = self.file.line_of(offset);
        let start = self.file.line_start(line);
        let text = self.file.line_text(line);
        let within = ((offset - start.min(offset)) as usize).min(text.len());
        (line, display_width(&text[..within]))
    }

    fn snippet(&mut self, marks: &mut [Mark], lines: &[u32], width: usize) {
        let margin = 2 * marks.iter().filter(|m| m.multi()).count();
        let mut prev: Option<u32> = None;
        for &line in lines {
            if prev.is_some_and(|p| line > p + 1) {
                self.chrome("...\n");
            }
            prev = Some(line);
            let mut row = Row::default();
            for m in marks.iter().filter(|m| m.multi()) {
                let starts_here = line == m.line0 && self.starts_line(m);
                if starts_here {
                    row.put(2 * m.depth, '/', m.role);
                } else if m.line0 < line && line <= m.line1 {
                    row.put(2 * m.depth, '|', m.role);
                }
            }
            let window = self.window(marks, line);
            self.source_row(&mut row, margin, line, window);
            self.gutter(width, &line.to_string(), '|', row);
            for r in self.annotations(marks, line, margin, window) {
                self.gutter(width, "", '|', r);
            }
        }
    }

    /// Whether a label spanning lines starts at its line's first
    /// non-blank, which rustc draws as a `/` in the margin.
    fn starts_line(&self, m: &Mark) -> bool {
        let text = self.file.line_text(m.line0);
        m.col0 <= display_width(&text[..text.len() - text.trim_start().len()])
    }

    /// What of line `line` is drawn: all of it, or for a long line the
    /// stretch its labels sit on.
    fn window(&self, marks: &[Mark], line: u32) -> Window {
        let width = display_width(self.file.line_text(line));
        if width <= MAX_LINE_WIDTH {
            return Window { start: 0, end: width, width };
        }
        let cols = marks.iter().flat_map(|m| {
            let starts = (m.line0 == line).then_some(m.col0);
            let ends = (m.line1 == line).then_some(m.col1);
            starts.into_iter().chain(ends)
        });
        let (lo, hi) = cols.fold((usize::MAX, 0), |(lo, hi), c| (lo.min(c), hi.max(c)));
        let start = if lo == usize::MAX || lo <= CUT_CONTEXT + ELLIPSIS.len() { 0 } else { lo - CUT_CONTEXT };
        let end = (start + MAX_LINE_WIDTH).max(hi + CUT_CONTEXT).min(width);
        Window { start, end, width }
    }

    /// Line `line`'s text into `row` after the margin, highlighted, cut to
    /// `window`.
    fn source_row(&self, row: &mut Row, margin: usize, line: u32, window: Window) {
        let start = self.file.line_start(line);
        let text = self.file.line_text(line);
        let mut whole = Row::default();
        put_source(&mut whole, 0, text, start, self.classes);
        let mut at = margin;
        if window.start > 0 {
            row.put_str(at, ELLIPSIS, Role::Chrome);
            at += ELLIPSIS.len();
        }
        for &(c, role) in whole.0.get(window.start..window.end).unwrap_or_default() {
            row.put(at, c, role);
            at += 1;
        }
        if window.end < window.width {
            row.put_str(at, ELLIPSIS, Role::Chrome);
        }
    }

    /// The rows under line `line`: where a spanning label starts, the
    /// underlines and texts of the labels on it, and where a spanning label
    /// ends.
    fn annotations(&self, marks: &[Mark], line: u32, margin: usize, window: Window) -> Vec<Row> {
        let col = |c: usize| margin + window.at(c);
        let below = |m: &Mark| {
            m.multi() && line < m.line1 && (m.line0 < line || (m.line0 == line && self.starts_line(m)))
        };
        let rails = |row: &mut Row| {
            for m in marks.iter().filter(|m| below(m)) {
                row.put(2 * m.depth, '|', m.role);
            }
        };
        let mut rows = Vec::new();
        for m in marks.iter().filter(|m| m.multi() && m.line0 == line && !self.starts_line(m)) {
            let mut row = Row::default();
            rails(&mut row);
            for at in 2 * m.depth + 1..col(m.col0) {
                row.put(at, '_', m.role);
            }
            row.put(col(m.col0), if m.primary { '^' } else { '-' }, m.role);
            rows.push(row);
        }
        let mut singles: Vec<&Mark> = marks.iter().filter(|m| !m.multi() && m.line0 == line).collect();
        if !singles.is_empty() {
            // Wider spans first, so one nested in another still shows; of
            // two the same, the primary's `^` wins.
            singles.sort_by_key(|m| (std::cmp::Reverse(m.col1 - m.col0), m.primary, m.col0));
            let mut underline = Row::default();
            rails(&mut underline);
            // A line not cut on the right keeps a label just past its end.
            let limit = if window.end < window.width { window.end } else { usize::MAX };
            for m in &singles {
                for c in m.col0.max(window.start)..m.col1.min(limit) {
                    underline.put(col(c), if m.primary { '^' } else { '-' }, m.role);
                }
            }
            singles.sort_by_key(|m| (m.col0, m.col1));
            let last = singles[singles.len() - 1];
            let reach = singles.iter().map(|m| m.col1).max().unwrap_or(0);
            let inline = (!last.text.is_empty() && last.col1 == reach).then_some(last);
            if let Some(m) = inline {
                underline.put_str(col(m.col1) + 1, m.text, m.role);
            }
            rows.push(underline);
            let mut pending: Vec<&Mark> = singles
                .iter()
                .rev()
                .filter(|m| !m.text.is_empty() && !inline.is_some_and(|i| std::ptr::eq(**m, i)))
                .copied()
                .collect();
            if !pending.is_empty() {
                let mut connector = Row::default();
                rails(&mut connector);
                for m in &pending {
                    connector.put(col(m.col0), '|', m.role);
                }
                rows.push(connector);
                while !pending.is_empty() {
                    let m = pending.remove(0);
                    let mut row = Row::default();
                    rails(&mut row);
                    for left in &pending {
                        row.put(col(left.col0), '|', left.role);
                    }
                    row.put_str(col(m.col0), m.text, m.role);
                    rows.push(row);
                }
            }
        }
        for m in marks.iter().filter(|m| m.multi() && m.line1 == line) {
            let mut row = Row::default();
            rails(&mut row);
            row.put(2 * m.depth, '|', m.role);
            let caret = col(m.col1.max(1) - 1);
            for at in 2 * m.depth + 1..caret {
                row.put(at, '_', m.role);
            }
            row.put(caret, if m.primary { '^' } else { '-' }, m.role);
            if !m.text.is_empty() {
                row.put_str(caret + 2, m.text, m.role);
            }
            rows.push(row);
        }
        rows
    }

    /// The source as `edit` leaves it, lines it adds marked `+`, lines it
    /// changes `~`, and a one-line edit's new text underlined.
    fn suggestion(&mut self, edit: &Edit, width: usize) {
        let file = self.file;
        let len = file.text().len() as u32;
        let (lo, hi) = (edit.span.lo.min(len), edit.span.hi.clamp(edit.span.lo.min(len), len));
        let (first, last) = (file.line_of(lo), file.line_of(hi));
        let start = file.line_start(first) as usize;
        let end = file.line_start(last) as usize + file.line_text(last).len();
        let text = file.text();
        let old: Vec<&str> = text[start..end].split('\n').collect();
        let new = format!("{}{}{}", &text[start..lo as usize], edit.replacement, &text[hi as usize..end]);
        let classes = match self.paint.colors {
            Colors::Off => Vec::new(),
            _ => crate::lexer::classify(&new),
        };
        let new_lines: Vec<&str> = new.split('\n').collect();
        let same_head = old.iter().zip(&new_lines).take_while(|(a, b)| a == b).count();
        let same_tail = old[same_head..]
            .iter()
            .rev()
            .zip(new_lines[same_head..].iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        let changed = same_head..new_lines.len() - same_tail;
        let marker = if old.len() - same_head - same_tail == 0 { '+' } else { '~' };
        self.gutter(width, "", '|', Row::default());
        let mut offset = 0u32;
        for (i, line) in new_lines.iter().enumerate() {
            let mut row = Row::default();
            put_source(&mut row, 0, line, offset, &classes);
            let bar = if changed.contains(&i) && new_lines.len() > 1 { marker } else { '|' };
            self.gutter(width, &(first + i as u32).to_string(), bar, row);
            offset += line.len() as u32 + 1;
        }
        if new_lines.len() == 1 && !edit.replacement.is_empty() {
            let at = display_width(&text[start..lo as usize]);
            let mut row = Row::default();
            let mark = if lo == hi { '+' } else { '~' };
            for col in at..at + display_width(&edit.replacement) {
                row.put(col, mark, Role::Help);
            }
            self.gutter(width, "", '|', row);
        }
    }
}

/// The lines a snippet shows: every label's first and last, the lines
/// inside a label of up to three, and a single line between two shown ones.
fn shown_lines(marks: &[Mark]) -> Vec<u32> {
    let mut lines: Vec<u32> = Vec::new();
    for m in marks {
        lines.extend([m.line0, m.line1]);
        if m.line1 - m.line0 <= 4 {
            lines.extend(m.line0 + 1..m.line1);
        }
    }
    lines.sort_unstable();
    lines.dedup();
    let mut filled = Vec::with_capacity(lines.len());
    for (i, &line) in lines.iter().enumerate() {
        if i > 0 && line == lines[i - 1] + 2 {
            filled.push(line - 1);
        }
        filled.push(line);
    }
    filled
}

/// `text`, which starts at byte `base` of what `classes` covers, into `row`
/// from column `at`, tabs expanded.
fn put_source(row: &mut Row, at: usize, text: &str, base: u32, classes: &[(Span, Class)]) {
    let mut col = at;
    let mut next = classes.partition_point(|(s, _)| s.hi <= base);
    for (i, c) in text.char_indices() {
        let offset = base + i as u32;
        while next < classes.len() && classes[next].0.hi <= offset {
            next += 1;
        }
        let role = classes
            .get(next)
            .filter(|(s, _)| s.lo <= offset)
            .map_or(Role::Plain, |&(_, class)| Role::Source(class));
        if c == '\t' {
            for _ in 0..TAB_WIDTH {
                row.put(col, ' ', role);
                col += 1;
            }
        } else {
            row.put(col, c, role);
            col += 1;
        }
    }
}

fn display_width(s: &str) -> usize {
    s.chars().map(|c| if c == '\t' { TAB_WIDTH } else { 1 }).sum()
}
