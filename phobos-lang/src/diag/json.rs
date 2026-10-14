// One JSON object per diagnostic and line, for an editor or a script:
// spans as byte ranges and as 1-based lines and columns, and each help's
// edit, so a tool can apply it.

use std::fmt::Write as _;

use super::{Diagnostics, Span};

pub(super) fn to_json(ds: &Diagnostics) -> String {
    let mut out = String::new();
    for d in &ds.list {
        out.push('{');
        let _ = write!(out, "\"severity\":{}", quote(d.severity.word()));
        let _ = write!(out, ",\"code\":{}", d.code.map_or("null".into(), quote));
        let _ = write!(out, ",\"message\":{}", quote(&d.message));
        out.push_str(",\"file\":");
        out.push_str(&quote(ds.source.name()));
        out.push_str(",\"labels\":[");
        for (i, l) in d.labels.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{{},\"primary\":{},\"text\":{}}}",
                span(ds, l.span),
                l.primary,
                quote(&l.text)
            );
        }
        out.push_str("],\"notes\":[");
        out.push_str(&d.notes.iter().map(|n| quote(n)).collect::<Vec<_>>().join(","));
        out.push_str("],\"help\":[");
        for (i, h) in d.help.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "{{\"text\":{}", quote(&h.text));
            if let Some(e) = &h.edit {
                let _ = write!(out, ",\"edit\":{{{},\"replacement\":{}}}", span(ds, e.span), quote(&e.replacement));
            }
            out.push('}');
        }
        out.push_str("]}\n");
    }
    out
}

fn span(ds: &Diagnostics, s: Span) -> String {
    let (line, col) = ds.source.line_col(s.lo);
    let (end_line, end_col) = ds.source.line_col(s.hi);
    format!(
        "\"lo\":{},\"hi\":{},\"line\":{line},\"col\":{col},\"end_line\":{end_line},\"end_col\":{end_col}",
        s.lo, s.hi
    )
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
