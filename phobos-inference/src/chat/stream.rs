//! Reading a reply as it arrives.
//!
//! The markers that delimit reasoning and tool calls can be split across
//! tokens, so the parser holds back any tail that could still turn out to be
//! the start of one; see [`held_back`].

use serde_json::Value;

use super::ToolCall;
use super::dialect::{Dialect, FUNCTION_START, THINK_END, THINK_START};
use super::tools::{parse_function_block, parse_json_call, parse_minicpm_function};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum OutputEvent {
    Reasoning(String),
    Content(String),
    ToolCall(Box<ToolCall>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Reasoning,
    Content,
    ToolCall,
}

/// Splits a generation into reasoning, prose and tool calls as it arrives.
/// Incremental because of streaming: a chunk can end halfway through `</think>`
/// or `<tool_call>`, so the tail that could still become a marker is held back
/// until the next chunk decides it.
pub(crate) struct OutputParser {
    pub(crate) tools: Option<Value>,
    pub(crate) dialect: Dialect,
    pub(crate) mode: Mode,
    pub(crate) buf: String,
    pub(crate) calls: usize,
    pub(crate) at_content_start: bool,
}

impl OutputParser {
    pub(crate) fn new(tools: Option<Value>, thinking: bool, dialect: Dialect) -> OutputParser {
        OutputParser {
            tools,
            dialect,
            // The prompt prefills `<think>`, so a thinking pass starts inside
            // it and the model only ever emits the closing tag.
            mode: if thinking {
                Mode::Reasoning
            } else {
                Mode::Content
            },
            buf: String::new(),
            calls: 0,
            at_content_start: true,
        }
    }

    pub(crate) fn push(&mut self, chunk: &str) -> Vec<OutputEvent> {
        self.buf.push_str(chunk);
        self.drain(false)
    }

    /// Flush what is left, holding nothing back.
    pub(crate) fn finish(&mut self) -> Vec<OutputEvent> {
        self.drain(true)
    }

    pub(crate) fn drain(&mut self, flush: bool) -> Vec<OutputEvent> {
        let mut events = Vec::new();
        loop {
            match self.mode {
                Mode::Reasoning => {
                    if let Some(at) = self.buf.find(THINK_END) {
                        let text = self.buf[..at].to_string();
                        self.buf = self.buf[at + THINK_END.len()..].to_string();
                        self.mode = Mode::Content;
                        emit(&mut events, OutputEvent::Reasoning, text);
                        continue;
                    }
                    let take = self.buf.len() - held_back(&self.buf, THINK_END, flush);
                    let text: String = self.buf.drain(..take).collect();
                    emit(&mut events, OutputEvent::Reasoning, text);
                    break;
                }
                Mode::Content => {
                    let (start, _) = self.dialect.call_markers();
                    // A model whose template prefills no think block, as
                    // MiniCPM5's does not, opens one itself. Without this the
                    // reasoning arrives as prose with its tags intact.
                    let think_at = self.buf.find(THINK_START);
                    let call_at = self.buf.find(start);
                    let (marker, at, next) = match (think_at, call_at) {
                        (Some(think), Some(call)) if call < think => {
                            (start, Some(call), Mode::ToolCall)
                        }
                        (Some(think), _) => (THINK_START, Some(think), Mode::Reasoning),
                        (None, Some(call)) => (start, Some(call), Mode::ToolCall),
                        (None, None) => ("", None, Mode::Content),
                    };
                    if let Some(at) = at {
                        let text = self.buf[..at].to_string();
                        self.buf = self.buf[at + marker.len()..].to_string();
                        self.mode = next;
                        self.emit_content(&mut events, text);
                        continue;
                    }
                    // Either marker could still be growing at the tail.
                    let hold = held_back(&self.buf, start, flush).max(held_back(
                        &self.buf,
                        THINK_START,
                        flush,
                    ));
                    let take = self.buf.len() - hold;
                    let text: String = self.buf.drain(..take).collect();
                    self.emit_content(&mut events, text);
                    break;
                }
                Mode::ToolCall => {
                    let (start, end) = self.dialect.call_markers();
                    let Some(at) = self.buf.find(end) else {
                        if flush {
                            // An unterminated block never became a call.
                            let raw = std::mem::take(&mut self.buf);
                            self.mode = Mode::Content;
                            self.emit_content(&mut events, format!("{start}{raw}"));
                        }
                        break;
                    };
                    let raw = self.buf[..at].to_string();
                    self.buf = self.buf[at + end.len()..].to_string();
                    self.mode = Mode::Content;
                    match self.parse_call(&raw) {
                        Some(call) => {
                            self.calls += 1;
                            events.push(OutputEvent::ToolCall(Box::new(call)));
                        }
                        None => self.emit_content(&mut events, format!("{start}{raw}{end}")),
                    }
                    continue;
                }
            }
        }
        events
    }

    pub(crate) fn parse_call(&self, raw: &str) -> Option<ToolCall> {
        // MiniCPM5's start marker is part of the call, so its body still opens
        // with the name and must not be trimmed at the front.
        if self.dialect == Dialect::MiniCpm {
            return parse_minicpm_function(raw, self.tools.as_ref(), self.calls);
        }
        let raw = raw.trim();
        if raw.contains(FUNCTION_START) {
            return parse_function_block(raw, self.tools.as_ref(), self.calls);
        }
        parse_json_call(&serde_json::from_str::<Value>(raw).ok()?, self.calls)
    }

    /// Blank lines follow the `</think>` closer, and the caller must not see them.
    pub(crate) fn emit_content(&mut self, events: &mut Vec<OutputEvent>, text: String) {
        let text = if self.at_content_start {
            text.trim_start().to_string()
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        self.at_content_start = false;
        events.push(OutputEvent::Content(text));
    }
}

pub(crate) fn emit(events: &mut Vec<OutputEvent>, make: fn(String) -> OutputEvent, text: String) {
    if !text.is_empty() {
        events.push(make(text));
    }
}

/// How many bytes at the end of `buf` could still grow into `marker`.
pub(crate) fn held_back(buf: &str, marker: &str, flush: bool) -> usize {
    if flush {
        return 0;
    }
    (1..marker.len().min(buf.len() + 1))
        .rev()
        .find(|&k| {
            buf.is_char_boundary(buf.len() - k)
                && buf.as_bytes()[buf.len() - k..] == marker.as_bytes()[..k]
        })
        .unwrap_or(0)
}

#[derive(Default)]
pub(crate) struct AssistantOutput {
    pub(crate) content: String,
    pub(crate) reasoning: String,
    pub(crate) tool_calls: Vec<ToolCall>,
}

impl AssistantOutput {
    pub(crate) fn collect(
        text: &str,
        tools: Option<Value>,
        thinking: bool,
        dialect: Dialect,
    ) -> AssistantOutput {
        let mut parser = OutputParser::new(tools, thinking, dialect);
        let mut out = AssistantOutput::default();
        let mut events = parser.push(text);
        events.extend(parser.finish());
        for event in events {
            match event {
                OutputEvent::Reasoning(text) => out.reasoning.push_str(&text),
                OutputEvent::Content(text) => out.content.push_str(&text),
                OutputEvent::ToolCall(call) => out.tool_calls.push(*call),
            }
        }
        out.content = out.content.trim_end().to_string();
        out.reasoning = out.reasoning.trim().to_string();
        out
    }
}
