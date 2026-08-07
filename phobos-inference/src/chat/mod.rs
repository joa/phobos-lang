pub mod dialect;
pub mod stream;
pub mod tools;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) use dialect::Dialect;
pub(crate) use stream::{AssistantOutput, OutputEvent, OutputParser};

use dialect::{THINK_END, THINK_START};
use tools::{default_tool_call_type, has_tools, render_tool_calls, render_tools_system};

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Array(Vec<ContentPart>),
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "default_tool_call_type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

pub(crate) fn message_content_text(content: &Option<MessageContent>) -> String {
    match content {
        Some(MessageContent::Text(text)) => text.clone(),
        Some(MessageContent::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                (part.part_type == "text")
                    .then_some(part.text.as_deref())
                    .flatten()
            })
            .collect::<Vec<_>>()
            .join(""),
        None => String::new(),
    }
}

/// An assistant turn's reasoning and visible content, from either the explicit
/// `reasoning_content` field or an inline `<think>` block.
pub(crate) fn split_reasoning(message: &ChatMessage, content: &str) -> (String, String) {
    if let Some(reasoning) = &message.reasoning_content {
        return (reasoning.trim().to_string(), content.to_string());
    }
    match content.split_once(THINK_END) {
        Some((before, after)) => (
            before
                .rsplit(THINK_START)
                .next()
                .unwrap_or(before)
                .trim()
                .to_string(),
            after.trim_start_matches('\n').to_string(),
        ),
        None => (String::new(), content.to_string()),
    }
}

pub(crate) fn format_chat(
    messages: &[ChatMessage],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    thinking: bool,
    dialect: Dialect,
    bos: Option<&str>,
) -> String {
    let mut prompt = String::new();

    // MiniCPM5's template opens `{{- bos_token }}` and its file sets
    // `add_bos_token = false`, so nothing else would add one.
    if let Some(bos) = bos {
        prompt.push_str(bos);
    }

    let system = messages
        .first()
        .filter(|message| message.role == "system")
        .map(|message| message_content_text(&message.content));
    let mut index = usize::from(system.is_some());

    if has_tools(tools) {
        prompt.push_str(&render_tools_system(
            tools.unwrap(),
            system.as_deref(),
            tool_choice,
            dialect,
        ));
    } else if let Some(system) = &system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system.trim());
        prompt.push_str("<|im_end|>\n");
    }

    // Assistant turns past the caller's last real question are tool-loop steps
    // and keep their reasoning; earlier ones drop it.
    let last_query = messages
        .iter()
        .rposition(|message| message.role == "user")
        .unwrap_or(messages.len().saturating_sub(1));

    while index < messages.len() {
        let message = &messages[index];
        let content = message_content_text(&message.content);
        let content = content.trim();

        match message.role.as_str() {
            // A run of tool results is one user turn of <tool_response>s.
            "tool" => {
                prompt.push_str("<|im_start|>user");
                while let Some(result) = messages.get(index).filter(|m| m.role == "tool") {
                    prompt.push_str("\n<tool_response>\n");
                    prompt.push_str(message_content_text(&result.content).trim());
                    prompt.push_str("\n</tool_response>");
                    index += 1;
                }
                prompt.push_str("<|im_end|>\n");
                continue;
            }
            "assistant" => {
                let (reasoning, content) = split_reasoning(message, content);
                prompt.push_str("<|im_start|>assistant\n");
                if index > last_query {
                    prompt.push_str(THINK_START);
                    prompt.push('\n');
                    prompt.push_str(&reasoning);
                    prompt.push_str("\n</think>\n\n");
                }
                prompt.push_str(&content);
                render_tool_calls(
                    &mut prompt,
                    message.tool_calls.as_deref().unwrap_or_default(),
                    !content.is_empty(),
                    dialect,
                );
                prompt.push_str("<|im_end|>\n");
            }
            // A second system message is outside the template's grammar, so
            // it folds in as a plain turn rather than being dropped.
            role => {
                prompt.push_str("<|im_start|>");
                prompt.push_str(role);
                prompt.push('\n');
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            }
        }
        index += 1;
    }

    prompt.push_str("<|im_start|>assistant\n");
    // Qwen's template prefills a think block either way, and leaving it out
    // invites the model to open one itself, which leaks into the visible
    // content. MiniCPM5's prefills nothing unless thinking was asked for, and
    // an empty block it was not trained to see costs it the turn's opening.
    prompt.push_str(match (thinking, dialect.prefills_empty_think()) {
        (true, _) => "<think>\n",
        (false, true) => "<think>\n\n</think>\n\n",
        (false, false) => "",
    });
    prompt
}
