// GLM-4's chat format. Turns open with a role marker and nothing closes them:
// the model ends its own turn by opening the next one, `<|user|>` or
// `<|observation|>`. Rendered as the file's `tokenizer.chat_template` does.

use serde_json::Value;

use super::dialect::{THINK_END, THINK_START, TOOL_CALL_END, TOOL_CALL_START};
use super::dialect::Dialect;
use super::tools::{argument_text, coerce_argument, default_tool_call_type, has_tools, render_tools_system};
use super::{ChatMessage, Thinking, ToolCall, ToolCallFunction, message_content_text, split_reasoning};

/// What every GLM prompt opens with, in place of a BOS.
pub(crate) const GLM_OPENER: &str = "[gMASK]<sop>";

/// Where the assistant's turn opens, and with it the generation prompt.
pub(crate) const GLM_ASSISTANT: &str = "<|assistant|>";

const ARG_KEY_START: &str = "<arg_key>";
const ARG_KEY_END: &str = "</arg_key>";
const ARG_VALUE_START: &str = "<arg_value>";
const ARG_VALUE_END: &str = "</arg_value>";

/// Appended to a user turn when thinking is off, which is how the template
/// asks the model to skip its reasoning.
const NO_THINK: &str = "/nothink";

/// The tool block's opening, verbatim from GLM-4's template.
pub(crate) const GLM_TOOLS_PREAMBLE: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";

/// What follows the tool list, also verbatim.
pub(crate) const GLM_TOOL_FORMAT: &str = "\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>";

pub(crate) fn format_glm(
    messages: &[ChatMessage],
    tools: Option<&Value>,
    tool_choice: Option<&Value>,
    thinking: Thinking,
) -> String {
    let mut prompt = String::from(GLM_OPENER);

    // The caller's system prompt stays a turn of its own, after this one.
    if let Some(tools) = tools.filter(|tools| has_tools(Some(tools))) {
        prompt.push_str(&render_tools_system(tools, None, tool_choice, Dialect::Glm));
    }

    // Assistant turns past the caller's last question are tool-loop steps
    // and keep their reasoning; earlier ones drop it.
    let last_query = messages.iter().rposition(|message| message.role == "user");

    for (index, message) in messages.iter().enumerate() {
        let content = message_content_text(&message.content);
        let content = content.trim();
        match message.role.as_str() {
            "assistant" => {
                let (reasoning, content) = split_reasoning(message, content);
                prompt.push_str(GLM_ASSISTANT);
                prompt.push('\n');
                prompt.push_str(THINK_START);
                if last_query.is_some_and(|last| index > last) {
                    prompt.push_str(reasoning.trim());
                }
                prompt.push_str(THINK_END);
                if !content.trim().is_empty() {
                    prompt.push('\n');
                    prompt.push_str(content.trim());
                }
                render_glm_calls(&mut prompt, message.tool_calls.as_deref().unwrap_or_default());
            }
            // A run of tool results shares one observation marker.
            "tool" => {
                if index == 0 || messages[index - 1].role != "tool" {
                    prompt.push_str("<|observation|>");
                }
                prompt.push_str("\n<tool_response>\n");
                prompt.push_str(content);
                prompt.push_str("\n</tool_response>\n");
            }
            "user" => {
                prompt.push_str("<|user|>\n");
                prompt.push_str(content);
                if thinking == Thinking::Off && !content.ends_with(NO_THINK) {
                    prompt.push_str(NO_THINK);
                }
            }
            role => {
                prompt.push_str("<|");
                prompt.push_str(role);
                prompt.push_str("|>\n");
                prompt.push_str(content);
            }
        }
    }

    prompt.push_str(GLM_ASSISTANT);
    prompt.push('\n');
    match thinking {
        Thinking::Off => {
            prompt.push_str(THINK_START);
            prompt.push_str(THINK_END);
            prompt.push('\n');
        }
        Thinking::Open => prompt.push_str(THINK_START),
        Thinking::Prefilled(prefill) => {
            prompt.push_str(THINK_START);
            prompt.push_str(prefill);
        }
    }
    prompt
}

/// Calls as the model writes them: the name on the marker's line, then one
/// key and value pair per argument.
pub(crate) fn render_glm_calls(prompt: &mut String, calls: &[ToolCall]) {
    for call in calls {
        let arguments = match serde_json::from_str::<Value>(&call.function.arguments) {
            Ok(Value::Object(arguments)) => arguments,
            _ => serde_json::Map::new(),
        };
        prompt.push('\n');
        prompt.push_str(TOOL_CALL_START);
        prompt.push_str(&call.function.name);
        prompt.push('\n');
        for (key, value) in arguments {
            prompt.push_str(ARG_KEY_START);
            prompt.push_str(&key);
            prompt.push_str(ARG_KEY_END);
            prompt.push('\n');
            prompt.push_str(ARG_VALUE_START);
            prompt.push_str(&argument_text(&value));
            prompt.push_str(ARG_VALUE_END);
            prompt.push('\n');
        }
        prompt.push_str(TOOL_CALL_END);
    }
}

/// A call's body between `<tool_call>` and `</tool_call>`: the name, then
/// `<arg_key>`/`<arg_value>` pairs.
pub(crate) fn parse_glm_call(raw: &str, tools: Option<&Value>, index: usize) -> Option<ToolCall> {
    let name_end = raw.find(['\n', '<']).unwrap_or(raw.len());
    let name = raw[..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }

    let mut arguments = serde_json::Map::new();
    let mut rest = &raw[name_end..];
    while let Some(at) = rest.find(ARG_KEY_START) {
        let after_key = &rest[at + ARG_KEY_START.len()..];
        let (key, after_key) = after_key.split_once(ARG_KEY_END)?;
        let key = key.trim().to_string();
        let after_value = &after_key[after_key.find(ARG_VALUE_START)? + ARG_VALUE_START.len()..];
        let (value, tail) = after_value.split_once(ARG_VALUE_END).unwrap_or((after_value, ""));
        arguments.insert(key.clone(), coerce_argument(value, &name, &key, tools));
        rest = tail;
    }

    Some(ToolCall {
        id: format!("call_{index}"),
        call_type: default_tool_call_type(),
        function: ToolCallFunction {
            name,
            arguments: Value::Object(arguments).to_string(),
        },
    })
}
