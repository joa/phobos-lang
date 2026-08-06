use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    sync::mpsc,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    Runtime, choose,
    sampling::{Rng, SampleConfig, Sequence},
};

#[derive(Deserialize, Default)]
pub struct SampleOverrides {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

impl SampleOverrides {
    fn resolve(&self, defaults: &SampleConfig) -> SampleConfig {
        SampleConfig {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_k: self.top_k.unwrap_or(defaults.top_k),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            min_p: self.min_p.unwrap_or(defaults.min_p),
            presence_penalty: self.presence_penalty.unwrap_or(defaults.presence_penalty),
            repetition_penalty: self
                .repetition_penalty
                .unwrap_or(defaults.repetition_penalty),
        }
    }
}

pub struct Defaults {
    pub sample: SampleConfig,
    pub seed: u64,
    pub max_tokens: usize,
}

impl Defaults {
    fn describe(&self) -> String {
        let s = &self.sample;
        format!(
            "temperature={} top_k={} top_p={} min_p={} presence_penalty={} \
             repetition_penalty={} seed={} max_tokens={}",
            s.temperature,
            s.top_k,
            s.top_p,
            s.min_p,
            s.presence_penalty,
            s.repetition_penalty,
            self.seed,
            self.max_tokens,
        )
    }
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(flatten)]
    pub sample: SampleOverrides,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<()>,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    fn new(prompt_tokens: usize, completion_tokens: usize) -> Usage {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(flatten)]
    pub sample: SampleOverrides,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub tools: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// The knob vLLM and llama.cpp expose for the template's `enable_thinking`.
    #[serde(default)]
    pub chat_template_kwargs: Option<Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

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

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// A request as it reaches the inference loop, still unresolved: the loop is
/// where the server's defaults live.
struct GenerationRequest {
    prompt: String,
    max_tokens: Option<usize>,
    sample: SampleOverrides,
    seed: Option<u64>,
}

struct GeneratedText {
    model: String,
    text: String,
    finish_reason: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

/// MiniCPM5's call syntax: one self-contained element, the name in an
/// attribute, and no `<tool_call>` wrapper.
const MINICPM_FUNCTION_START: &str = "<function name=\"";
const MINICPM_PARAM_START: &str = "<param name=\"";
const MINICPM_PARAM_END: &str = "</param>";
const CDATA_START: &str = "<![CDATA[";
const CDATA_END: &str = "]]>";

/// Which conventions a model's `tokenizer.chat_template` asks for. The two
/// templates in use agree on ChatML, `<think>` and `<tool_response>`, and
/// disagree on everything below. A model handed the other dialect's
/// instructions answers in a blend of the two that parses as neither.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dialect {
    /// Qwen3.5: no opening token, a `<tool_call>` wrapper around
    /// `<function=name>`, and an empty think block prefilled whenever thinking
    /// was not asked for.
    Qwen,
    /// MiniCPM5: opens with BOS, calls are a bare `<function name="...">` with
    /// `<param name="...">` children and CDATA for awkward values, and an
    /// unspecified `enable_thinking` prefills nothing at all.
    MiniCpm,
}

impl Dialect {
    /// Read off the file's own template rather than `general.architecture`:
    /// the template is what the model was trained against, and MiniCPM5
    /// declares the `llama` architecture.
    pub fn detect(template: Option<&str>) -> Dialect {
        match template {
            Some(template) if template.contains(MINICPM_FUNCTION_START) => Dialect::MiniCpm,
            _ => Dialect::Qwen,
        }
    }

    /// Whether an unspecified `enable_thinking` still prefills an empty think
    /// block. Qwen's template does, MiniCPM5's leaves the turn open.
    fn prefills_empty_think(self) -> bool {
        self == Dialect::Qwen
    }

    /// The markers that open and close a call in the generated text.
    fn call_markers(self) -> (&'static str, &'static str) {
        match self {
            Dialect::Qwen => (TOOL_CALL_START, TOOL_CALL_END),
            Dialect::MiniCpm => (MINICPM_FUNCTION_START, FUNCTION_END),
        }
    }
}

/// The tool-calling contract, verbatim from the model's own
/// `tokenizer.chat_template`. Qwen3.5 is trained on an XML call syntax, not the
/// JSON one most servers ask for, and asking for JSON gets a blend of the two
/// back.
const TOOL_FORMAT_INSTRUCTION: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

fn default_tool_call_type() -> String {
    "function".to_string()
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn response_id(prefix: &str) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{prefix}-{millis}")
}

fn message_content_text(content: &Option<MessageContent>) -> String {
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

/// Whether the caller wants a reasoning pass. The template treats an undefined
/// `enable_thinking` as off, so this defaults the same way.
fn thinking_enabled(req: &ChatCompletionRequest) -> bool {
    if let Some(enabled) = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("enable_thinking"))
        .and_then(Value::as_bool)
    {
        return enabled;
    }
    matches!(
        req.reasoning_effort.as_deref(),
        Some("low" | "medium" | "high")
    )
}

fn has_tools(tools: Option<&Value>) -> bool {
    tools.is_some_and(|tools| tools.as_array().is_some_and(|list| !list.is_empty()))
}

/// The tools the model should see. The template has no notion of
/// `tool_choice`, but "none" means the caller wants the tool block gone.
fn selected_tools(tools: Option<&Value>, tool_choice: Option<&Value>) -> Option<Value> {
    if tool_choice.and_then(Value::as_str) == Some("none") {
        return None;
    }
    tools.filter(|tools| has_tools(Some(tools))).cloned()
}

/// What `tool_choice` adds to the system turn when it demands a call. The
/// template cannot express this, so it is stated in prose.
fn tool_choice_instruction(tool_choice: Option<&Value>) -> Option<String> {
    match tool_choice? {
        Value::String(choice) if choice == "required" || choice == "any" => Some(
            "\n\nYou MUST call one of the functions above; do not answer directly.".to_string(),
        ),
        choice => {
            let name = choice.pointer("/function/name").and_then(Value::as_str)?;
            Some(format!(
                "\n\nYou MUST call the function {name}; do not answer directly."
            ))
        }
    }
}

/// MiniCPM5's equivalent of [`TOOL_FORMAT_INSTRUCTION`], also verbatim from
/// its template. The CDATA rule is not decoration: an `edit` call carries file
/// contents, the multi-line text with `<` in it that the bare form cannot
/// represent.
const MINICPM_TOOL_GUIDELINES: &str = "\n\nTool usage guidelines:\n- You may call zero or more functions. If no function calls are needed, just answer normally and do not include any <function ... </function>.\n- When calling a function, return an XML object within <function ... </function> using:\n<function name=\"function-name\"><param name=\"param-name\">param-value</param></function>\n- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a CDATA block: <param name=\"param-name\"><![CDATA[...multi-line value...]]></param>";

/// Where MiniCPM5's template substitutes the tool definitions when the
/// caller's system prompt places them explicitly.
const TOOL_DEF_SEP: &str = "<tool_def_sep>";

/// The tool list and call format, without the surrounding system turn.
fn render_tool_definitions(tools: &Value, dialect: Dialect) -> String {
    let mut text = String::from(match dialect {
        Dialect::Qwen => "# Tools\n\nYou have access to the following functions:\n\n<tools>",
        Dialect::MiniCpm => {
            "# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        }
    });
    for tool in tools.as_array().map(Vec::as_slice).unwrap_or_default() {
        text.push('\n');
        text.push_str(&serde_json::to_string(tool).unwrap_or_default());
    }
    text.push_str("\n</tools>");
    text.push_str(match dialect {
        Dialect::Qwen => TOOL_FORMAT_INSTRUCTION,
        Dialect::MiniCpm => MINICPM_TOOL_GUIDELINES,
    });
    text
}

/// The system turn a tool-carrying request opens with. The two templates order
/// it differently and the order is what the model was trained on: Qwen puts the
/// tool block first and the caller's system prompt after it, MiniCPM5 the other
/// way round unless the prompt places the definitions itself.
fn render_tools_system(
    tools: &Value,
    system: Option<&str>,
    tool_choice: Option<&Value>,
    dialect: Dialect,
) -> String {
    let mut definitions = render_tool_definitions(tools, dialect);
    if let Some(instruction) = tool_choice_instruction(tool_choice) {
        definitions.push_str(&instruction);
    }
    let system = system.map(str::trim).filter(|text| !text.is_empty());

    let mut text = String::from("<|im_start|>system\n");
    match dialect {
        Dialect::Qwen => {
            text.push_str(&definitions);
            if let Some(system) = system {
                text.push_str("\n\n");
                text.push_str(system);
            }
        }
        Dialect::MiniCpm => match system {
            Some(system) if system.contains(TOOL_DEF_SEP) => {
                text.push_str(&system.replace(TOOL_DEF_SEP, &definitions));
            }
            Some(system) => {
                text.push_str(system);
                text.push_str("\n\n");
                text.push_str(&definitions);
            }
            None => text.push_str(&definitions),
        },
    }
    text.push_str("<|im_end|>\n");
    text
}

/// An assistant turn's reasoning and visible content, from either the explicit
/// `reasoning_content` field or an inline `<think>` block.
fn split_reasoning(message: &ChatMessage, content: &str) -> (String, String) {
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

fn argument_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Whether a value needs CDATA to survive MiniCPM5's form, the rule its own
/// template states.
fn needs_cdata(value: &str) -> bool {
    value.contains(['<', '&', '\n'])
}

fn render_tool_calls(
    prompt: &mut String,
    calls: &[ToolCall],
    after_content: bool,
    dialect: Dialect,
) {
    for (index, call) in calls.iter().enumerate() {
        match index {
            0 if after_content => prompt.push_str("\n\n"),
            0 => {}
            _ => prompt.push('\n'),
        }
        let arguments = match serde_json::from_str::<Value>(&call.function.arguments) {
            Ok(Value::Object(arguments)) => arguments,
            _ => serde_json::Map::new(),
        };
        match dialect {
            Dialect::Qwen => {
                prompt.push_str(TOOL_CALL_START);
                prompt.push('\n');
                prompt.push_str(FUNCTION_START);
                prompt.push_str(&call.function.name);
                prompt.push_str(">\n");
                for (name, value) in arguments {
                    prompt.push_str(PARAMETER_START);
                    prompt.push_str(&name);
                    prompt.push_str(">\n");
                    prompt.push_str(&argument_text(&value));
                    prompt.push('\n');
                    prompt.push_str(PARAMETER_END);
                    prompt.push('\n');
                }
                prompt.push_str(FUNCTION_END);
                prompt.push('\n');
                prompt.push_str(TOOL_CALL_END);
            }
            Dialect::MiniCpm => {
                prompt.push_str(MINICPM_FUNCTION_START);
                prompt.push_str(&call.function.name);
                prompt.push_str("\">");
                for (name, value) in arguments {
                    prompt.push_str(MINICPM_PARAM_START);
                    prompt.push_str(&name);
                    prompt.push_str("\">");
                    let text = argument_text(&value);
                    if needs_cdata(&text) {
                        prompt.push_str(CDATA_START);
                        prompt.push_str(&text);
                        prompt.push_str(CDATA_END);
                    } else {
                        prompt.push_str(&text);
                    }
                    prompt.push_str(MINICPM_PARAM_END);
                }
                prompt.push_str(FUNCTION_END);
            }
        }
    }
}

/// Render a conversation the way the model's `tokenizer.chat_template` does.
fn format_chat(
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

/// The declared type of one tool parameter, when the caller sent a schema.
fn schema_type<'a>(tools: Option<&'a Value>, function: &str, parameter: &str) -> Option<&'a str> {
    tools?
        .as_array()?
        .iter()
        .find(|tool| tool.pointer("/function/name").and_then(Value::as_str) == Some(function))?
        .pointer("/function/parameters/properties")?
        .get(parameter)?
        .get("type")?
        .as_str()
}

/// Parameters arrive as text and the schema says how to read them. Without one
/// only unambiguously structured values are promoted, so "007" stays a
/// string.
fn coerce_argument(text: &str, function: &str, parameter: &str, tools: Option<&Value>) -> Value {
    let parsed = serde_json::from_str::<Value>(text);
    match schema_type(tools, function, parameter) {
        Some("string") | Some("null") => Value::String(text.to_string()),
        Some(_) => parsed.unwrap_or_else(|_| Value::String(text.to_string())),
        None => match parsed {
            Ok(value @ (Value::Object(_) | Value::Array(_) | Value::Bool(_))) => value,
            _ => Value::String(text.to_string()),
        },
    }
}

/// The model's XML call form: `<function=name>` wrapping `<parameter=k>`
/// blocks, each value on its own lines.
fn parse_function_block(raw: &str, tools: Option<&Value>, index: usize) -> Option<ToolCall> {
    let after_start = &raw[raw.find(FUNCTION_START)? + FUNCTION_START.len()..];
    let name = after_start[..after_start.find('>')?].trim().to_string();
    let body = &after_start[after_start.find('>')? + 1..];
    let body = body.split(FUNCTION_END).next().unwrap_or(body);

    let mut arguments = serde_json::Map::new();
    let mut rest = body;
    while let Some(at) = rest.find(PARAMETER_START) {
        let after_start = &rest[at + PARAMETER_START.len()..];
        let Some(close) = after_start.find('>') else {
            break;
        };
        let key = after_start[..close].trim().to_string();
        let (value, tail) = match after_start[close + 1..].split_once(PARAMETER_END) {
            Some((value, tail)) => (value, tail),
            None => (&after_start[close + 1..], ""),
        };
        // One newline follows the tag and one precedes the closer.
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
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

/// MiniCPM5's call form. `raw` starts just past `<function name="`, so it
/// opens with the name and its closing quote and ends before `</function>`.
fn parse_minicpm_function(raw: &str, tools: Option<&Value>, index: usize) -> Option<ToolCall> {
    let (name, body) = raw.split_once("\">")?;
    let name = name.trim().to_string();

    let mut arguments = serde_json::Map::new();
    let mut rest = body;
    while let Some(at) = rest.find(MINICPM_PARAM_START) {
        let after_start = &rest[at + MINICPM_PARAM_START.len()..];
        let Some((key, after_key)) = after_start.split_once("\">") else {
            break;
        };
        let key = key.trim().to_string();
        // CDATA carries values holding `<`, `&` or newlines, so its closer
        // has to be found first, or a value containing one ends the parameter
        // early.
        let (value, tail) = match after_key.strip_prefix(CDATA_START) {
            Some(inner) => match inner.split_once(CDATA_END) {
                Some((value, tail)) => (value, tail),
                None => (inner, ""),
            },
            None => match after_key.split_once(MINICPM_PARAM_END) {
                Some((value, tail)) => (value, tail),
                None => (after_key, ""),
            },
        };
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

/// The JSON call form other servers ask for, which the model still reaches for
/// occasionally.
fn parse_json_call(json_value: &Value, index: usize) -> Option<ToolCall> {
    let function = json_value.get("function").unwrap_or(json_value);
    let name = function.get("name")?.as_str()?.to_string();
    let arguments = function
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = match arguments {
        Value::String(text) => text,
        other => serde_json::to_string(&other).ok()?,
    };

    Some(ToolCall {
        id: json_value
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| format!("call_{index}")),
        call_type: json_value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("function")
            .to_string(),
        function: ToolCallFunction { name, arguments },
    })
}

#[derive(Debug, PartialEq, Eq)]
enum OutputEvent {
    Reasoning(String),
    Content(String),
    ToolCall(Box<ToolCall>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Reasoning,
    Content,
    ToolCall,
}

/// Splits a generation into reasoning, prose and tool calls as it arrives.
/// Incremental because of streaming: a chunk can end halfway through `</think>`
/// or `<tool_call>`, so the tail that could still become a marker is held back
/// until the next chunk decides it.
struct OutputParser {
    tools: Option<Value>,
    dialect: Dialect,
    mode: Mode,
    buf: String,
    calls: usize,
    at_content_start: bool,
}

impl OutputParser {
    fn new(tools: Option<Value>, thinking: bool, dialect: Dialect) -> OutputParser {
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

    fn push(&mut self, chunk: &str) -> Vec<OutputEvent> {
        self.buf.push_str(chunk);
        self.drain(false)
    }

    /// Flush what is left, holding nothing back.
    fn finish(&mut self) -> Vec<OutputEvent> {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> Vec<OutputEvent> {
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

    fn parse_call(&self, raw: &str) -> Option<ToolCall> {
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
    fn emit_content(&mut self, events: &mut Vec<OutputEvent>, text: String) {
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

fn emit(events: &mut Vec<OutputEvent>, make: fn(String) -> OutputEvent, text: String) {
    if !text.is_empty() {
        events.push(make(text));
    }
}

/// How many bytes at the end of `buf` could still grow into `marker`.
fn held_back(buf: &str, marker: &str, flush: bool) -> usize {
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
struct AssistantOutput {
    content: String,
    reasoning: String,
    tool_calls: Vec<ToolCall>,
}

impl AssistantOutput {
    fn collect(
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

/// OpenAI clients accept only a fixed set here, so a worker-side stop is a stop.
fn normalize_finish_reason(reason: &str) -> &str {
    match reason {
        "length" => "length",
        _ => "stop",
    }
}

fn tool_call_delta(index: usize, call: &ToolCall) -> Value {
    json!({
        "tool_calls": [{
            "index": index,
            "id": call.id,
            "type": call.call_type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            }
        }]
    })
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.into(),
                "type": "invalid_request_error",
                "param": null,
                "code": null
            }
        })),
    )
        .into_response()
}

fn model_name(requested: Option<&str>, loaded: &str) -> String {
    requested
        .filter(|name| !name.is_empty())
        .unwrap_or(loaded)
        .to_string()
}

pub enum InferenceResponse {
    Start {
        model: String,
        prompt_tokens: usize,
    },
    Chunk(String),
    Done {
        reason: String,
        completion_tokens: usize,
    },
}

struct InferenceRequest {
    req: GenerationRequest,
    responder: tokio::sync::mpsc::UnboundedSender<std::result::Result<InferenceResponse, String>>,
}

#[derive(Clone)]
struct AppState {
    tx: mpsc::SyncSender<InferenceRequest>,
    model: String,
    /// Resolved once from the loaded model's chat template. Every request
    /// renders and parses against it.
    dialect: Dialect,
    /// The token a chat prompt opens with, when the template calls for one.
    bos: Option<String>,
}

async fn root_handler() -> &'static str {
    "phobos-inference"
}

async fn models_handler(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model,
            "object": "model",
            "created": 0,
            "owned_by": "phobos"
        }]
    }))
}

async fn fallback_handler(
    uri: axum::http::Uri,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> StatusCode {
    println!("404 Not Found: {} {}", method, uri);
    let body_str = String::from_utf8_lossy(&body);
    if !body_str.is_empty() {
        println!("Body: {}", body_str);
    }
    StatusCode::NOT_FOUND
}

pub fn serve(addr: String, runtime: Runtime, defaults: Defaults) -> Result<()> {
    let (tx, rx) = mpsc::sync_channel::<InferenceRequest>(100);
    let model = runtime.label();
    let dialect = Dialect::detect(runtime.chat_template());
    let bos = runtime.bos_text().map(str::to_string);

    let state = AppState {
        tx,
        model,
        dialect,
        bos,
    };

    println!("request defaults: {}", defaults.describe());
    println!("chat dialect: {dialect:?}");

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let app = Router::new()
                .route("/", get(root_handler))
                .route("/v1/completions", post(handle_completions))
                .route("/v1/chat/completions", post(handle_chat_completions))
                .route("/v1/models", get(models_handler))
                .fallback(fallback_handler)
                .with_state(state);

            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
            println!("Listening on http://{}", addr);
            axum::serve(listener, app).await.unwrap();
        });
    });

    // Inference loop on the main thread.
    for inf_req in rx {
        let req = inf_req.req;

        let mut rng = Rng::new(req.seed.unwrap_or(defaults.seed));
        let sample = req.sample.resolve(&defaults.sample);
        let max_tokens = req.max_tokens.unwrap_or(defaults.max_tokens);

        let ids_result = runtime.encode(&req.prompt);
        if ids_result.is_err() {
            let _ = inf_req
                .responder
                .send(Err("Failed to encode prompt".to_string()));
            continue;
        }
        let ids = ids_result.unwrap();

        let start_result = runtime.start(&ids);
        if start_result.is_err() {
            let _ = inf_req
                .responder
                .send(Err("Failed to start inference".to_string()));
            continue;
        }
        let (logits, mut run_state) = start_result.unwrap();

        let _ = inf_req.responder.send(Ok(InferenceResponse::Start {
            model: runtime.label(),
            prompt_tokens: ids.len(),
        }));

        let mut sequence = Sequence::new(ids);
        let first = choose(&logits, &sample, sequence.history(), &mut rng);
        let context_limit = runtime.context_limit();

        let mut next = first;
        let mut produced = 0usize;
        let mut emitted = 0usize;
        let mut pending = Vec::new();

        let stop_reason = loop {
            if runtime.is_eog(next) {
                break "stop";
            }
            if run_state.len() >= context_limit {
                break "length";
            }

            pending.extend(runtime.decode_bytes(&[next]));
            emitted += 1;
            let valid = match std::str::from_utf8(&pending) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid > 0 {
                let token_str = std::str::from_utf8(&pending[..valid]).unwrap().to_string();
                pending.drain(..valid);
                if inf_req
                    .responder
                    .send(Ok(InferenceResponse::Chunk(token_str)))
                    .is_err()
                {
                    break "client_disconnect";
                }
            }

            sequence.push(next);
            match runtime.advance(&mut run_state, next) {
                Ok(next_logits) => {
                    next = choose(&next_logits, &sample, sequence.history(), &mut rng);
                    produced += 1;
                    if produced >= max_tokens {
                        break "length";
                    }
                }
                Err(_) => {
                    break "error";
                }
            }
        };

        if !pending.is_empty() {
            let token_str = String::from_utf8_lossy(&pending).into_owned();
            let _ = inf_req
                .responder
                .send(Ok(InferenceResponse::Chunk(token_str)));
        }

        let _ = inf_req.responder.send(Ok(InferenceResponse::Done {
            reason: stop_reason.to_string(),
            completion_tokens: emitted,
        }));

        // Dropping the state would strand its caches on the device, a few
        // hundred megabytes a request.
        runtime.finish(run_state);
    }

    Ok(())
}

fn dispatch(
    state: &AppState,
    req: GenerationRequest,
) -> std::result::Result<
    tokio::sync::mpsc::UnboundedReceiver<std::result::Result<InferenceResponse, String>>,
    StatusCode,
> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    match state.tx.send(InferenceRequest { req, responder: tx }) {
        Ok(()) => Ok(rx),
        Err(_) => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

async fn run_generation(
    state: AppState,
    req: GenerationRequest,
) -> std::result::Result<GeneratedText, StatusCode> {
    let mut rx = dispatch(&state, req)?;

    let mut model = state.model;
    let mut text = String::new();
    let mut finish_reason = "stop".to_string();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;

    while let Some(msg) = rx.recv().await {
        match msg {
            Ok(InferenceResponse::Start {
                model: label,
                prompt_tokens: count,
            }) => {
                model = label;
                prompt_tokens = count;
            }
            Ok(InferenceResponse::Chunk(chunk)) => text.push_str(&chunk),
            Ok(InferenceResponse::Done {
                reason,
                completion_tokens: count,
            }) => {
                finish_reason = reason;
                completion_tokens = count;
            }
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    Ok(GeneratedText {
        model,
        text,
        finish_reason,
        prompt_tokens,
        completion_tokens,
    })
}

async fn handle_completions(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
    let body_str = String::from_utf8_lossy(&body);
    println!("Received request: {}", body_str);

    let req: CompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            println!("Failed to parse request: {}", e);
            return error_response(StatusCode::BAD_REQUEST, e.to_string());
        }
    };

    if req.stream.unwrap_or(false) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "streaming text completions are not implemented; use /v1/chat/completions",
        );
    }

    let requested_model = req.model.clone();
    let gen_req = GenerationRequest {
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        sample: req.sample,
        seed: req.seed,
    };

    let generated = match run_generation(state, gen_req).await {
        Ok(generated) => generated,
        Err(status) => return error_response(status, "inference failed"),
    };

    let res = CompletionResponse {
        id: response_id("cmpl"),
        object: "text_completion".to_string(),
        created: unix_timestamp(),
        model: model_name(requested_model.as_deref(), &generated.model),
        choices: vec![CompletionChoice {
            text: generated.text,
            index: 0,
            logprobs: None,
            finish_reason: normalize_finish_reason(&generated.finish_reason).to_string(),
        }],
        usage: Usage::new(generated.prompt_tokens, generated.completion_tokens),
    };

    Json(res).into_response()
}

async fn handle_chat_completions(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let body_str = String::from_utf8_lossy(&body);
    println!("Received chat request: {}", body_str);

    let chat_req: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            println!("Failed to parse chat request: {}", e);
            return error_response(StatusCode::BAD_REQUEST, e.to_string());
        }
    };

    let thinking = thinking_enabled(&chat_req);
    let tools = selected_tools(chat_req.tools.as_ref(), chat_req.tool_choice.as_ref());
    let prompt = format_chat(
        &chat_req.messages,
        tools.as_ref(),
        chat_req.tool_choice.as_ref(),
        thinking,
        state.dialect,
        state.bos.as_deref(),
    );
    let max_tokens = chat_req.max_completion_tokens.or(chat_req.max_tokens);
    let requested_model = chat_req.model.clone();

    let gen_req = GenerationRequest {
        prompt,
        max_tokens,
        sample: chat_req.sample,
        seed: chat_req.seed,
    };

    if chat_req.stream.unwrap_or(false) {
        let include_usage = chat_req
            .stream_options
            .map(|options| options.include_usage)
            .unwrap_or(false);
        return stream_chat(
            state,
            gen_req,
            requested_model,
            tools,
            thinking,
            include_usage,
        );
    }

    let dialect = state.dialect;
    let generated = match run_generation(state, gen_req).await {
        Ok(generated) => generated,
        Err(status) => return error_response(status, "inference failed"),
    };

    let out = AssistantOutput::collect(&generated.text, tools, thinking, dialect);
    let has_tool_calls = !out.tool_calls.is_empty();
    let finish_reason = if has_tool_calls {
        "tool_calls".to_string()
    } else {
        normalize_finish_reason(&generated.finish_reason).to_string()
    };

    let chat_resp = ChatCompletionResponse {
        id: response_id("chatcmpl"),
        object: "chat.completion".to_string(),
        created: unix_timestamp(),
        model: model_name(requested_model.as_deref(), &generated.model),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                // OpenAI clients expect a null content beside tool calls.
                content: (!has_tool_calls || !out.content.is_empty())
                    .then_some(MessageContent::Text(out.content)),
                reasoning_content: (!out.reasoning.is_empty()).then_some(out.reasoning),
                name: None,
                tool_call_id: None,
                tool_calls: has_tool_calls.then_some(out.tool_calls),
            },
            finish_reason,
        }],
        usage: Usage::new(generated.prompt_tokens, generated.completion_tokens),
    };
    Json(chat_resp).into_response()
}

/// Stream the generation as it is produced. Deltas go out token by token
/// except inside a `<tool_call>`, which means nothing until it is whole.
fn stream_chat(
    state: AppState,
    req: GenerationRequest,
    requested_model: Option<String>,
    tools: Option<Value>,
    thinking: bool,
    include_usage: bool,
) -> Response {
    let mut rx = match dispatch(&state, req) {
        Ok(rx) => rx,
        Err(status) => return error_response(status, "inference worker is unavailable"),
    };

    let id = response_id("chatcmpl");
    let created = unix_timestamp();
    let dialect = state.dialect;
    let model = model_name(requested_model.as_deref(), &state.model);

    let stream = async_stream::stream! {
        let chunk = |delta: Value, finish: Value| {
            Ok::<_, std::convert::Infallible>(Event::default().data(json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish
                }]
            }).to_string()))
        };

        let mut parser = OutputParser::new(tools, thinking, dialect);
        let mut calls = 0usize;
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason = "stop".to_string();

        yield chunk(json!({ "role": "assistant", "content": "" }), Value::Null);

        while let Some(msg) = rx.recv().await {
            let events = match msg {
                Ok(InferenceResponse::Start { prompt_tokens: count, .. }) => {
                    prompt_tokens = count;
                    Vec::new()
                }
                Ok(InferenceResponse::Chunk(text)) => parser.push(&text),
                Ok(InferenceResponse::Done { reason, completion_tokens: count }) => {
                    finish_reason = reason;
                    completion_tokens = count;
                    parser.finish()
                }
                Err(_) => {
                    finish_reason = "error".to_string();
                    parser.finish()
                }
            };

            for event in events {
                match event {
                    OutputEvent::Reasoning(text) => {
                        yield chunk(json!({ "reasoning_content": text }), Value::Null);
                    }
                    OutputEvent::Content(text) => {
                        yield chunk(json!({ "content": text }), Value::Null);
                    }
                    OutputEvent::ToolCall(call) => {
                        yield chunk(tool_call_delta(calls, &call), Value::Null);
                        calls += 1;
                    }
                }
            }
        }

        let finish = if calls > 0 {
            "tool_calls"
        } else {
            normalize_finish_reason(&finish_reason)
        };
        yield chunk(json!({}), Value::String(finish.to_string()));

        if include_usage {
            yield Ok(Event::default().data(json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }).to_string()));
        }

        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(text.to_string())),
            ..ChatMessage::default()
        }
    }

    fn bash_tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": { "type": "string" },
                        "timeout": { "type": "integer" }
                    }
                }
            }
        }])
    }

    #[test]
    fn parses_the_models_xml_tool_call() {
        let out = AssistantOutput::collect(
            "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>",
            Some(bash_tools()),
            false,
            Dialect::Qwen,
        );

        assert_eq!(out.content, "");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "bash");
        assert_eq!(
            out.tool_calls[0].function.arguments,
            "{\"command\":\"ls -la\",\"timeout\":30}"
        );
    }

    #[test]
    fn a_multiline_parameter_keeps_its_newlines() {
        let out = AssistantOutput::collect(
            "<tool_call>\n<function=bash>\n<parameter=command>\nfirst\nsecond\n</parameter>\n</function>\n</tool_call>",
            Some(bash_tools()),
            false,
            Dialect::Qwen,
        );

        assert_eq!(
            out.tool_calls[0].function.arguments,
            "{\"command\":\"first\\nsecond\"}"
        );
    }

    #[test]
    fn reasoning_never_reaches_the_content() {
        let out = AssistantOutput::collect(
            " The user wants a list.\n</think>\n\nHere you go.",
            None,
            true,
            Dialect::Qwen,
        );

        assert_eq!(out.reasoning, "The user wants a list.");
        assert_eq!(out.content, "Here you go.");
    }

    #[test]
    fn a_marker_split_across_chunks_still_parses() {
        let mut parser = OutputParser::new(Some(bash_tools()), true, Dialect::Qwen);
        let mut events = Vec::new();
        for chunk in [
            "think",
            "ing</",
            "thin",
            "k>\n\nprose <",
            "tool_call>\n<function=bash>\n",
            "<parameter=command>\nls\n</parameter>\n</function>\n</tool",
            "_call>",
        ] {
            events.extend(parser.push(chunk));
        }
        events.extend(parser.finish());

        let content: String = events
            .iter()
            .filter_map(|event| match event {
                OutputEvent::Content(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let calls: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                OutputEvent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();

        assert_eq!(content, "prose ");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, "{\"command\":\"ls\"}");
    }

    #[test]
    fn text_that_only_looks_like_a_call_stays_text() {
        let out = AssistantOutput::collect(
            "before <tool_call>not a call</tool_call>",
            None,
            false,
            Dialect::Qwen,
        );

        assert_eq!(out.content, "before <tool_call>not a call</tool_call>");
        assert!(out.tool_calls.is_empty());
    }

    #[test]
    fn the_json_call_form_still_parses() {
        let out = AssistantOutput::collect(
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}</tool_call>",
            None,
            false,
            Dialect::Qwen,
        );

        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            out.tool_calls[0].function.arguments,
            "{\"city\":\"Berlin\"}"
        );
    }

    #[test]
    fn a_tools_request_opens_with_the_templates_system_turn() {
        let prompt = format_chat(
            &[
                ChatMessage {
                    role: "system".to_string(),
                    content: Some(MessageContent::Text("You are pi.".to_string())),
                    ..ChatMessage::default()
                },
                user("list the files"),
            ],
            Some(&bash_tools()),
            None,
            false,
            Dialect::Qwen,
            None,
        );

        assert!(prompt.starts_with(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"
        ));
        // The caller's system prompt follows the tool contract, not the reverse.
        assert!(prompt.contains("</IMPORTANT>\n\nYou are pi.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nlist the files<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    /// The two templates disagree on the call syntax, and a model handed the
    /// other answers in a blend of the two that parses as neither.
    #[test]
    fn the_dialect_comes_off_the_models_own_template() {
        assert_eq!(Dialect::detect(None), Dialect::Qwen);
        assert_eq!(
            Dialect::detect(Some("{{- '<tool_call>\\n<function=' + name }}")),
            Dialect::Qwen
        );
        assert_eq!(
            Dialect::detect(Some("{{- '<function name=\"' ~ tool_call.name ~ '\">' }}")),
            Dialect::MiniCpm
        );
    }

    #[test]
    fn minicpm_calls_are_one_element_with_named_attributes() {
        let out = AssistantOutput::collect(
            "<function name=\"bash\"><param name=\"command\">ls -la</param><param name=\"timeout\">30</param></function>",
            Some(bash_tools()),
            false,
            Dialect::MiniCpm,
        );

        assert_eq!(out.content, "");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "bash");
        assert_eq!(
            out.tool_calls[0].function.arguments,
            "{\"command\":\"ls -la\",\"timeout\":30}"
        );
    }

    /// What the guidelines exist for: an argument holding the `<` and the
    /// newlines that would otherwise close the parameter early.
    #[test]
    fn a_cdata_argument_survives_its_own_markup() {
        let out = AssistantOutput::collect(
            "<function name=\"write\"><param name=\"content\"><![CDATA[<html>\nif a < b\n</html>]]></param></function>",
            None,
            false,
            Dialect::MiniCpm,
        );

        assert_eq!(
            out.tool_calls[0].function.arguments,
            "{\"content\":\"<html>\\nif a < b\\n</html>\"}"
        );
    }

    #[test]
    fn a_minicpm_prompt_opens_with_bos_and_prefills_no_think_block() {
        let prompt = format_chat(
            &[user("hi")],
            None,
            None,
            false,
            Dialect::MiniCpm,
            Some("<s>"),
        );

        assert!(prompt.starts_with("<s><|im_start|>user\nhi<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    /// MiniCPM5 puts the caller's system prompt first and the tool block after
    /// it, the reverse of Qwen's order.
    #[test]
    fn the_minicpm_tool_block_follows_the_system_prompt() {
        let prompt = format_chat(
            &[
                ChatMessage {
                    role: "system".to_string(),
                    content: Some(MessageContent::Text("You are pi.".to_string())),
                    ..ChatMessage::default()
                },
                user("list the files"),
            ],
            Some(&bash_tools()),
            None,
            false,
            Dialect::MiniCpm,
            Some("<s>"),
        );

        assert!(prompt.starts_with(
            "<s><|im_start|>system\nYou are pi.\n\n# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"
        ));
        assert!(prompt.contains("<param name=\"param-name\">param-value</param>"));
        assert!(!prompt.contains(TOOL_CALL_START));
    }

    /// What the parser reads out of a call has to render back into the prompt
    /// the same way, or a tool loop's second turn drifts.
    #[test]
    fn a_minicpm_call_renders_back_into_the_prompt() {
        let call = ToolCall {
            id: "call_0".to_string(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: "write".to_string(),
                arguments: "{\"content\":\"a\\nb\",\"path\":\"x.txt\"}".to_string(),
            },
        };
        let mut prompt = String::new();
        render_tool_calls(&mut prompt, &[call], false, Dialect::MiniCpm);

        assert_eq!(
            prompt,
            "<function name=\"write\"><param name=\"content\"><![CDATA[a\nb]]></param>\
             <param name=\"path\">x.txt</param></function>"
        );
    }

    /// MiniCPM5's template prefills no think block, so the model opens one
    /// itself and the parser picks it up mid-content rather than assuming it
    /// started inside one.
    #[test]
    fn a_think_block_the_model_opens_itself_is_still_reasoning() {
        let out = AssistantOutput::collect(
            "<think>\nweighing it up\n</think>\n\nHere you go.",
            None,
            false,
            Dialect::MiniCpm,
        );

        assert_eq!(out.reasoning, "weighing it up");
        assert_eq!(out.content, "Here you go.");
    }

    #[test]
    fn a_thinking_request_leaves_the_block_open() {
        let prompt = format_chat(&[user("hi")], None, None, true, Dialect::Qwen, None);
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    #[test]
    fn tool_results_become_one_user_turn() {
        let call = ToolCall {
            id: "call_0".to_string(),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: "bash".to_string(),
                arguments: "{\"command\":\"ls\"}".to_string(),
            },
        };
        let tool_result = |text: &str| ChatMessage {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(text.to_string())),
            tool_call_id: Some("call_0".to_string()),
            ..ChatMessage::default()
        };
        let prompt = format_chat(
            &[
                user("list the files"),
                ChatMessage {
                    role: "assistant".to_string(),
                    tool_calls: Some(vec![call]),
                    ..ChatMessage::default()
                },
                tool_result("a.txt"),
                tool_result("b.txt"),
            ],
            Some(&bash_tools()),
            None,
            false,
            Dialect::Qwen,
            None,
        );

        assert!(prompt.contains(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        ));
        assert!(prompt.contains(
            "<|im_start|>user\n<tool_response>\na.txt\n</tool_response>\n<tool_response>\nb.txt\n</tool_response><|im_end|>\n"
        ));
        assert!(!prompt.contains("tool_call_id"));
    }

    #[test]
    fn an_earlier_assistant_turn_drops_its_reasoning() {
        let prompt = format_chat(
            &[
                user("hi"),
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(MessageContent::Text(
                        "<think>\npondering\n</think>\n\nhello".to_string(),
                    )),
                    ..ChatMessage::default()
                },
                user("again"),
            ],
            None,
            None,
            false,
            Dialect::Qwen,
            None,
        );

        assert!(prompt.contains("<|im_start|>assistant\nhello<|im_end|>\n"));
        assert!(!prompt.contains("pondering"));
    }

    #[test]
    fn an_unschemad_argument_stays_a_string() {
        let out = AssistantOutput::collect(
            "<tool_call>\n<function=lookup>\n<parameter=zip>\n007\n</parameter>\n</function>\n</tool_call>",
            None,
            false,
            Dialect::Qwen,
        );

        assert_eq!(out.tool_calls[0].function.arguments, "{\"zip\":\"007\"}");
    }

    /// A server started with the Qwen non-thinking preset.
    fn command_line() -> SampleConfig {
        SampleConfig {
            temperature: 1.0,
            top_k: 20,
            top_p: 1.0,
            min_p: 0.0,
            presence_penalty: 2.0,
            repetition_penalty: 1.0,
        }
    }

    #[test]
    fn a_bare_request_inherits_the_command_line() {
        let req: CompletionRequest = serde_json::from_str(r#"{"prompt": "hi"}"#).unwrap();
        let sample = req.sample.resolve(&command_line());

        assert_eq!(sample.temperature, 1.0);
        assert_eq!(sample.top_k, 20);
        assert_eq!(sample.presence_penalty, 2.0);
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.seed, None);
    }

    #[test]
    fn a_sent_field_wins_and_the_rest_still_inherit() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages": [], "temperature": 0.6, "top_p": 0.95, "min_p": 0.05}"#,
        )
        .unwrap();
        let sample = req.sample.resolve(&command_line());

        assert_eq!(sample.temperature, 0.6);
        assert_eq!(sample.top_p, 0.95);
        assert_eq!(sample.min_p, 0.05);
        // Untouched by the request, so still the command line's.
        assert_eq!(sample.top_k, 20);
        assert_eq!(sample.presence_penalty, 2.0);
    }

    #[test]
    fn a_sent_field_wins_even_when_it_equals_the_built_in_default() {
        // Why the request fields are Options: a client asking for greedy has
        // to get greedy, not the temperature the server started with.
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages": [], "temperature": 0.0}"#).unwrap();
        let sample = req.sample.resolve(&command_line());

        assert_eq!(sample.temperature, 0.0);
        assert!(sample.is_greedy());
    }

    #[test]
    fn max_completion_tokens_outranks_max_tokens() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages": [], "max_tokens": 10}"#).unwrap();
        assert_eq!(req.max_completion_tokens.or(req.max_tokens), Some(10));

        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages": [], "max_tokens": 10, "max_completion_tokens": 20}"#,
        )
        .unwrap();
        assert_eq!(req.max_completion_tokens.or(req.max_tokens), Some(20));
    }
}
