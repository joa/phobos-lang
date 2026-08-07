use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::chat::{ChatMessage, ToolCall};
use crate::sampling::SampleConfig;

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
    pub(crate) fn resolve(&self, defaults: &SampleConfig) -> SampleConfig {
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
    pub(crate) fn describe(&self) -> String {
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
    pub(crate) fn new(prompt_tokens: usize, completion_tokens: usize) -> Usage {
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

pub(crate) struct GeneratedText {
    pub(crate) model: String,
    pub(crate) text: String,
    pub(crate) finish_reason: String,
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
}

pub(crate) fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(crate) fn response_id(prefix: &str) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{prefix}-{millis}")
}

/// Whether the caller wants a reasoning pass. The template treats an undefined
/// `enable_thinking` as off, so this defaults the same way.
pub(crate) fn thinking_enabled(req: &ChatCompletionRequest) -> bool {
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

/// OpenAI clients accept only a fixed set here, so a worker-side stop is a stop.
pub(crate) fn normalize_finish_reason(reason: &str) -> &str {
    match reason {
        "length" => "length",
        _ => "stop",
    }
}

pub(crate) fn tool_call_delta(index: usize, call: &ToolCall) -> Value {
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

pub(crate) fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
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

pub(crate) fn model_name(requested: Option<&str>, loaded: &str) -> String {
    requested
        .filter(|name| !name.is_empty())
        .unwrap_or(loaded)
        .to_string()
}
