//! The route handlers.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use serde_json::{Value, json};

use crate::chat::tools::selected_tools;
use crate::chat::{
    AssistantOutput, ChatMessage, MessageContent, OutputEvent, OutputParser, format_chat,
};

use super::protocol::*;
use super::{AppState, GenerationRequest, InferenceResponse, dispatch};

pub(crate) async fn run_generation(
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

pub(crate) async fn handle_completions(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
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

pub(crate) async fn handle_chat_completions(
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
pub(crate) fn stream_chat(
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
