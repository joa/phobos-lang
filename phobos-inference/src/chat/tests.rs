use super::dialect::*;
use super::stream::*;
use super::tools::*;
use super::*;
use crate::sampling::SampleConfig;
use crate::server::protocol::*;
use serde_json::json;

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

    let req: ChatCompletionRequest =
        serde_json::from_str(r#"{"messages": [], "max_tokens": 10, "max_completion_tokens": 20}"#)
            .unwrap();
    assert_eq!(req.max_completion_tokens.or(req.max_tokens), Some(20));
}
