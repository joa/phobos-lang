use serde_json::{Value, json};

use super::dialect::{
    CDATA_END, CDATA_START, Dialect, FUNCTION_END, FUNCTION_START, MINICPM_FUNCTION_START,
    MINICPM_PARAM_END, MINICPM_PARAM_START, PARAMETER_END, PARAMETER_START, TOOL_CALL_END,
    TOOL_CALL_START,
};
use super::{ToolCall, ToolCallFunction};

/// The tool-calling contract, verbatim from the model's own
/// `tokenizer.chat_template`. Qwen3.5 is trained on an XML call syntax, not the
/// JSON one most servers ask for, and asking for JSON gets a blend of the two
/// back.
pub(crate) const TOOL_FORMAT_INSTRUCTION: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

pub(crate) fn default_tool_call_type() -> String {
    "function".to_string()
}

pub(crate) fn has_tools(tools: Option<&Value>) -> bool {
    tools.is_some_and(|tools| tools.as_array().is_some_and(|list| !list.is_empty()))
}

pub(crate) fn selected_tools(tools: Option<&Value>, tool_choice: Option<&Value>) -> Option<Value> {
    if tool_choice.and_then(Value::as_str) == Some("none") {
        return None;
    }
    tools.filter(|tools| has_tools(Some(tools))).cloned()
}

pub(crate) fn tool_choice_instruction(tool_choice: Option<&Value>) -> Option<String> {
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
pub(crate) const MINICPM_TOOL_GUIDELINES: &str = "\n\nTool usage guidelines:\n- You may call zero or more functions. If no function calls are needed, just answer normally and do not include any <function ... </function>.\n- When calling a function, return an XML object within <function ... </function> using:\n<function name=\"function-name\"><param name=\"param-name\">param-value</param></function>\n- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a CDATA block: <param name=\"param-name\"><![CDATA[...multi-line value...]]></param>";

/// Where MiniCPM5's template substitutes the tool definitions when the
/// caller's system prompt places them explicitly.
pub(crate) const TOOL_DEF_SEP: &str = "<tool_def_sep>";

/// The tool list and call format, without the surrounding system turn.
pub(crate) fn render_tool_definitions(tools: &Value, dialect: Dialect) -> String {
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
pub(crate) fn render_tools_system(
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

pub(crate) fn argument_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Whether a value needs CDATA to survive MiniCPM5's form, the rule its own
/// template states.
pub(crate) fn needs_cdata(value: &str) -> bool {
    value.contains(['<', '&', '\n'])
}

pub(crate) fn render_tool_calls(
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

/// The declared type of one tool parameter, when the caller sent a schema.
pub(crate) fn schema_type<'a>(
    tools: Option<&'a Value>,
    function: &str,
    parameter: &str,
) -> Option<&'a str> {
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
pub(crate) fn coerce_argument(
    text: &str,
    function: &str,
    parameter: &str,
    tools: Option<&Value>,
) -> Value {
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
pub(crate) fn parse_function_block(
    raw: &str,
    tools: Option<&Value>,
    index: usize,
) -> Option<ToolCall> {
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
pub(crate) fn parse_minicpm_function(
    raw: &str,
    tools: Option<&Value>,
    index: usize,
) -> Option<ToolCall> {
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
pub(crate) fn parse_json_call(json_value: &Value, index: usize) -> Option<ToolCall> {
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
