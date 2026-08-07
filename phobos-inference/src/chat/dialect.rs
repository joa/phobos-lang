//! Which chat family a model belongs to, detected from its own template.

pub(crate) const THINK_START: &str = "<think>";

pub(crate) const THINK_END: &str = "</think>";

pub(crate) const TOOL_CALL_START: &str = "<tool_call>";

pub(crate) const TOOL_CALL_END: &str = "</tool_call>";

pub(crate) const FUNCTION_START: &str = "<function=";

pub(crate) const FUNCTION_END: &str = "</function>";

pub(crate) const PARAMETER_START: &str = "<parameter=";

pub(crate) const PARAMETER_END: &str = "</parameter>";

/// MiniCPM5's call syntax: one self-contained element, the name in an
/// attribute, and no `<tool_call>` wrapper.
pub(crate) const MINICPM_FUNCTION_START: &str = "<function name=\"";

pub(crate) const MINICPM_PARAM_START: &str = "<param name=\"";

pub(crate) const MINICPM_PARAM_END: &str = "</param>";

pub(crate) const CDATA_START: &str = "<![CDATA[";

pub(crate) const CDATA_END: &str = "]]>";

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
    pub(crate) fn prefills_empty_think(self) -> bool {
        self == Dialect::Qwen
    }

    /// The markers that open and close a call in the generated text.
    pub(crate) fn call_markers(self) -> (&'static str, &'static str) {
        match self {
            Dialect::Qwen => (TOOL_CALL_START, TOOL_CALL_END),
            Dialect::MiniCpm => (MINICPM_FUNCTION_START, FUNCTION_END),
        }
    }
}
