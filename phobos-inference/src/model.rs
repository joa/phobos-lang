use anyhow::Result;

pub struct ModelInfo {
    pub label: String,
    pub backend: &'static str,
    pub vocab_size: usize,
    pub context_limit: usize,
    pub chat_template: Option<String>,
}

pub trait Model {
    fn info(&self) -> &ModelInfo;

    fn tokenizer(&self) -> &dyn Tokenizer;

    fn session(&self) -> Result<Box<dyn Session + '_>>;
}

pub trait Session {
    /// Run `ids` and return the logits for the position after the last one.
    ///
    /// The prompt is one call and each generated token another, so a caller
    /// never has to keep a model and a state in step by hand. An implementation
    /// is free to split a long call into batches; that is a property of the
    /// backend, not of the interface.
    fn extend(&mut self, ids: &[i64]) -> Result<Vec<f32>>;

    /// Tokens consumed so far, prompt included.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>>;

    fn decode_bytes(&self, ids: &[i64]) -> Vec<u8>;

    fn is_eog(&self, id: i64) -> bool;

    fn bos_text(&self) -> Option<&str> {
        None
    }

    fn decode(&self, ids: &[i64]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}
