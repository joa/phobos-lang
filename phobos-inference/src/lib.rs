//! The inference runtime: what a model has to be, and everything driving one.
//!
//! This crate defines [`Model`], [`Session`] and [`Tokenizer`], and nothing in
//! it names a model format. `phobos-gguf` and `phobos-onnx` implement the
//! traits; `phobos-cli` is the binary that decides which of them to load.

pub mod chat;
pub mod generate;
pub mod model;
pub mod sampling;
pub mod server;

pub use model::{Model, ModelInfo, Session, Tokenizer};
