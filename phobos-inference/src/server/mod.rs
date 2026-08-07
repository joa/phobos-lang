pub mod handlers;
pub mod protocol;

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use serde_json::{Value, json};
use std::sync::mpsc;

use crate::chat::Dialect;
use crate::generate::{self, Flow};
use crate::model::Model;
use crate::sampling::Rng;

use handlers::{handle_chat_completions, handle_completions};
pub use protocol::Defaults;
use protocol::SampleOverrides;

pub(crate) struct GenerationRequest {
    pub(crate) prompt: String,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) sample: SampleOverrides,
    pub(crate) seed: Option<u64>,
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

pub(crate) struct InferenceRequest {
    pub(crate) req: GenerationRequest,
    pub(crate) responder:
        tokio::sync::mpsc::UnboundedSender<std::result::Result<InferenceResponse, String>>,
}

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) tx: mpsc::SyncSender<InferenceRequest>,
    pub(crate) model: String,
    pub(crate) dialect: Dialect,
    pub(crate) bos: Option<String>,
}

pub(crate) async fn root_handler() -> &'static str {
    "phobos-inference"
}

pub(crate) async fn models_handler(State(state): State<AppState>) -> Json<Value> {
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

pub(crate) async fn fallback_handler(
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

pub fn serve(addr: String, model: Box<dyn Model>, defaults: Defaults) -> Result<()> {
    let (tx, rx) = mpsc::sync_channel::<InferenceRequest>(100);
    let info = model.info();
    let name = info.label.clone();
    let dialect = Dialect::detect(info.chat_template.as_deref());
    let bos = model.tokenizer().bos_text().map(str::to_string);

    let state = AppState {
        tx,
        model: name.clone(),
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

    for inf_req in rx {
        let req = inf_req.req;
        let responder = inf_req.responder;

        let mut rng = Rng::new(req.seed.unwrap_or(defaults.seed));
        let config = generate::Config {
            sample: req.sample.resolve(&defaults.sample),
            max_tokens: req.max_tokens.unwrap_or(defaults.max_tokens),
        };

        let Ok(ids) = model.tokenizer().encode(&req.prompt) else {
            let _ = responder.send(Err("Failed to encode prompt".to_string()));
            continue;
        };
        let Ok(mut session) = model.session() else {
            let _ = responder.send(Err("Failed to start inference".to_string()));
            continue;
        };

        let _ = responder.send(Ok(InferenceResponse::Start {
            model: name.clone(),
            prompt_tokens: ids.len(),
        }));

        let mut sink =
            |text: &str| match responder.send(Ok(InferenceResponse::Chunk(text.to_string()))) {
                Ok(()) => Flow::Continue,
                Err(_) => Flow::Stop,
            };
        let outcome = generate::generate(
            model.as_ref(),
            session.as_mut(),
            &ids,
            &config,
            &mut rng,
            &mut sink,
        );

        match outcome {
            Ok(outcome) => {
                let _ = responder.send(Ok(InferenceResponse::Done {
                    reason: outcome.stop.finish_reason().to_string(),
                    completion_tokens: outcome.tokens,
                }));
            }
            Err(_) => {
                let _ = responder.send(Ok(InferenceResponse::Done {
                    reason: "error".to_string(),
                    completion_tokens: 0,
                }));
            }
        }
        // Dropping the session returns its device allocations; see `Model`.
    }

    Ok(())
}

pub(crate) fn dispatch(
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
