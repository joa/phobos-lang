pub mod handlers;
pub mod protocol;

#[cfg(test)]
mod tests;

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use phobos_base::log::Level;

use crate::chat::{Dialect, TURN_START};
use crate::generate::{self, Flow};
use crate::model::{CacheStats, Model, Session, Tokenizer};
use crate::sampling::Rng;
use crate::telemetry::{Fixed, Meter};

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
    pub(crate) meter: Arc<Meter>,
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
    State(state): State<AppState>,
    uri: axum::http::Uri,
    method: axum::http::Method,
    body: axum::body::Bytes,
) -> StatusCode {
    state.meter.log(Level::Info, format!("404 {method} {uri}"));
    let body = String::from_utf8_lossy(&body);
    if !body.is_empty() {
        state.meter.log(Level::Debug, format!("body: {body}"));
    }
    StatusCode::NOT_FOUND
}

/// Rewind `session` to where `held` and `ids` stop agreeing.
///
/// Returns the session and the positions it kept, or `None` if it should be
/// dropped for a fresh one. Stops at least one position short of the whole
/// prompt, since decoding needs fresh logits for the last prompt position.
fn rewind<'a>(
    mut session: Box<dyn Session + 'a>,
    held: &[i64],
    ids: &[i64],
    enabled: bool,
) -> Option<(Box<dyn Session + 'a>, usize)> {
    if !enabled {
        return None;
    }
    let shared = common_prefix(held, ids).min(ids.len().saturating_sub(1));
    // Drop a session that shares nothing, to free its buffers.
    if shared == 0 {
        return None;
    }
    let kept = session.truncate(shared).filter(|&kept| kept > 0)?;
    Some((session, kept))
}

/// Where the prompt's last turn opens, used as the session checkpoint.
///
/// The next request repeats everything before this point, but may render
/// this turn differently once it is history, since earlier turns lose their
/// reasoning. `None` for a prompt with no turns, such as a raw completion.
fn last_turn(tokenizer: &dyn Tokenizer, ids: &[i64]) -> Option<usize> {
    let [turn] = tokenizer.encode(TURN_START).ok()?[..] else {
        return None;
    };
    ids.iter().rposition(|&id| id == turn)
}

fn common_prefix(held: &[i64], ids: &[i64]) -> usize {
    held.iter().zip(ids).take_while(|(a, b)| a == b).count()
}

/// A log line saying where a new prompt diverges from the kept session, with
/// the text around that point. `None` when the prompt extends the session.
/// It shows which part of the chat rendering changed.
fn divergence(tokenizer: &dyn Tokenizer, held: &[i64], ids: &[i64]) -> Option<String> {
    const CONTEXT_TOKENS: usize = 12;
    let at = common_prefix(held, ids);
    if at == held.len() {
        return None;
    }
    let from = at.saturating_sub(CONTEXT_TOKENS);
    let window = |ids: &[i64]| tokenizer.decode(&ids[from..(at + CONTEXT_TOKENS).min(ids.len())]);
    Some(format!(
        "prompt leaves the kept session at token {at} of {}: held {:?}, prompt {:?}",
        held.len(),
        window(held),
        window(ids),
    ))
}

/// A one-line summary of a request's expert cache lookups, from the counts
/// before and after it. `None` for a model that streams no experts.
///
/// Decode and prompt rates are reported separately, since they count
/// different things.
fn describe_experts(before: &CacheStats, after: &CacheStats) -> Option<String> {
    let decode = (
        after.expert_hits - before.expert_hits,
        after.expert_misses - before.expert_misses,
    );
    let prompt = (
        after.expert_prompt_hits - before.expert_prompt_hits,
        after.expert_prompt_misses - before.expert_prompt_misses,
    );
    if decode == (0, 0) && prompt == (0, 0) {
        return None;
    }
    let part = |(hits, misses): (u64, u64)| match hits + misses {
        0 => "none".to_string(),
        lookups => format!("{:.1}% of {lookups}", 100.0 * hits as f64 / lookups as f64),
    };
    let host = match after.expert_cpu_misses - before.expert_cpu_misses {
        0 => String::new(),
        misses => format!(
            "; {misses} computed on the host in {:.0} ms",
            (after.expert_cpu_nanos - before.expert_cpu_nanos) as f64 / 1e6
        ),
    };
    Some(format!(
        "experts resident: decode {}, prompt {}; {:.2} GB copied{host}",
        part(decode),
        part(prompt),
        (after.expert_bytes - before.expert_bytes) as f64 / 1e9,
    ))
}

/// A one-line summary of a finished request: token counts and rates.
fn describe(done: &crate::telemetry::Request) -> String {
    format!(
        "{} prompt ({} reused) + {} generated, {:.1} pp, {:.1} tg tok/s, stopped on {}",
        done.prompt_tokens,
        done.reused,
        done.completion_tokens,
        done.prefill_rate,
        done.decode_rate,
        done.reason,
    )
}

/// How long the idle worker waits for a request before waking.
///
/// It wakes to refresh the device memory reading and to notice a request to
/// quit.
const IDLE_TICK: Duration = Duration::from_millis(250);

pub fn serve(
    addr: String,
    model: Box<dyn Model>,
    defaults: Defaults,
    meter: Arc<Meter>,
) -> Result<()> {
    let reuse = defaults.prefix_cache;
    let (tx, rx) = mpsc::sync_channel::<InferenceRequest>(100);
    let info = model.info();
    let name = info.label.clone();
    let dialect = Dialect::detect(info.chat_template.as_deref());
    let bos = model.tokenizer().bos_text().map(str::to_string);

    meter.describe(Fixed::of(model.as_ref(), Some(&addr)));
    meter.set_device_memory(model.device_memory());
    meter.set_cache_stats(model.cache_stats());

    let state = AppState {
        tx,
        model: name.clone(),
        dialect,
        bos,
        meter: meter.clone(),
    };

    meter.log(
        Level::Info,
        format!("request defaults: {}", defaults.describe()),
    );
    meter.log(Level::Info, format!("chat dialect: {dialect:?}"));

    let serving = meter.clone();
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
            serving.log(Level::Info, format!("listening on http://{addr}"));
            axum::serve(listener, app).await.unwrap();
        });
    });

    // After the listener, so the port is open during the upload. Requests
    // arriving meanwhile wait in the channel.
    meter.log(Level::Info, "uploading weights");
    let started = Instant::now();
    model.warm_up()?;
    meter.log(
        Level::Info,
        format!("warmed up in {:.1} s", started.elapsed().as_secs_f64()),
    );
    meter.set_device_memory(model.device_memory());
    meter.set_cache_stats(model.cache_stats());

    // The last request's session and the tokens it holds.
    let mut kept: Option<(Box<dyn Session + '_>, Vec<i64>)> = None;

    // Not `for req in rx`: the worker must wake between requests to refresh
    // the meter and to notice a request to quit.
    while meter.running() {
        let inf_req = match rx.recv_timeout(IDLE_TICK) {
            Ok(req) => req,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                meter.set_device_memory(model.device_memory());
                meter.set_cache_stats(model.cache_stats());
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let req = inf_req.req;
        let responder = inf_req.responder;

        let mut rng = Rng::new(req.seed.unwrap_or(defaults.seed));
        let Ok(ids) = model.tokenizer().encode(&req.prompt) else {
            let _ = responder.send(Err("Failed to encode prompt".to_string()));
            continue;
        };
        let config = generate::Config {
            sample: req.sample.resolve(&defaults.sample),
            max_tokens: req.max_tokens.unwrap_or(defaults.max_tokens),
            meter: Some(meter.clone()),
            checkpoint_at: last_turn(model.tokenizer(), &ids),
        };
        // Reuse the kept session as far as the two prompts agree. It is
        // rewound past that point, or dropped.
        if let Some((_, held)) = &kept
            && let Some(line) = divergence(model.tokenizer(), held, &ids)
        {
            meter.log(Level::Info, line);
        }
        let reused = match kept.take() {
            Some((session, held)) => match rewind(session, &held, &ids, reuse) {
                Some((session, at)) => {
                    kept = Some((session, held));
                    at
                }
                None => 0,
            },
            None => 0,
        };
        let mut session = match kept.take() {
            Some((session, _)) => session,
            None => match model.session() {
                Ok(session) => session,
                Err(_) => {
                    let _ = responder.send(Err("Failed to start inference".to_string()));
                    continue;
                }
            },
        };

        meter.request_started(ids.len(), reused);
        let stats_before = model.cache_stats();
        let _ = responder.send(Ok(InferenceResponse::Start {
            model: name.clone(),
            prompt_tokens: ids.len(),
        }));

        // Stops when the client hangs up or a quit is requested. Checked per
        // token; a prompt pass cannot be interrupted part way.
        let mut sink = |text: &str| {
            if !meter.running() {
                return Flow::Stop;
            }
            match responder.send(Ok(InferenceResponse::Chunk(text.to_string()))) {
                Ok(()) => Flow::Continue,
                Err(_) => Flow::Stop,
            }
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
                let reason = outcome.stop.finish_reason().to_string();
                if let Some(done) = meter.request_finished(&reason) {
                    meter.log(Level::Info, describe(&done));
                }
                if let (Some(before), Some(after)) = (stats_before, model.cache_stats())
                    && let Some(line) = describe_experts(&before, &after)
                {
                    meter.log(Level::Info, line);
                }
                // Keep the session for the next request, likely the same
                // conversation plus a turn. `held` is what the session holds,
                // which can differ from what was emitted.
                if reuse {
                    meter.set_cache(session.len(), session.cache_bytes());
                    kept = Some((session, outcome.held));
                } else {
                    drop(session);
                    meter.set_cache(0, None);
                }
                let _ = responder.send(Ok(InferenceResponse::Done {
                    reason,
                    completion_tokens: outcome.tokens,
                }));
            }
            Err(e) => {
                meter.log(Level::Info, format!("generation failed: {e:#}"));
                meter.request_finished("error");
                // A failed pass leaves the session in an unknown state.
                drop(session);
                meter.set_cache(0, None);
                let _ = responder.send(Ok(InferenceResponse::Done {
                    reason: "error".to_string(),
                    completion_tokens: 0,
                }));
            }
        }
        meter.set_device_memory(model.device_memory());
        meter.set_cache_stats(model.cache_stats());
    }
    // Drop before `model`, which it borrows.
    drop(kept);

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
