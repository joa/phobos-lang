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

/// Put `session` back to where `held` and `ids` stop agreeing, if it can go
/// there and there is anything to gain.
///
/// Returns the session and the number of positions it kept, or nothing if it
/// should be dropped and a fresh one started. One position short of the whole
/// prompt on purpose: the token that decoding starts from is chosen from the
/// logits of the last prompt position, and those have to come from a pass run
/// now rather than from whatever the session last did.
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
    // Nothing to keep is not worth keeping: a session holding none of this
    // prompt still holds its buffers, and dropping it hands them back.
    if shared == 0 {
        return None;
    }
    let kept = session.truncate(shared).filter(|&kept| kept > 0)?;
    Some((session, kept))
}

/// Where the prompt's last turn opens, the reply's own header. The next
/// request repeats everything before it, but may render this turn
/// differently once it is history: an earlier turn loses its reasoning. So
/// that is where a session that cannot rewind by position keeps its
/// checkpoint. `None` for a prompt with no turns, such as a raw completion.
fn last_turn(tokenizer: &dyn Tokenizer, ids: &[i64]) -> Option<usize> {
    let [turn] = tokenizer.encode(TURN_START).ok()?[..] else {
        return None;
    };
    ids.iter().rposition(|&id| id == turn)
}

fn common_prefix(held: &[i64], ids: &[i64]) -> usize {
    held.iter().zip(ids).take_while(|(a, b)| a == b).count()
}

/// Where a new prompt stops agreeing with what the kept session holds, with
/// the text on either side of that point, or `None` when the prompt extends
/// it. A backend that cannot rewind starts over from nothing whenever the
/// two differ, and the text says which part of the chat rendering moved.
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

/// A request's expert cache lookups in a line, from the counts before and
/// after it, or `None` for a model that streams no experts. A decode step
/// counts one lookup per routed expert, a prompt pass one per expert it
/// routes to, so the two rates are given apart.
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
    Some(format!(
        "experts resident: decode {}, prompt {}; {:.2} GB copied",
        part(decode),
        part(prompt),
        (after.expert_bytes - before.expert_bytes) as f64 / 1e9,
    ))
}

/// One finished request in a line: the two rates a reader compares runs by,
/// and the counts they were measured over.
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

/// How long the worker waits for a request before looking up from it.
///
/// It wakes to re-read the card, which is what keeps a memory gauge moving
/// while nothing is being generated, and to notice that a viewer has asked to
/// quit. Short enough to feel live, long enough that an idle server is idle.
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

    // After the listener, so the port is open while the weights go up; a
    // request that arrives meanwhile waits in the channel.
    meter.log(Level::Info, "uploading weights");
    let started = Instant::now();
    model.warm_up()?;
    meter.log(
        Level::Info,
        format!("warmed up in {:.1} s", started.elapsed().as_secs_f64()),
    );
    meter.set_device_memory(model.device_memory());
    meter.set_cache_stats(model.cache_stats());

    // The session from the last request, with the tokens it holds, for as
    // long as nothing has replaced it.
    let mut kept: Option<(Box<dyn Session + '_>, Vec<i64>)> = None;

    // Not `for req in rx`: the worker has to look up between requests, both
    // to re-read the card and to notice a viewer asking to quit.
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
        // A session kept from the last request is worth reusing for as far as
        // the two prompts agree. What it holds past that is wrong for this
        // one, so it is either rewound or given up.
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

        // Stops for either reason a generation stops being wanted: the client
        // hung up, or a viewer asked to quit. Checked per token, so quitting
        // costs one step rather than the rest of the response. A prompt pass
        // is one call and cannot be interrupted part way.
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
                // Kept for the next request, which may well be this
                // conversation with one more turn on the end. `held` is what
                // the session holds and not what was emitted; the two differ
                // whenever a generation stops without feeding its last token
                // back.
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
                // A failed pass leaves the session holding who knows what, so
                // it goes rather than being reused.
                drop(session);
                meter.set_cache(0, None);
                let _ = responder.send(Ok(InferenceResponse::Done {
                    reason: "error".to_string(),
                    completion_tokens: 0,
                }));
            }
        }
        // Whatever was released is back with the backend by now; see `Model`.
        meter.set_device_memory(model.device_memory());
        meter.set_cache_stats(model.cache_stats());
    }
    // Before `model`, which it borrows.
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
