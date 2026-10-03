//! HTTP server for the named-question API (native only), built on ntex.
//!
//! * `POST /v1/classifier` (alias `POST /v1/systemone`): `{"model"?, "state", "questions"}` where
//!   `state` is text or a JSON object/array and `questions` is the `predict_typed` mapping. Answers
//!   come back as `{"model", "answers": {id: {type, choice|score|noul, probabilities, ..}}, "usage"}`.
//! * `GET /health`: `{"status":"ready","model":..}`, no inference.
//!
//! Requests run strictly one at a time (one forward in flight) behind a bounded admission queue: a
//! request that finds `1 + max_queued` requests already admitted gets `429` with `Retry-After: 1`.
//! Validation failures (bad JSON, bad questions, strict-encoding rejections) are `422` with
//! `{"error": {message, type: "invalid_request_error", code: 422, param}}`; a failed forward is
//! `500 {"detail":"internal error"}`.
//!
//! [`start_server`] must run inside an ntex runtime. ntex owns SIGINT/SIGTERM: the listener stops,
//! in-flight requests (including a running forward) finish, and [`ServerHandle::wait`] resolves.

use crate::engine::Engine;
use crate::request::{Request, State};
use crate::typed::{Answers, answers_from_logits, typed_rows};
use anyhow::Result;
use ntex::http::{StatusCode, header};
use ntex::server::Server;
use ntex::util::BytesMut;
use ntex::web::types::{Payload, State as WebState};
use ntex::web::{self, App, HttpRequest, HttpResponse, HttpServer, ServiceConfig};
use serde_json::{Map, Value, json};
use std::net::TcpListener;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};

/// Request-body cap.
pub const MAX_BODY_BYTES: usize = 1_048_576;
/// How much of a rejected (oversized) body is read and discarded before replying.
const DRAIN_LIMIT: usize = 16 * MAX_BODY_BYTES;

/// What the server needs from a model; [`Engine`] implements it, tests use a stand-in.
pub trait Scorer: Send + Sync + 'static {
    /// Token length of each row. An `Err` means the request itself is invalid (strict encoding
    /// rejected it) and is reported as `422` before any forward pass.
    fn token_lengths(&self, rows: &[Request]) -> Result<Vec<usize>>;
    /// One forward pass over all rows (synchronous; runs on the blocking pool).
    fn logits(&self, rows: &[Request]) -> Result<Vec<Vec<f32>>>;
}

impl Scorer for Engine {
    fn token_lengths(&self, rows: &[Request]) -> Result<Vec<usize>> {
        Ok(self.encode(rows)?.iter().map(|e| e.ids.len()).collect())
    }

    fn logits(&self, rows: &[Request]) -> Result<Vec<Vec<f32>>> {
        Engine::logits(self, rows)
    }
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Max questions per request.
    pub max_request_branches: usize,
    /// Waiting slots on top of the one in-flight request.
    pub max_queued: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self { max_request_branches: 100, max_queued: 16 }
    }
}

#[derive(Debug)]
pub enum ExecError {
    /// 422: the request is not answerable (`param` names the offending field).
    Invalid { param: &'static str, message: String },
    /// 429: the admission queue is full.
    QueueFull,
    /// 500: the forward pass failed.
    Internal(String),
}

struct Inner {
    scorer: Arc<dyn Scorer>,
    model_name: String,
    config: ServerConfig,
    model_lock: Arc<Mutex<()>>,
    admission: Arc<Semaphore>,
}

/// Shared state behind the routes (cheap to clone).
#[derive(Clone)]
pub struct ServerState(Arc<Inner>);

impl ServerState {
    pub fn new(scorer: Arc<dyn Scorer>, model_name: impl Into<String>, config: ServerConfig) -> Self {
        let admission = Arc::new(Semaphore::new(config.max_queued.saturating_add(1)));
        Self(Arc::new(Inner { scorer, model_name: model_name.into(), config, model_lock: Arc::new(Mutex::new(())), admission }))
    }

    pub fn model_name(&self) -> &str {
        &self.0.model_name
    }

    /// Admission slots currently free (0 means the next request is rejected with 429).
    pub fn admission_available_slots(&self) -> usize {
        self.0.admission.available_permits()
    }

    /// Validate, queue, and answer one request: `(answers, input_tokens)`.
    pub async fn execute(&self, state: State, questions: Value) -> Result<(Answers, usize), ExecError> {
        let invalid = |param, e: anyhow::Error| ExecError::Invalid { param, message: e.to_string() };
        let count = questions.as_object().map_or(0, Map::len);
        if count > self.0.config.max_request_branches {
            return Err(ExecError::Invalid {
                param: "questions",
                message: format!("{count} questions exceed the {}-branch request limit", self.0.config.max_request_branches),
            });
        }
        let (rows, metadata) = typed_rows(&state, &questions).map_err(|e| invalid("questions", e))?;
        let tokens: usize = self.0.scorer.token_lengths(&rows).map_err(|e| invalid("state", e))?.iter().sum();

        let permit = self.0.admission.clone().try_acquire_owned().map_err(|_| ExecError::QueueFull)?;
        let lock = self.0.model_lock.clone().lock_owned().await;
        let scorer = Arc::clone(&self.0.scorer);
        // The permit and lock move into the blocking closure, so a client disconnect (dropping this
        // future) can never let a second forward overlap one that is still running.
        let scores = ntex::rt::spawn_blocking(move || {
            let scores = scorer.logits(&rows);
            drop(lock);
            drop(permit);
            scores
        })
        .await
        .map_err(|e| ExecError::Internal(format!("forward task failed: {e}")))?
        .map_err(|e| ExecError::Internal(e.to_string()))?;
        let answers = answers_from_logits(metadata, scores).map_err(|e| ExecError::Internal(e.to_string()))?;
        Ok((answers, tokens))
    }
}

fn json_response(status: StatusCode, retry_after: bool, body: Value) -> HttpResponse {
    let mut builder = HttpResponse::build(status);
    builder.content_type("application/json");
    if retry_after {
        builder.header("retry-after", "1");
    }
    builder.body(body.to_string())
}

fn invalid(param: &str, message: impl Into<String>) -> HttpResponse {
    let message = message.into();
    json_response(
        StatusCode::UNPROCESSABLE_ENTITY,
        false,
        json!({"error": {"message": message, "type": "invalid_request_error", "code": 422, "param": param}}),
    )
}

/// Discard the rest of a body we are rejecting, so the client gets our 422 instead of a connection
/// reset mid-upload. Nothing is buffered, and we give up after [`DRAIN_LIMIT`] bytes.
async fn drain(mut payload: Payload) {
    let mut seen = 0;
    while let Some(Ok(chunk)) = payload.recv().await {
        seen += chunk.len();
        if seen > DRAIN_LIMIT {
            break;
        }
    }
}

async fn read_body(req: &HttpRequest, mut payload: Payload) -> Option<Vec<u8>> {
    let declared = req.headers().get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > MAX_BODY_BYTES) {
        drain(payload).await;
        return None;
    }
    let mut buf = BytesMut::new();
    while let Some(chunk) = payload.recv().await {
        let chunk = chunk.ok()?;
        if buf.len() + chunk.len() > MAX_BODY_BYTES {
            drain(payload).await;
            return None;
        }
        buf.extend_from_slice(&chunk);
    }
    Some(buf.to_vec())
}

async fn classify(state: WebState<ServerState>, req: HttpRequest, payload: Payload) -> HttpResponse {
    let Some(bytes) = read_body(&req, payload).await else {
        return invalid("body", "request body exceeds the 1 MiB limit");
    };
    if let Some(ct) = req.headers().get(header::CONTENT_TYPE) {
        let mime = ct.to_str().unwrap_or_default().split(';').next().unwrap_or_default().trim();
        if mime != "application/json" && !mime.ends_with("+json") {
            return invalid("Content-Type", format!("Content-Type must be application/json (got '{mime}')"));
        }
    }
    let body: Value = match serde_json::from_slice(&bytes) {
        Ok(Value::Object(o)) => Value::Object(o),
        Ok(_) => return invalid("body", "request body must be a JSON object"),
        Err(e) => return invalid("body", format!("invalid JSON: {e}")),
    };
    if let Some(model) = body.get("model") {
        if model.as_str() != Some(state.model_name()) {
            return invalid("model", format!("Loaded model is '{}'", state.model_name()));
        }
    }
    let Some(julia_state) = body.get("state").and_then(State::from_value) else {
        return invalid("state", "state must be text or a JSON object/array");
    };
    let Some(questions) = body.get("questions").cloned() else {
        return invalid("questions", "questions is required");
    };
    match state.execute(julia_state, questions).await {
        Ok((answers, input_tokens)) => {
            let answers: Map<String, Value> = answers.iter().map(|(id, a)| (id.clone(), a.to_json())).collect();
            json_response(
                StatusCode::OK,
                false,
                json!({"model": state.model_name(), "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}),
            )
        }
        Err(ExecError::Invalid { param, message }) => invalid(param, message),
        Err(ExecError::QueueFull) => json_response(StatusCode::TOO_MANY_REQUESTS, true, json!({"detail": "Scoring queue is full"})),
        Err(ExecError::Internal(_)) => json_response(StatusCode::INTERNAL_SERVER_ERROR, false, json!({"detail": "internal error"})),
    }
}

async fn health(state: WebState<ServerState>) -> HttpResponse {
    json_response(StatusCode::OK, false, json!({"status": "ready", "model": state.model_name()}))
}

/// Register the routes (and shared state) on an app: `App::new().configure(routes(state))`.
pub fn routes(state: ServerState) -> impl FnOnce(&mut ServiceConfig) {
    move |cfg| {
        cfg.state(state)
            .route("/v1/classifier", web::post().to(classify))
            .route("/v1/systemone", web::post().to(classify))
            .route("/health", web::get().to(health));
    }
}

/// A running server: the bound port, the model identity, and the ntex server.
pub struct ServerHandle {
    pub port: u16,
    pub model_name: String,
    server: Server,
}

impl ServerHandle {
    /// Resolves once the server has stopped (SIGINT/SIGTERM or [`Self::stop`]).
    pub async fn wait(self) {
        let _ = self.server.await;
    }

    /// Graceful stop: no new connections, in-flight requests finish.
    pub async fn stop(self) {
        self.server.stop(true).await;
    }
}

/// Bind `host:port` (port 0 = ephemeral) and serve. The model must already be loaded.
pub async fn start_server(host: &str, port: u16, scorer: Arc<dyn Scorer>, model_name: impl Into<String>, config: ServerConfig) -> Result<ServerHandle> {
    let model_name = model_name.into();
    let state = ServerState::new(scorer, model_name.clone(), config);
    let listener = TcpListener::bind((host, port)).map_err(|e| anyhow::anyhow!("failed to bind {host}:{port}: {e}"))?;
    let port = listener.local_addr()?.port();
    // Inference is serial, so a couple of workers are plenty for connection handling.
    let server = HttpServer::new(async move || App::new().configure(routes(state.clone()))).workers(2).listen(listener)?.run();
    Ok(ServerHandle { port, model_name, server })
}
