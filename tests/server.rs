//! Model-free tests for the HTTP server: routes on ntex's in-process test server over a stand-in `Scorer`.
use anyhow::{Result, bail};
use julia1::Request;
use julia1::server::{Scorer, ServerConfig, ServerState, routes, start_server};
use ntex::client::ClientResponse;
use ntex::http::StatusCode;
use ntex::web::App;
use ntex::web::test::{self, TestServer};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Scores every row with uniform logits; optionally parks inside the forward until released.
#[derive(Default)]
struct Mock {
    fail_once: AtomicBool,
    reject_encoding: AtomicBool,
    forwards: AtomicUsize,
    running: AtomicUsize,
    max_running: AtomicUsize,
    gate: Option<Gate>,
}

#[derive(Default)]
struct Gate(Mutex<bool>, Condvar);

impl Gate {
    fn open(&self) {
        *self.0.lock().unwrap() = true;
        self.1.notify_all();
    }
    fn wait(&self) {
        let mut open = self.0.lock().unwrap();
        while !*open {
            open = self.1.wait(open).unwrap();
        }
    }
}

impl Scorer for Mock {
    fn token_lengths(&self, rows: &[Request]) -> Result<Vec<usize>> {
        if self.reject_encoding.load(Ordering::SeqCst) {
            bail!("state does not fit the strict encoding budget");
        }
        Ok(rows.iter().map(|_| 10).collect())
    }

    fn logits(&self, rows: &[Request]) -> Result<Vec<Vec<f32>>> {
        let now = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_running.fetch_max(now, Ordering::SeqCst);
        self.forwards.fetch_add(1, Ordering::SeqCst);
        if let Some(g) = &self.gate {
            g.wait();
        }
        self.running.fetch_sub(1, Ordering::SeqCst);
        if self.fail_once.swap(false, Ordering::SeqCst) {
            bail!("simulated forward failure");
        }
        Ok(rows.iter().map(|r| vec![0.0; r.options.len()]).collect())
    }
}

fn body() -> String {
    json!({
        "model": "julia-1",
        "state": "I was charged twice.",
        "questions": {
            "team": {"type": "choice", "instructions": "who?", "criteria": {"billing": "Payments", "access": "Login"}},
            "severity": {"type": "score", "instructions": "how bad?", "criteria": ["low", "medium", "high"]},
            "approved": {"type": "noul", "instructions": "approve?"}
        }
    })
    .to_string()
}

async fn server(mock: Arc<Mock>, config: ServerConfig) -> (TestServer, ServerState) {
    let state = ServerState::new(mock, "julia-1", config);
    let s = state.clone();
    (test::server(async move || App::new().configure(routes(s.clone()))).await, state)
}

async fn post(srv: &TestServer, path: &str, content_type: &str, body: String) -> (StatusCode, Value, ClientResponse) {
    let res = srv.post(path).header("content-type", content_type).send_body(body).await.unwrap();
    let bytes = res.body().limit(usize::MAX).await.unwrap();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (res.status(), value, res)
}

#[ntex::test]
async fn answers_keep_request_order_and_the_alias_is_identical() {
    let mock = Arc::new(Mock::default());
    let (srv, _) = server(mock.clone(), ServerConfig::default()).await;
    let (s1, v1, _) = post(&srv, "/v1/classifier", "application/json", body()).await;
    let (s2, v2, _) = post(&srv, "/v1/systemone", "application/json", body()).await;
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK));
    assert_eq!(v1, v2);
    assert_eq!(v1["model"], "julia-1");
    let ids: Vec<&String> = v1["answers"].as_object().unwrap().keys().collect();
    assert_eq!(ids, ["team", "severity", "approved"]);
    assert_eq!(v1["answers"]["team"]["type"], "choice");
    assert_eq!(v1["answers"]["team"]["probabilities"]["billing"], 0.5);
    assert_eq!(v1["answers"]["approved"]["noul"], 0.5);
    assert_eq!(v1["usage"], json!({"input_tokens": 30, "output_tokens": 0}));
}

#[ntex::test]
async fn health_reports_the_model_without_a_forward() {
    let mock = Arc::new(Mock::default());
    let (srv, _) = server(mock.clone(), ServerConfig::default()).await;
    let res = srv.get("/health").send().await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let v: Value = serde_json::from_slice(&res.body().await.unwrap()).unwrap();
    assert_eq!(v, json!({"status": "ready", "model": "julia-1"}));
    assert_eq!(mock.forwards.load(Ordering::SeqCst), 0);
}

#[ntex::test]
async fn invalid_requests_are_422_before_any_forward() {
    let mock = Arc::new(Mock::default());
    let (srv, _) = server(mock.clone(), ServerConfig { max_request_branches: 2, ..Default::default() }).await;
    let mut wrong_model: Value = serde_json::from_str(&body()).unwrap();
    wrong_model["model"] = json!("other");
    let mut bad_state: Value = serde_json::from_str(&body()).unwrap();
    bad_state["state"] = json!(7);
    let mut bad_question: Value = serde_json::from_str(&body()).unwrap();
    bad_question["questions"] = json!({"team": {"type": "rank", "instructions": "who?"}});
    let big = json!({"state": "x".repeat(2 * 1024 * 1024), "questions": {}}).to_string();

    let cases = [
        ("model", "Loaded model is 'julia-1'", "application/json", wrong_model.to_string()),
        ("state", "state must be", "application/json", bad_state.to_string()),
        ("questions", "Unsupported question type", "application/json", bad_question.to_string()),
        ("questions", "3 questions exceed the 2-branch", "application/json", body()),
        ("body", "invalid JSON", "application/json", "{nope".to_string()),
        ("Content-Type", "application/json", "text/plain", body()),
        ("body", "1 MiB", "application/json", big),
    ];
    for (param, hint, content_type, payload) in cases {
        let (status, v, _) = post(&srv, "/v1/classifier", content_type, payload).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{hint}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["code"], 422);
        assert_eq!(v["error"]["param"], param, "{v}");
        assert!(v["error"]["message"].as_str().unwrap().contains(hint), "{hint}: {v}");
    }
    mock.reject_encoding.store(true, Ordering::SeqCst);
    let one_question = json!({"state": "s", "questions": {"approved": {"type": "noul", "instructions": "approve?"}}}).to_string();
    let (status, v, _) = post(&srv, "/v1/classifier", "application/json", one_question).await;
    assert_eq!((status, v["error"]["param"].as_str()), (StatusCode::UNPROCESSABLE_ENTITY, Some("state")));
    assert_eq!(mock.forwards.load(Ordering::SeqCst), 0, "rejected requests never reach the model");
}

#[ntex::test]
async fn forward_failure_is_500_and_the_server_recovers() {
    let mock = Arc::new(Mock { fail_once: AtomicBool::new(true), ..Default::default() });
    let (srv, _) = server(mock.clone(), ServerConfig::default()).await;
    let (status, v, _) = post(&srv, "/v1/classifier", "application/json", body()).await;
    assert_eq!((status, v), (StatusCode::INTERNAL_SERVER_ERROR, json!({"detail": "internal error"})));
    let (status, ..) = post(&srv, "/v1/classifier", "application/json", body()).await;
    assert_eq!(status, StatusCode::OK);
}

#[ntex::test]
async fn full_queue_is_429_with_retry_after_and_forwards_stay_serial() {
    let mock = Arc::new(Mock { gate: Some(Gate::default()), ..Default::default() });
    let (srv, state) = server(mock.clone(), ServerConfig { max_queued: 1, ..Default::default() }).await;
    let srv = Arc::new(srv);
    let spawn_request = || {
        let srv = srv.clone();
        ntex::rt::spawn(async move { post(&srv, "/v1/classifier", "application/json", body()).await.0 })
    };

    let a = spawn_request(); // in flight, parked inside the forward
    while mock.running.load(Ordering::SeqCst) == 0 {
        ntex::time::sleep(ntex::time::Millis(5)).await;
    }
    let b = spawn_request(); // takes the single waiting slot
    while state.admission_available_slots() > 0 {
        ntex::time::sleep(ntex::time::Millis(5)).await;
    }
    let (status, v, res) = post(&srv, "/v1/classifier", "application/json", body()).await;
    assert_eq!((status, v), (StatusCode::TOO_MANY_REQUESTS, json!({"detail": "Scoring queue is full"})));
    assert_eq!(res.header("retry-after").unwrap(), "1");

    mock.gate.as_ref().unwrap().open();
    assert_eq!(a.await.unwrap(), StatusCode::OK);
    assert_eq!(b.await.unwrap(), StatusCode::OK);
    assert_eq!(mock.forwards.load(Ordering::SeqCst), 2, "the 429 request never reached the model");
    assert_eq!(mock.max_running.load(Ordering::SeqCst), 1, "one forward at a time");
}

#[ntex::test]
async fn stop_lets_an_in_flight_request_finish_then_refuses_connections() {
    let mock = Arc::new(Mock { gate: Some(Gate::default()), ..Default::default() });
    let handle = start_server("127.0.0.1", 0, mock.clone(), "julia-1", ServerConfig::default()).await.unwrap();
    assert_ne!(handle.port, 0);
    let addr = format!("127.0.0.1:{}", handle.port);
    let request = {
        let addr = addr.clone();
        ntex::rt::spawn_blocking(move || raw_post(&addr, &body()))
    };
    while mock.running.load(Ordering::SeqCst) == 0 {
        ntex::time::sleep(ntex::time::Millis(5)).await;
    }
    let gate = mock.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        gate.gate.as_ref().unwrap().open();
    });
    handle.stop().await;
    let status = request.await.unwrap();
    assert_eq!(status, Some(200), "the in-flight request completes during graceful stop");
    assert!(std::net::TcpStream::connect(&addr).is_err(), "the listener is closed after stop");
}

/// Bare HTTP/1.1 POST (no client dependency): the status code, or `None` if the connection failed.
fn raw_post(addr: &str, body: &str) -> Option<u16> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(addr).ok()?;
    let req = format!(
        "POST /v1/classifier HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).ok()?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw).ok()?;
    raw.lines().next()?.split_whitespace().nth(1)?.parse().ok()
}
