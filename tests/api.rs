//! Model-free API tests, ported from the Python suite (test_typed_api.py,
//! router/tests/test_router.py, probability display and row validation).
use anyhow::Result;
use julia1::request::argmax;
use julia1::{Logits, QType, Request, Router, State, display_probabilities, predict_typed};
use serde_json::{Value, json};
use std::cell::RefCell;

/// Returns fixed logits and records the rows it was asked to score.
struct Fixed {
    logits: RefCell<Vec<Vec<f32>>>,
    calls: RefCell<Vec<Vec<Request>>>,
}

impl Fixed {
    fn new(logits: Vec<Vec<f32>>) -> Self {
        Self { logits: RefCell::new(logits), calls: RefCell::new(Vec::new()) }
    }
}

impl Logits for Fixed {
    fn logits(&self, rows: &[Request]) -> Result<Vec<Vec<f32>>> {
        self.calls.borrow_mut().push(rows.to_vec());
        Ok(self.logits.borrow().clone())
    }
}

fn state() -> State {
    State::from("context")
}

#[test]
fn named_questions_keep_ids_and_full_probabilities() {
    let engine = Fixed::new(vec![vec![0., 2.], vec![0., 0., 0.], vec![0., 0.]]);
    let questions = json!({
        "team": {"type": "choice", "instructions": "route", "criteria": {"billing": "Payments", "access": "Login"}},
        "severity": {"type": "score", "instructions": "rate", "criteria": ["low", "medium", "high"]},
        "approved": {"type": "noul", "instructions": "approve?"},
    });
    let answers = predict_typed(&engine, &state(), &questions).unwrap();
    let get = |id: &str| &answers.iter().find(|(k, _)| k == id).unwrap().1;
    assert_eq!(get("team").choice(), Some("access"));
    assert!((get("severity").score().unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(get("approved").noul(), Some(0.5));
    assert!((get("team").probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert_eq!(engine.calls.borrow()[0][0].options, ["Payments", "Login"]);
    // JSON shape matches the Python dict.
    let out = julia1::answers_to_json(&answers);
    assert_eq!(out["answers"]["team"]["choice"], "access");
    assert_eq!(out["answers"]["approved"].get("max_probability"), None);
    assert!(out["answers"]["severity"]["max_probability"].is_number());
}

#[test]
fn invalid_questions_rejected_before_inference() {
    let engine = Fixed::new(vec![]);
    let many: serde_json::Map<String, Value> = (0..21).map(|i| (i.to_string(), json!(i.to_string()))).collect();
    for questions in [
        json!({}),
        json!({"q": {"type": "choice", "instructions": "q", "criteria": {"a": "a"}}}),
        json!({"q": {"type": "choice", "instructions": "q", "criteria": many}}),
        json!({"q": {"type": "other", "instructions": "q", "criteria": {"a": "a", "b": "b"}}}),
        json!({"q": {"type": "choice", "criteria": {"a": "a", "b": "b"}}}),
    ] {
        assert!(predict_typed(&engine, &State::from("s"), &questions).is_err(), "{questions}");
    }
    assert!(engine.calls.borrow().is_empty());
}

#[test]
fn noul_preserves_descriptions_in_false_true_order() {
    let engine = Fixed::new(vec![vec![0., 2.]]);
    let questions = json!({"review": {"type": "noul", "instructions": "Needs human review?",
        "criteria": {"true": "A human should inspect this run.", "false": "No human attention is warranted."}}});
    let answers = predict_typed(&engine, &state(), &questions).unwrap();
    assert_eq!(engine.calls.borrow()[0][0].options, ["No human attention is warranted.", "A human should inspect this run."]);
    assert_eq!(answers[0].1.keys, ["false", "true"]);
    assert!(answers[0].1.noul().unwrap() > 0.5);
}

#[test]
fn noul_rejects_invalid_criteria_before_inference() {
    let engine = Fixed::new(vec![vec![0., 2.]]);
    for criteria in [
        json!({}),
        json!({"true": "yes"}),
        json!({"false": "no", "true": "yes", "other": "maybe"}),
        json!(["no", "yes"]),
        json!({"false": "", "true": "yes"}),
    ] {
        let questions = json!({"q": {"type": "noul", "instructions": "q", "criteria": criteria}});
        assert!(predict_typed(&engine, &state(), &questions).is_err(), "{criteria}");
    }
    assert!(engine.calls.borrow().is_empty());
}

#[test]
fn display_rounds_decisive_answer_and_redistributes_small_values() {
    assert_eq!(display_probabilities(&[0.955, 0.044, 0.001]), [1.0, 0.0, 0.0]);
    assert_ne!(display_probabilities(&[0.951, 0.049]), [1.0, 0.0]);
    let r = display_probabilities(&[0.6, 0.395, 0.005]);
    assert_eq!(r[2], 0.0);
    assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((r[0] / r[1] - 0.6 / 0.395).abs() < 1e-12);
}

#[test]
fn rejects_malformed_rows() {
    let valid = json!({"state": "context", "question": "Choose", "options": ["A", "B"]});
    assert!(Request::from_value(&valid, 1).is_ok());
    let with = |key: &str, v: Value| {
        let mut row = valid.clone();
        row[key] = v;
        row
    };
    for row in [
        Value::Null,
        json!([]),
        with("teacher_logits", json!("bad")),
        with("teacher_logits", json!([true, 0.0])),
        with("options", json!(["A"])),
        with("options", json!(["A", ""])),
        with("state", json!(3)),
        with("type", json!("other")),
        with("type", json!("noul")).as_object().map(|o| {
            let mut o = o.clone();
            o.insert("options".into(), json!(["a", "b", "c"]));
            Value::Object(o)
        }).unwrap(),
        with("target", json!(2)),
    ] {
        assert!(Request::from_value(&row, 1).is_err(), "{row}");
    }
    let parsed = Request::from_value(&with("state", json!({"a": [1, 2.5]})), 1).unwrap();
    assert_eq!(parsed.state.render(), r#"{"a": [1, 2.5]}"#);
    assert_eq!(parsed.qtype, QType::Choice);
}

/// Router test engine: the logit of an option is its numeric label (times `scale`).
struct Numeric {
    scale: f32,
    batches: RefCell<Vec<Vec<Request>>>,
}

impl Numeric {
    fn new(scale: f32) -> Self {
        Self { scale, batches: RefCell::new(Vec::new()) }
    }
}

impl Logits for Numeric {
    fn logits(&self, rows: &[Request]) -> Result<Vec<Vec<f32>>> {
        self.batches.borrow_mut().push(rows.to_vec());
        Ok(rows.iter().map(|r| r.options.iter().map(|o| self.scale * o.parse::<f32>().unwrap()).collect()).collect())
    }
}

fn request(n: usize) -> Request {
    Request::new("Olá, 世界", "Choose the largest number", (0..n).map(|i| i.to_string()).collect(), QType::Choice)
}

#[test]
fn router_direct_and_hierarchical() {
    let engine = Numeric::new(1.0);
    let router = Router::new(&engine, 20, 2, 3, 0, 4096).unwrap();
    let results = router.route_many(&[2, 20, 21, 77, 4096].map(request)).unwrap();
    assert_eq!(results.iter().map(|r| r.index).collect::<Vec<_>>(), [1, 19, 20, 76, 4095]);
    for r in &results {
        assert!((r.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(r.candidates.contains(&r.index));
    }
    assert_eq!(results[0].probability_scope, "all_options");
    assert_eq!(results[4].probability_scope, "final_candidates");
    assert!(engine.batches.borrow().iter().all(|b| b.len() <= 3));
    assert!(engine.batches.borrow().iter().flatten().all(|r| (2..=20).contains(&r.options.len())));
    assert!(router.route_many(&[]).unwrap().is_empty());
}

#[test]
fn router_confident_group_shortcut() {
    assert!(Router::<Numeric>::confident_winner(&[0.0, 4.0, -3.0], 1));
    assert!(!Router::<Numeric>::confident_winner(&[0.0, 3.0, -3.0], 1));
    let engine = Numeric::new(10.0);
    let result = Router::with_defaults(&engine).route(&request(2001)).unwrap();
    assert_eq!((result.index, result.rounds, result.model_rows), (2000, 3, 106));
}

#[test]
fn router_cache_and_dedup() {
    let engine = Numeric::new(1.0);
    let router = Router::new(&engine, 20, 2, 16, 1, 4096).unwrap();
    let results = router.route_many(&[request(5), request(5)]).unwrap();
    assert_eq!((results[0].model_rows, results[0].cache_hits), (1, 1));
    assert_eq!(router.route(&request(5)).unwrap().model_rows, 0);
    router.clear_cache();
    assert_eq!(router.route(&request(5)).unwrap().model_rows, 1);
    router.route(&request(6)).unwrap();
    assert_eq!(router.route(&request(5)).unwrap().model_rows, 1);
}

#[test]
fn router_rejects_invalid_and_bad_engines() {
    let engine = Numeric::new(1.0);
    let router = Router::with_defaults(&engine);
    let mut score = request(30);
    score.qtype = QType::Score;
    let mut noul = request(3);
    noul.qtype = QType::Noul;
    for row in [request(1), score, noul] {
        assert!(router.route(&row).is_err());
    }
    let bad = Fixed::new(vec![vec![0.0]]);
    assert!(Router::with_defaults(&bad).route(&request(4)).is_err());
    assert_eq!(argmax(&[3, 3, -1]), 0);
}
