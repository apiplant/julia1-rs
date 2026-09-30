//! Tests against the real Julia-1 checkpoint (JULIA_CHECKPOINT, default ../../ai/Julia-1).
//! They are skipped when the checkpoint is not present.
use julia1::encode::Encoder;
use julia1::{Device, Engine, EngineOptions, QType, Request, State};
use serde_json::json;
use std::path::PathBuf;
use std::sync::OnceLock;

fn checkpoint() -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var("JULIA_CHECKPOINT").unwrap_or_else(|_| "../../ai/Julia-1".into()));
    if path.join("model.safetensors").is_file() {
        Some(path)
    } else {
        eprintln!("skipping: no checkpoint at {}", path.display());
        None
    }
}

fn cpu_engine() -> Option<&'static Engine> {
    static ENGINE: OnceLock<Option<Engine>> = OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let opts = EngineOptions { strict_encoding: true, head_length: 512, threads: Some(4), ..Default::default() };
            checkpoint().map(|p| Engine::load(p, opts).unwrap())
        })
        .as_ref()
}

fn row(state: &str) -> Request {
    Request::new(state, "choose", vec!["a".into(), "b".into()], QType::Choice)
}

#[test]
fn context_defaults_to_native_limit_and_rejects_beyond() {
    let Some(path) = checkpoint() else { return };
    assert_eq!(julia1::config::ModelConfig::load(&path).unwrap().max_positions, 8192);
    let opts = EngineOptions { max_length: Some(8193), ..Default::default() };
    assert!(Engine::load(&path, opts).is_err());
}

#[test]
fn full_budget_and_strict_overflow() {
    let Some(path) = checkpoint() else { return };
    let encoder = Encoder::load(&path.join("tokenizer"), 8192, 256, true, 0, 0).unwrap();
    let overhead = encoder.encode(&row("")).unwrap().ids.len();
    // " x" is one token, so n words fill exactly n state positions.
    let fill = |n: usize| vec!["x"; n].join(" ");
    assert_eq!(encoder.encode(&row(&fill(8192 - overhead))).unwrap().ids.len(), 8192);
    assert!(encoder.encode(&row(&fill(8193 - overhead))).is_err());
}

#[test]
fn strict_encoding_rejects_markers_and_loss() {
    let Some(path) = checkpoint() else { return };
    let encoder = Encoder::load(&path.join("tokenizer"), 1024, 256, true, 16, 16).unwrap();
    let ok = row("plain text");
    let first = encoder.encode(&ok).unwrap();
    assert!(std::sync::Arc::ptr_eq(&first, &encoder.encode(&ok).unwrap()), "encoding cache reuse");
    assert_eq!(first.markers.len(), 2);
    assert!(first.markers.iter().all(|&m| first.ids[m as usize] == encoder.mask_id));
    let mut json_marker = row("");
    json_marker.state = State::Json(json!({"value": "<mask>"}));
    let mut long_option = row("x");
    long_option.options[0] = "word ".repeat(49);
    for bad in [row("<mask>"), json_marker, long_option, row(&"x ".repeat(2000))] {
        assert!(encoder.encode(&bad).is_err());
    }
    // Non-strict mode replaces markers and truncates the state instead.
    let lax = Encoder::load(&path.join("tokenizer"), 64, 32, false, 0, 0).unwrap();
    let e = lax.encode(&row(&format!("<mask> {}", "x ".repeat(200)))).unwrap();
    assert!(e.truncated && e.ids.len() == 64);
    assert_eq!(e.ids.iter().filter(|&&id| id == lax.mask_id).count(), 2);
}

#[test]
fn readme_example_routes_to_billing() {
    let Some(engine) = cpu_engine() else { return };
    let questions = json!({"team": {"type": "choice", "instructions": "Which team should handle this request?",
        "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery",
                     "access": "Account access and login"}}});
    let answers = engine.predict_typed(&State::from("I was charged twice for the same order."), &questions).unwrap();
    assert_eq!(answers[0].1.choice(), Some("billing"));
    let legacy = engine
        .predict(&[Request::new("I was charged twice.", "Which team should handle this request?",
            vec!["Billing".into(), "Shipping".into(), "Account access".into()], QType::Choice)], true)
        .unwrap();
    assert_eq!(legacy[0].index, 0);
    assert!((legacy[0].probabilities.as_ref().unwrap().iter().sum::<f64>() - 1.0).abs() < 1e-6);
}

#[test]
fn batching_does_not_change_results() {
    let Some(engine) = cpu_engine() else { return };
    let rows: Vec<Request> = (0..7)
        .map(|i| Request::new(format!("Order {i}: {}", "late package ".repeat(i * 9)), "What happened?",
            (0..2 + i % 4).map(|k| format!("option {k}")).collect(), QType::Choice))
        .collect();
    let together = engine.logits(&rows).unwrap();
    for (row, expected) in rows.iter().zip(&together) {
        let alone = &engine.logits(std::slice::from_ref(row)).unwrap()[0];
        for (a, b) in alone.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_cpu() {
    let (Some(path), Some(cpu)) = (checkpoint(), cpu_engine()) else { return };
    let opts = EngineOptions { device: Device::Cuda(0), strict_encoding: true, head_length: 512, ..Default::default() };
    let Ok(gpu) = Engine::load(path, opts) else {
        eprintln!("skipping: CUDA device unavailable");
        return;
    };
    let rows: Vec<Request> = (0..12)
        .map(|i| Request::new(format!("Ticket {i}: {}", "the payment failed twice and the card was charged ".repeat(1 + i * 7)),
            "Which team should handle this?", vec!["Billing".into(), "Shipping".into(), "Login".into()], QType::Choice))
        .collect();
    let (a, b) = (cpu.logits(&rows).unwrap(), gpu.logits(&rows).unwrap());
    for (x, y) in a.iter().zip(&b) {
        for (p, q) in x.iter().zip(y) {
            assert!((p - q).abs() < 0.5, "cpu {x:?} vs cuda {y:?}");
        }
    }
    let _ = Device::Cpu;
}
