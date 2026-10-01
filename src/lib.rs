//! Rust inference runtime for the Supersonic Labs **Julia-1** decision model.
//!
//! Port of the Python `julia` package (`FastEngine`, the named-question API and
//! the hierarchical `Router`) with an FP32 CPU backend and a BF16 CUDA backend
//! (feature `cuda`). Tokenization, marker serialization and option order follow
//! the Python runtime exactly; see `bench/` for the parity and speed checks.
//!
//! ```no_run
//! use julia1::{Engine, EngineOptions, State};
//! let engine = Engine::load("Julia-1", EngineOptions { strict_encoding: true, head_length: 512, ..Default::default() })?;
//! let questions = serde_json::json!({"team": {"type": "choice",
//!     "instructions": "Which team should handle this request?",
//!     "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery"}}});
//! let answers = engine.predict_typed(&State::from("I was charged twice."), &questions)?;
//! println!("{}", answers[0].1.choice().unwrap());
//! # anyhow::Ok(())
//! ```
pub mod config;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod encode;
pub mod engine;
pub mod model;
pub mod pyjson;
pub mod request;
pub mod router;
pub mod typed;
pub mod weights;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use engine::{Device, Engine, EngineOptions, Logits, Prediction, parse_rows};
pub use request::{QType, Request, State, display_probabilities};
pub use router::{RouteResult, Router};
pub use typed::{Answer, Answers, answers_to_json, predict_typed};
