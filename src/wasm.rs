//! wasm-bindgen bindings for running Julia-1 in the browser.
//!
//! The checkpoint files (`model.safetensors`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`,
//! `julia_config.json`, `encoder/config.json`) are fetched by JS (typically from the Hugging Face Hub) and
//! handed to [`WasmJulia::load`] as bytes; there is no filesystem here. Requests and answers are JSON strings
//! so the JS side never needs a Rust struct layout. Single-threaded, FP32, CPU only.

use crate::engine::{Device, Engine, EngineOptions};
use crate::request::State;
use crate::typed::answers_to_json;
use serde_json::Value;
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Like [`js_err`] but keeps an anyhow error's whole context chain.
fn anyhow_err(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

/// Readable panic messages in the browser console.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct WasmJulia {
    engine: Engine,
}

#[wasm_bindgen]
impl WasmJulia {
    /// Loads a checkpoint from its file contents. `head_length` is the token budget for the question head
    /// (the Python runtime's default for the typed API is 512).
    #[wasm_bindgen]
    pub fn load(
        weights: Vec<u8>,
        tokenizer_json: &[u8],
        tokenizer_config: &str,
        julia_config: &str,
        encoder_config: &str,
        head_length: usize,
    ) -> Result<WasmJulia, JsValue> {
        let julia: Value = serde_json::from_str(julia_config).map_err(js_err)?;
        let encoder: Value = serde_json::from_str(encoder_config).map_err(js_err)?;
        let options = EngineOptions {
            device: Device::Cpu,
            strict_encoding: true,
            head_length,
            // Nothing to size a cache against in a tab: keep the defaults small.
            encoding_cache: 256,
            token_cache: 1024,
            threads: Some(1),
            ..Default::default()
        };
        let engine =
            Engine::from_parts(&julia, &encoder, tokenizer_json, tokenizer_config, weights, options).map_err(anyhow_err)?;
        Ok(WasmJulia { engine })
    }

    /// Answers named typed questions about a state: `state_json` is a JSON string or object, `questions_json`
    /// maps IDs to `{type, instructions, criteria}` (`choice` / `score` / `noul`). Returns the same JSON the
    /// `julia1 predict` command prints for one row.
    pub fn predict(&self, state_json: &str, questions_json: &str) -> Result<String, JsValue> {
        let state: Value = serde_json::from_str(state_json).map_err(js_err)?;
        let state = State::from_value(&state).ok_or_else(|| js_err("state must be a string, object or array"))?;
        let questions: Value = serde_json::from_str(questions_json).map_err(js_err)?;
        let answers = self.engine.predict_typed(&state, &questions).map_err(anyhow_err)?;
        serde_json::to_string_pretty(&answers_to_json(&answers)).map_err(js_err)
    }
}
