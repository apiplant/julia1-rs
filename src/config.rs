//! Checkpoint configuration (`julia_config.json` + `encoder/config.json`).
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub layers: usize,
    /// Per encoder layer: full (global) attention, otherwise a |i-j| <= `half_window` band.
    pub global: Vec<bool>,
    pub half_window: usize,
    /// Per encoder layer RoPE base.
    pub rope_theta: Vec<f32>,
    pub norm_eps: f32,
    pub vocab: usize,
    pub max_positions: usize,
    pub head_layers: usize,
    pub head_heads: usize,
    pub head_ff: usize,
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

fn usize_of(v: &Value, key: &str) -> Result<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize).with_context(|| format!("config is missing {key}"))
}

impl ModelConfig {
    pub fn load(root: &Path) -> Result<Self> {
        let julia = read_json(&root.join("julia_config.json"))?;
        let enc = read_json(&root.join("encoder/config.json"))?;
        Self::from_values(&julia, &enc)
    }

    /// Builds the configuration from the parsed `julia_config.json` and `encoder/config.json`.
    pub fn from_values(julia: &Value, enc: &Value) -> Result<Self> {
        let (julia, enc) = (julia.clone(), enc.clone());
        ensure!(julia["format_version"] == 1, "Unsupported Julia checkpoint format");
        ensure!(
            julia.get("weight_dtype").and_then(Value::as_str).unwrap_or("float32") == "float32",
            "Only float32 Julia checkpoints are supported"
        );
        ensure!(enc["model_type"] == "modernbert", "Encoder must be ModernBERT");
        for key in ["attention_bias", "mlp_bias", "norm_bias"] {
            ensure!(enc.get(key).and_then(Value::as_bool) == Some(false), "Unsupported encoder setting {key}");
        }
        ensure!(enc["hidden_activation"] == "gelu", "Unsupported encoder activation");
        let hidden = usize_of(&enc, "hidden_size")?;
        let heads = usize_of(&enc, "num_attention_heads")?;
        let layers = usize_of(&enc, "num_hidden_layers")?;
        let every = usize_of(&enc, "global_attn_every_n_layers")?;
        let local = usize_of(&enc, "local_attention")?;
        ensure!(hidden % heads == 0 && hidden / heads == 64, "Runtime kernels require head_dim 64");
        let types: Vec<String> = match enc.get("layer_types").and_then(Value::as_array) {
            Some(t) => t.iter().map(|x| x.as_str().unwrap_or_default().to_owned()).collect(),
            None => (0..layers)
                .map(|i| if i % every == 0 { "full_attention" } else { "sliding_attention" }.to_owned())
                .collect(),
        };
        ensure!(types.len() == layers, "layer_types must list every layer");
        let theta = |kind: &str| -> Result<f32> {
            if let Some(p) = enc.get("rope_parameters").and_then(|p| p.get(kind)) {
                ensure!(p["rope_type"] == "default", "Only default RoPE is supported");
                return p["rope_theta"].as_f64().map(|x| x as f32).context("rope_theta");
            }
            let key = if kind == "full_attention" { "global_rope_theta" } else { "local_rope_theta" };
            enc.get(key).and_then(Value::as_f64).map(|x| x as f32).with_context(|| format!("missing {key}"))
        };
        let rope_theta = types.iter().map(|t| theta(t)).collect::<Result<Vec<_>>>()?;
        let head_layers = usize_of(&julia, "head_layers")?;
        Ok(Self {
            hidden,
            heads,
            head_dim: hidden / heads,
            intermediate: usize_of(&enc, "intermediate_size")?,
            layers,
            // ModernBertAttention: sliding iff layer_id % global_attn_every_n_layers != 0.
            global: (0..layers).map(|i| i % every == 0).collect(),
            half_window: local / 2,
            rope_theta,
            norm_eps: enc.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32,
            vocab: usize_of(&enc, "vocab_size")?,
            max_positions: usize_of(&enc, "max_position_embeddings")?,
            head_layers,
            head_heads: (hidden / 64).max(1),
            head_ff: 4 * hidden,
        })
    }
}

/// RoPE tables as torch computes them: f32 inv_freq, f32 position products, cos/sin.
pub fn rope_table(theta: f32, head_dim: usize, positions: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf((2 * i) as f32 / head_dim as f32)).collect();
    let mut cos = vec![0f32; positions * half];
    let mut sin = vec![0f32; positions * half];
    for p in 0..positions {
        for i in 0..half {
            let f = inv[i] * p as f32;
            cos[p * half + i] = f.cos();
            sin[p * half + i] = f.sin();
        }
    }
    (cos, sin)
}
