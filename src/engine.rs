//! Resident inference engine: `FastEngine` semantics on a CPU or CUDA backend.
use crate::config::ModelConfig;
use crate::encode::{Encoded, Encoder};
use crate::model::{Batch, HostWeights};
use crate::request::{Request, argmax, display_probabilities, softmax_f32};
use crate::weights::SafeTensors;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda(0)),
            _ => match s.strip_prefix("cuda:").map(str::parse) {
                Some(Ok(i)) => Ok(Self::Cuda(i)),
                _ => bail!("device must be cpu, cuda or cuda:N"),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub device: Device,
    /// Combined token limit; defaults to the checkpoint's `max_position_embeddings`.
    pub max_length: Option<usize>,
    pub head_length: usize,
    /// Rows per forward pass (sorted by length first, as in Python).
    pub batch_size: usize,
    /// Upper bound on packed tokens per forward pass.
    pub max_batch_tokens: usize,
    pub strict_encoding: bool,
    /// CPU worker threads; defaults to `JULIA_CPU_THREADS` or 4 like the Python runtime.
    pub threads: Option<usize>,
    pub encoding_cache: usize,
    pub token_cache: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            max_length: None,
            head_length: 256,
            batch_size: 16,
            max_batch_tokens: 65_536,
            strict_encoding: false,
            threads: None,
            encoding_cache: 2048,
            token_cache: 8192,
        }
    }
}

enum Backend {
    Cpu(Box<crate::cpu::CpuModel>),
    #[cfg(feature = "cuda")]
    Cuda(Box<crate::cuda::CudaModel>),
}

pub struct Engine {
    pub encoder: Encoder,
    pub config: ModelConfig,
    backend: Backend,
    options: EngineOptions,
    lock: Mutex<()>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct Prediction {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<Vec<f64>>,
}

impl Engine {
    pub fn load(checkpoint: impl AsRef<Path>, options: EngineOptions) -> Result<Self> {
        let root = checkpoint.as_ref();
        let config = ModelConfig::load(root)?;
        let max_length = options.max_length.unwrap_or(config.max_positions);
        if !(1..=config.max_positions).contains(&max_length) {
            bail!("max_length must be an integer between 1 and {}", config.max_positions);
        }
        if options.batch_size < 1 || options.max_batch_tokens < max_length {
            bail!("Invalid batch size / token budget");
        }
        if root.join("INCOMPLETE").exists() || root.join("quantization.json").exists() {
            bail!("INT8 checkpoints are not supported by this runtime");
        }
        let encoder = Encoder::load(
            &root.join("tokenizer"),
            max_length,
            options.head_length,
            options.strict_encoding,
            options.token_cache,
            options.encoding_cache,
        )?;
        let tensors = SafeTensors::open(&root.join("model.safetensors"))?;
        let weights = HostWeights::load(tensors, &config)?;
        let backend = match options.device {
            Device::Cpu => {
                let threads = options.threads.unwrap_or_else(|| {
                    std::env::var("JULIA_CPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
                });
                Backend::Cpu(Box::new(crate::cpu::CpuModel::new(config.clone(), weights, threads.max(1))?))
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(ordinal) => Backend::Cuda(Box::new(crate::cuda::CudaModel::new(config.clone(), &weights, ordinal)?)),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => bail!("This build has no CUDA support; rebuild with --features cuda"),
        };
        let options = EngineOptions { max_length: Some(max_length), ..options };
        Ok(Self { encoder, config, backend, options, lock: Mutex::new(()) })
    }

    pub fn options(&self) -> &EngineOptions {
        &self.options
    }

    pub fn clear_cache(&self) {
        self.encoder.clear_cache();
    }

    /// Validate and encode every row (in parallel), before any inference.
    pub fn encode(&self, rows: &[Request]) -> Result<Vec<Arc<Encoded>>> {
        for (i, row) in rows.iter().enumerate() {
            row.validate(i + 1)?;
        }
        rows.par_iter().map(|row| self.encoder.encode(row)).collect()
    }

    /// Length-sorted micro-batches (bounded rows and tokens), as index lists.
    fn batches(&self, encoded: &[Arc<Encoded>]) -> Vec<Vec<usize>> {
        let mut order: Vec<usize> = (0..encoded.len()).collect();
        order.sort_by_key(|&i| encoded[i].ids.len());
        let mut out = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        let mut tokens = 0;
        for i in order {
            let n = encoded[i].ids.len();
            if !group.is_empty() && (group.len() == self.options.batch_size || tokens + n > self.options.max_batch_tokens) {
                out.push(std::mem::take(&mut group));
                tokens = 0;
            }
            group.push(i);
            tokens += n;
        }
        if !group.is_empty() {
            out.push(group);
        }
        out
    }

    fn forward(&self, items: &[&Encoded]) -> Result<Vec<Vec<f32>>> {
        let batch = Batch::new(items);
        let scores = match &self.backend {
            Backend::Cpu(m) => m.forward(&batch)?,
            #[cfg(feature = "cuda")]
            Backend::Cuda(m) => m.forward(&batch)?,
        };
        if scores.iter().flatten().any(|v| !v.is_finite()) {
            bail!("Inference returned nonfinite logits");
        }
        Ok(scores)
    }

    pub fn logits_encoded(&self, encoded: &[Arc<Encoded>]) -> Result<Vec<Vec<f32>>> {
        let _guard = self.lock.lock().unwrap();
        let mut result = vec![Vec::new(); encoded.len()];
        for indices in self.batches(encoded) {
            let items: Vec<&Encoded> = indices.iter().map(|&i| &*encoded[i]).collect();
            for (i, scores) in indices.into_iter().zip(self.forward(&items)?) {
                result[i] = scores;
            }
        }
        Ok(result)
    }

    /// Raw option logits per request, in option order.
    pub fn logits(&self, rows: &[Request]) -> Result<Vec<Vec<f32>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let encoded = self.encode(rows)?;
        self.logits_encoded(&encoded)
    }

    /// Legacy list API: argmax index plus display-formatted probabilities.
    pub fn predict(&self, rows: &[Request], probabilities: bool) -> Result<Vec<Prediction>> {
        Ok(self
            .logits(rows)?
            .into_iter()
            .map(|z| Prediction {
                index: argmax(&z),
                probabilities: probabilities.then(|| {
                    display_probabilities(&softmax_f32(&z).into_iter().map(f64::from).collect::<Vec<_>>())
                }),
            })
            .collect())
    }

    /// `encoding_info`: audit of the strict encoding used for inference.
    pub fn encoding_info(&self, rows: &[Request]) -> Result<Vec<serde_json::Value>> {
        if !self.options.strict_encoding {
            bail!("Lossless encoding audit requires strict_encoding=True");
        }
        Ok(self
            .encode(rows)?
            .iter()
            .map(|e| {
                serde_json::json!({
                    "tokens": e.ids.len(), "optionTokens": e.option_tokens, "headLength": self.options.head_length,
                    "stateTruncated": false, "optionsTruncated": false,
                })
            })
            .collect())
    }

    pub fn device_name(&self) -> String {
        match &self.backend {
            Backend::Cpu(m) => format!("cpu ({} threads)", m.threads()),
            #[cfg(feature = "cuda")]
            Backend::Cuda(m) => m.name(),
        }
    }
}

/// Anything that scores legacy rows (the engine, or a test double for the router).
pub trait Logits {
    fn logits(&self, rows: &[Request]) -> Result<Vec<Vec<f32>>>;
}

impl Logits for Engine {
    fn logits(&self, rows: &[Request]) -> Result<Vec<Vec<f32>>> {
        Engine::logits(self, rows)
    }
}

/// Parse a JSONL document of legacy rows.
pub fn parse_rows(text: &str) -> Result<Vec<Request>> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            let v: serde_json::Value = serde_json::from_str(l).with_context(|| format!("line {}: invalid JSON", i + 1))?;
            Request::from_value(&v, i + 1)
        })
        .collect()
}
