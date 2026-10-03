//! julia1: predict / check / bench for the Julia-1 Rust runtime.
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use julia1::{Device, Engine, EngineOptions, Request, State, answers_to_json, parse_rows};
use serde_json::{Value, json};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(version, about = "Julia-1 decision model runtime (CPU / CUDA)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct EngineArgs {
    /// Checkpoint directory (model.safetensors, encoder/, tokenizer/). Defaults to Julia-1 in the julia1-rs
    /// cache directory (`$XDG_CACHE_HOME/julia1-rs`, else `~/.cache/julia1-rs`), downloaded there from Hugging
    /// Face first if it is not present.
    #[arg(long, env = "JULIA_CHECKPOINT")]
    checkpoint: Option<PathBuf>,
    /// cpu, cuda or cuda:N
    #[arg(long, default_value = "cpu")]
    device: Device,
    #[arg(long)]
    max_length: Option<usize>,
    #[arg(long, default_value_t = 512)]
    head_length: usize,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    /// CPU threads (default: JULIA_CPU_THREADS or 4).
    #[arg(long)]
    threads: Option<usize>,
    /// Reject marker injection and any truncation (README recommendation).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    strict: bool,
}

impl EngineArgs {
    fn load(&self) -> Result<Engine> {
        let checkpoint = match &self.checkpoint {
            Some(dir) => dir.clone(),
            None => julia1::download::download()?,
        };
        let started = Instant::now();
        let engine = Engine::load(
            &checkpoint,
            EngineOptions {
                device: self.device,
                max_length: self.max_length,
                head_length: self.head_length,
                batch_size: self.batch_size,
                strict_encoding: self.strict,
                threads: self.threads,
                ..Default::default()
            },
        )?;
        eprintln!("loaded {} on {} in {:.2}s", checkpoint.display(), engine.device_name(), started.elapsed().as_secs_f64());
        Ok(engine)
    }
}

#[derive(Subcommand)]
enum Command {
    /// Read JSONL requests (legacy rows or {"state", "questions"}) and write JSONL results.
    Predict {
        #[command(flatten)]
        engine: EngineArgs,
        /// Input JSONL (default: stdin).
        #[arg(long)]
        input: Option<PathBuf>,
        /// Emit raw logits for legacy rows instead of index/probabilities.
        #[arg(long)]
        logits: bool,
    },
    /// Serve the named-question API over HTTP (`POST /v1/classifier`, alias `/v1/systemone`, `GET /health`).
    /// Loads the checkpoint, binds, then runs until Ctrl-C/SIGTERM (an in-flight forward finishes first).
    Serve {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        /// Model identity reported by /health; a request's optional `model` field must match it.
        #[arg(long, default_value = "julia-1")]
        model_name: String,
        /// Max questions per request.
        #[arg(long, env = "JULIA_MAX_REQUEST_BRANCHES", default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..))]
        max_request_branches: u32,
        /// Waiting slots on top of the one in-flight request; beyond that requests get 429.
        #[arg(long, env = "JULIA_MAX_QUEUED", default_value_t = 16, value_parser = clap::value_parser!(u32).range(1..))]
        max_queued: u32,
    },
    /// Compare token ids and logits with the Python reference dumps from bench/py_bench.py.
    Check {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long, default_value = "bench/data")]
        data: PathBuf,
        /// Reference file name (ref-cpu.json or ref-cuda.json).
        #[arg(long, default_value = "ref-cpu.json")]
        reference: String,
    },
    /// Benchmark with the same protocol as bench/py_bench.py.
    Bench {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long, default_value = "bench/data")]
        data: PathBuf,
        #[arg(long, default_value_t = 200)]
        single: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long, default_value_t = 3)]
        long_repeats: usize,
        /// Only use the first N typed rows for the batched pass (default: all 2,000).
        #[arg(long)]
        limit: Option<usize>,
        /// Append the JSON result line to this file.
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Predict { engine, input, logits } => predict(&engine.load()?, input, logits),
        #[cfg(feature = "server")]
        Command::Serve { engine, host, port, model_name, max_request_branches, max_queued } => {
            let config = julia1::server::ServerConfig { max_request_branches: max_request_branches as usize, max_queued: max_queued as usize };
            ntex::rt::System::new("julia1", ntex::rt::DefaultRuntime).block_on(serve(engine, host, port, model_name, config))
        }
        #[cfg(not(feature = "server"))]
        Command::Serve { .. } => bail!("this julia1 binary was built without the `server` feature; rebuild with `--features server`"),
        Command::Check { engine, data, reference } => check(&engine.load()?, &data, &reference),
        Command::Bench { engine, data, single, repeats, long_repeats, limit, output } => {
            bench(&engine, &engine.load()?, &data, single, repeats, long_repeats, limit, output)
        }
    }
}

#[cfg(feature = "server")]
async fn serve(args: EngineArgs, host: String, port: u16, model_name: String, config: julia1::server::ServerConfig) -> Result<()> {
    let engine = std::sync::Arc::new(args.load()?);
    let handle = julia1::server::start_server(&host, port, engine, model_name, config).await?;
    eprintln!("julia1 serving '{}' on http://{}:{} (/v1/classifier, /v1/systemone, /health)", handle.model_name, host, handle.port);
    // ntex handles Ctrl-C/SIGTERM: stop accepting, finish in-flight requests, exit.
    handle.wait().await;
    eprintln!("julia1 stopped");
    Ok(())
}

fn predict(engine: &Engine, input: Option<PathBuf>, logits: bool) -> Result<()> {
    let text = match input {
        Some(p) => std::fs::read_to_string(p)?,
        None => std::io::read_to_string(std::io::stdin())?,
    };
    enum Line {
        Legacy(usize),
        Typed(State, Value),
    }
    let mut legacy = Vec::new();
    let mut lines = Vec::new();
    for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(l).with_context(|| format!("line {}: invalid JSON", i + 1))?;
        if let Some(q) = v.get("questions") {
            let state = v.get("state").and_then(State::from_value).context("state must be text/JSON")?;
            lines.push(Line::Typed(state, q.clone()));
        } else {
            lines.push(Line::Legacy(legacy.len()));
            legacy.push(Request::from_value(&v, i + 1)?);
        }
    }
    let legacy_out: Vec<Value> = if logits {
        engine.logits(&legacy)?.into_iter().map(|z| json!({ "logits": z })).collect()
    } else {
        engine.predict(&legacy, true)?.into_iter().map(|p| serde_json::to_value(p).unwrap()).collect()
    };
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in lines {
        let value = match line {
            Line::Legacy(i) => legacy_out[i].clone(),
            Line::Typed(state, q) => answers_to_json(&engine.predict_typed(&state, &q)?),
        };
        writeln!(out, "{value}")?;
    }
    Ok(())
}

fn read_rows(path: &std::path::Path) -> Result<Vec<Request>> {
    parse_rows(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)
}

fn check(engine: &Engine, data: &std::path::Path, reference: &str) -> Result<()> {
    let refs: Value = serde_json::from_str(&std::fs::read_to_string(data.join(reference))?)?;
    let mut ok = true;
    for name in ["typed.jsonl", "long.jsonl"] {
        let rows = read_rows(&data.join(name))?;
        let r = &refs[name];
        let encoded = engine.encode(&rows)?;
        let ref_tokens: Vec<usize> = serde_json::from_value(r["tokens"].clone())?;
        let ref_ids: Vec<Vec<u32>> = serde_json::from_value(r["ids"].clone())?;
        let token_mismatch = encoded.iter().zip(&ref_tokens).filter(|(e, t)| e.ids.len() != **t).count();
        let id_mismatch = encoded.iter().zip(&ref_ids).filter(|(e, ids)| &e.ids != *ids).count();
        let started = Instant::now();
        let logits = engine.logits(&rows)?;
        let elapsed = started.elapsed().as_secs_f64();
        let ref_logits: Vec<Vec<f32>> = serde_json::from_value(r["logits"].clone())?;
        let (mut agree, mut max_diff, mut sum_diff, mut count) = (0, 0f32, 0f64, 0usize);
        for (a, b) in logits.iter().zip(&ref_logits) {
            if a.len() != b.len() {
                bail!("option count mismatch");
            }
            agree += usize::from(julia1::request::argmax(a) == julia1::request::argmax(b));
            for (x, y) in a.iter().zip(b) {
                max_diff = max_diff.max((x - y).abs());
                sum_diff += (x - y).abs() as f64;
                count += 1;
            }
        }
        println!(
            "{name}: rows={} token-length mismatches={token_mismatch} id mismatches={id_mismatch}/{} argmax agree={agree}/{} max|Δlogit|={max_diff:.5} mean|Δlogit|={:.6} ({elapsed:.2}s)",
            rows.len(),
            ref_ids.len(),
            rows.len(),
            sum_diff / count as f64
        );
        ok &= token_mismatch == 0 && id_mismatch == 0;
    }
    if !ok {
        bail!("tokenization differs from the Python reference");
    }
    Ok(())
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q) as usize]
}

#[allow(clippy::too_many_arguments)]
fn bench(
    args: &EngineArgs,
    engine: &Engine,
    data: &std::path::Path,
    single: usize,
    repeats: usize,
    long_repeats: usize,
    limit: Option<usize>,
    output: Option<PathBuf>,
) -> Result<()> {
    let mut rows = read_rows(&data.join("typed.jsonl"))?;
    let long_rows = read_rows(&data.join("long.jsonl"))?;
    for row in &rows[..20] {
        engine.predict(std::slice::from_ref(row), true)?;
    }
    engine.predict(&rows[..64], true)?;
    for row in long_rows.iter().filter(|_| long_repeats > 0) {
        engine.predict(std::slice::from_ref(row), true)?;
    }

    engine.clear_cache();
    let mut lat = Vec::with_capacity(single);
    for row in &rows[..single] {
        let t = Instant::now();
        engine.predict(std::slice::from_ref(row), true)?;
        lat.push(t.elapsed().as_secs_f64());
    }
    let total: f64 = lat.iter().sum();
    let mut sorted = lat.clone();
    sorted.sort_by(f64::total_cmp);
    let single_json = json!({
        "n": single, "mean_ms": 1e3 * total / single as f64, "p50_ms": 1e3 * percentile(&sorted, 0.5),
        "p99_ms": 1e3 * percentile(&sorted, 0.99), "rows_per_s": single as f64 / total,
    });

    rows.truncate(limit.unwrap_or(rows.len()));
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        engine.clear_cache();
        let t = Instant::now();
        engine.predict(&rows, true)?;
        best = best.min(t.elapsed().as_secs_f64());
    }

    let mut long = Vec::new();
    for row in long_rows.iter().filter(|_| long_repeats > 0) {
        let tokens = engine.encode(std::slice::from_ref(row))?[0].ids.len();
        let mut b = f64::INFINITY;
        for _ in 0..long_repeats {
            engine.clear_cache();
            let t = Instant::now();
            engine.predict(std::slice::from_ref(row), true)?;
            b = b.min(t.elapsed().as_secs_f64());
        }
        long.push(json!({ "tokens": tokens, "best_ms": 1e3 * b }));
    }
    let device = match args.device {
        Device::Cpu => "cpu".to_owned(),
        Device::Cuda(i) => if i == 0 { "cuda".to_owned() } else { format!("cuda:{i}") },
    };
    let result = json!({
        "impl": "rust", "device": device, "threads": args.threads.unwrap_or(4), "batch_size": args.batch_size,
        "single": single_json,
        "batch": { "n": rows.len(), "best_s": best, "rows_per_s": rows.len() as f64 / best },
        "long": long,
    });
    println!("{result}");
    if std::env::var_os("JULIA_PROFILE").is_some() {
        eprintln!("profile: {}", julia1::cpu::profile_report());
    }
    if let Some(path) = output {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(f, "{result}")?;
    }
    Ok(())
}
