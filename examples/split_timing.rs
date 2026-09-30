//! Time tokenization vs model forward for the typed rows (diagnostic).
use julia1::{Device, Engine, EngineOptions, parse_rows};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let device: Device = std::env::args().nth(1).unwrap_or("cuda".into()).parse()?;
    let batch_size: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(16);
    let threads = std::env::var("THREADS").ok().map(|v| v.parse().unwrap());
    let engine = Engine::load("../../ai/Julia-1", EngineOptions { device, strict_encoding: true, head_length: 512, batch_size, threads, ..Default::default() })?;
    let rows = parse_rows(&std::fs::read_to_string("bench/data/typed.jsonl")?)?;
    engine.logits(&rows)?;
    for _ in 0..std::env::var("REPS").map(|v| v.parse().unwrap()).unwrap_or(3) {
        engine.clear_cache();
        let t = Instant::now();
        let enc = engine.encode(&rows)?;
        let te = t.elapsed().as_secs_f64();
        let t = Instant::now();
        engine.logits_encoded(&enc)?;
        let tf = t.elapsed().as_secs_f64();
        println!("batch {batch_size}: encode {:.1} ms, forward {:.1} ms", te * 1e3, tf * 1e3);
    }
    Ok(())
}
