//! Hierarchical choice routing for more than 20 options (`julia/router/router.py`).
//!
//! Up to `width` options: one unchanged model request. Larger *choice* requests
//! keep survivors per group and rerank until a final group remains. Final
//! probabilities are conditional on `candidates`, never a global distribution.
use crate::engine::Logits;
use crate::pyjson;
use crate::request::{QType, Request, argmax, display_probabilities};
use anyhow::{Result, bail};
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Mutex;

#[derive(Clone, Debug, serde::Serialize)]
pub struct RouteResult {
    pub index: usize,
    pub candidates: Vec<usize>,
    pub probabilities: Vec<f64>,
    pub rounds: usize,
    /// Shared totals for the whole `route_many` call, not per-request attribution.
    pub model_rows: usize,
    pub cache_hits: usize,
    pub hierarchical: bool,
    pub probability_scope: &'static str,
}

pub struct Router<'a, E: Logits + ?Sized> {
    engine: &'a E,
    width: usize,
    survivors: usize,
    batch_size: usize,
    max_options: usize,
    cache: Option<Mutex<LruCache<String, Vec<f32>>>>,
    lock: Mutex<()>,
}

impl<'a, E: Logits + ?Sized> Router<'a, E> {
    pub fn new(engine: &'a E, width: usize, survivors: usize, batch_size: usize, cache_size: usize, max_options: usize) -> Result<Self> {
        if !(2..=20).contains(&width) || !(1..width).contains(&survivors) {
            bail!("Require 2 <= width <= 20 and 1 <= survivors < width");
        }
        if batch_size < 1 || max_options < width {
            bail!("Invalid batch/cache/capacity limit");
        }
        Ok(Self {
            engine,
            width,
            survivors,
            batch_size,
            max_options,
            cache: NonZeroUsize::new(cache_size).map(|c| Mutex::new(LruCache::new(c))),
            lock: Mutex::new(()),
        })
    }

    /// Defaults of the Python Router: width 20, survivors 2, batch 16, no cache, 4096 options.
    pub fn with_defaults(engine: &'a E) -> Self {
        Self::new(engine, 20, 2, 16, 0, 4096).expect("valid defaults")
    }

    pub fn clear_cache(&self) {
        if let Some(c) = &self.cache {
            c.lock().unwrap().clear();
        }
    }

    fn validate(&self, row: &Request) -> Result<()> {
        if !(2..=self.max_options).contains(&row.options.len()) {
            bail!("Expected 2–{} options", self.max_options);
        }
        if row.options.iter().any(String::is_empty) {
            bail!("Options must be nonempty strings");
        }
        if row.qtype == QType::Noul && row.options.len() != 2 {
            bail!("noul requires [false, true]");
        }
        if row.qtype != QType::Choice && row.options.len() > self.width {
            bail!("Hierarchical routing supports choice decisions only");
        }
        Ok(())
    }

    fn key(row: &Request) -> String {
        format!("{}\u{0}{}\u{0}{}\u{0}{}", pyjson::dumps(&row.state.to_value()), row.question, row.qtype.name(), row.options.join("\u{0}"))
    }

    fn score(&self, jobs: &[Request]) -> Result<(Vec<Vec<f32>>, usize, usize)> {
        let mut values: Vec<Option<Vec<f32>>> = vec![None; jobs.len()];
        let mut missing: Vec<(String, Vec<usize>)> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut hits = 0;
        for (i, row) in jobs.iter().enumerate() {
            let key = Self::key(row);
            if let Some(c) = &self.cache
                && let Some(v) = c.lock().unwrap().get(&key)
            {
                values[i] = Some(v.clone());
                hits += 1;
                continue;
            }
            if let Some(&slot) = index.get(&key) {
                missing[slot].1.push(i);
                hits += 1;
            } else {
                index.insert(key.clone(), missing.len());
                missing.push((key, vec![i]));
            }
        }
        for chunk in missing.chunks(self.batch_size) {
            let rows: Vec<Request> = chunk.iter().map(|(_, idx)| jobs[idx[0]].clone()).collect();
            let output = self.engine.logits(&rows)?;
            if output.len() != chunk.len() {
                bail!("Engine returned the wrong number of rows");
            }
            for ((key, indices), scores) in chunk.iter().zip(output) {
                if scores.len() != jobs[indices[0]].options.len() || scores.iter().any(|v| !v.is_finite()) {
                    bail!("Engine logits must be finite and match option count");
                }
                for &i in indices {
                    values[i] = Some(scores.clone());
                }
                if let Some(c) = &self.cache {
                    c.lock().unwrap().put(key.clone(), scores);
                }
            }
        }
        Ok((values.into_iter().map(Option::unwrap).collect(), missing.len(), hits))
    }

    /// Keep one candidate when its local softmax dominates the group.
    pub fn confident_winner(scores: &[f32], best: usize) -> bool {
        let s: Vec<f64> = scores.iter().map(|&x| x as f64).collect();
        let runner_up = s.iter().enumerate().filter(|(i, _)| *i != best).map(|(_, v)| *v).fold(f64::NEG_INFINITY, f64::max);
        if runner_up - s[best] >= (0.045f64 / 0.95).ln() {
            return false;
        }
        let total: f64 = s.iter().map(|v| (v - s[best]).exp()).sum();
        1.0 / total > 0.95 && (runner_up - s[best]).exp() / total < 0.045
    }

    pub fn route(&self, row: &Request) -> Result<RouteResult> {
        Ok(self.route_many(std::slice::from_ref(row))?.remove(0))
    }

    /// Batch independent groups across requests; preserves request/option order.
    pub fn route_many(&self, rows: &[Request]) -> Result<Vec<RouteResult>> {
        let _guard = self.lock.lock().unwrap();
        for row in rows {
            self.validate(row)?;
        }
        let mut candidates: Vec<Vec<usize>> = rows.iter().map(|r| (0..r.options.len()).collect()).collect();
        let mut results: Vec<Option<RouteResult>> = vec![None; rows.len()];
        let mut rounds = vec![0usize; rows.len()];
        let (mut model_rows, mut hits) = (0, 0);
        while results.iter().any(Option::is_none) {
            let mut jobs = Vec::new();
            let mut layout = Vec::new();
            for (i, row) in rows.iter().enumerate() {
                if results[i].is_some() {
                    continue;
                }
                rounds[i] += 1;
                let current = std::mem::take(&mut candidates[i]);
                let last = current.len() <= self.width;
                for group in current.chunks(self.width) {
                    if group.len() == 1 {
                        candidates[i].extend_from_slice(group);
                        continue;
                    }
                    jobs.push(Request { options: group.iter().map(|&k| row.options[k].clone()).collect(), ..row.clone() });
                    layout.push((i, group.to_vec(), last));
                }
            }
            let (scored, used, cached) = self.score(&jobs)?;
            model_rows += used;
            hits += cached;
            for ((i, group, last), scores) in layout.into_iter().zip(scored) {
                let best = argmax(&scores);
                if last {
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let weights: Vec<f64> = scores.iter().map(|&x| (x as f64 - max).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    let hierarchical = rows[i].options.len() > self.width;
                    results[i] = Some(RouteResult {
                        index: group[best],
                        probabilities: display_probabilities(&weights.iter().map(|w| w / total).collect::<Vec<_>>()),
                        candidates: group,
                        rounds: rounds[i],
                        model_rows: 0,
                        cache_hits: 0,
                        hierarchical,
                        probability_scope: if hierarchical { "final_candidates" } else { "all_options" },
                    });
                } else if Self::confident_winner(&scores, best) {
                    candidates[i].push(group[best]);
                } else {
                    let mut remaining: Vec<usize> = (0..group.len()).collect();
                    let mut chosen = Vec::new();
                    for _ in 0..self.survivors.min(group.len()) {
                        let local: Vec<f32> = remaining.iter().map(|&k| scores[k]).collect();
                        chosen.push(group[remaining.remove(argmax(&local))]);
                    }
                    chosen.sort_unstable();
                    candidates[i].extend(chosen);
                }
            }
        }
        Ok(results.into_iter().map(|r| RouteResult { model_rows, cache_hits: hits, ..r.unwrap() }).collect())
    }
}
