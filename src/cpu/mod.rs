//! CPU FP32 forward pass.
//!
//! Sequences are packed without padding, so every GEMM runs on exactly the real
//! tokens. Sliding-window layers only visit their |i-j| <= 64 band, and the last
//! head layer computes queries/FFN for option markers only (keys/values stay full),
//! which is exact for the scores the model returns.
pub mod fastmath;
mod gemm;

use gemm::Epi;

use crate::config::{ModelConfig, rope_table};
use crate::model::{Batch, HostWeights, EMBEDDING};
use anyhow::Result;
use rayon::prelude::*;
use std::sync::Mutex;

/// RoPE (cos, sin) table shared by layers with the same base.
type Rope = std::sync::Arc<(Vec<f32>, Vec<f32>)>;

pub struct CpuModel {
    cfg: ModelConfig,
    w: HostWeights,
    /// RoPE cos/sin per encoder layer (layers sharing a base share a table).
    rope: Vec<Rope>,
    pool: rayon::ThreadPool,
    threads: usize,
    ws: Mutex<Workspace>,
    packed: Vec<PackedLayer>,
    packed_head: Vec<PackedHead>,
    packed_scorer: gemm::PackedW,
}

struct PackedLayer {
    wqkv: gemm::PackedW,
    wo: gemm::PackedW,
    wi: gemm::PackedW,
    wo2: gemm::PackedW,
}

struct PackedHead {
    in_w: gemm::PackedW,
    q_w: gemm::PackedW,
    kv_w: gemm::PackedW,
    out_w: gemm::PackedW,
    lin1: gemm::PackedW,
    lin2: gemm::PackedW,
}

#[derive(Default)]
struct Workspace {
    x: Vec<f32>,
    xn: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    h: Vec<f32>,
    g: Vec<f32>,
    pos: Vec<u32>,
    /// Packed per-(sequence, head) Kᵀ / V panels for attention.
    kv: Vec<f32>,
}

fn grow(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

impl SyncPtr {
    /// Method access keeps closures capturing the Send/Sync wrapper, not the raw field.
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Per-sequence attention geometry: `nq` query rows starting at `q_row` attend to
/// `len` key rows starting at `kv_row`. Query i has position i when `nq == len`.
#[derive(Clone, Copy)]
struct AttnSeq {
    q_row: usize,
    nq: usize,
    kv_row: usize,
    len: usize,
}

const QBLOCK: usize = 64;

/// Opt-in per-op wall-clock totals (`JULIA_PROFILE=1`), printed by [`profile_report`].
pub mod prof {
    use std::sync::LazyLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    pub const NAMES: [&str; 12] =
        ["embed", "ln", "gemm_qkv", "rope", "attn_global", "attn_local", "gemm_wo", "gemm_wi", "geglu", "gemm_wo2", "head", "final"];
    static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var_os("JULIA_PROFILE").is_some());
    static TOTALS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
    #[inline]
    pub fn time<R>(slot: usize, f: impl FnOnce() -> R) -> R {
        if !*ENABLED {
            return f();
        }
        let t = std::time::Instant::now();
        let r = f();
        TOTALS[slot].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    }
    pub fn report() -> String {
        let total: u64 = TOTALS.iter().map(|x| x.load(Ordering::Relaxed)).sum();
        NAMES
            .iter()
            .zip(&TOTALS)
            .map(|(n, v)| {
                let v = v.load(Ordering::Relaxed);
                format!("{n}={:.3}s ({:.1}%)", v as f64 * 1e-9, 100.0 * v as f64 / total.max(1) as f64)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

pub fn profile_report() -> String {
    prof::report()
}

impl CpuModel {
    pub fn new(cfg: ModelConfig, w: HostWeights, threads: usize) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).thread_name(|i| format!("julia-cpu-{i}")).build()?;
        let mut tables: Vec<(f32, Rope)> = Vec::new();
        let rope = cfg
            .rope_theta
            .iter()
            .map(|&theta| {
                if let Some((_, t)) = tables.iter().find(|(x, _)| *x == theta) {
                    return t.clone();
                }
                let t = std::sync::Arc::new(rope_table(theta, cfg.head_dim, cfg.max_positions));
                tables.push((theta, t.clone()));
                t
            })
            .collect();
        let h = cfg.hidden;
        let packed = w
            .layers
            .iter()
            .map(|l| PackedLayer {
                wqkv: gemm::PackedW::new(&l.wqkv, 3 * h, h),
                wo: gemm::PackedW::new(&l.wo, h, h),
                wi: gemm::PackedW::new(&l.wi, 2 * cfg.intermediate, h),
                wo2: gemm::PackedW::new(&l.wo2, h, cfg.intermediate),
            })
            .collect();
        let packed_head = w
            .head
            .iter()
            .map(|l| PackedHead {
                in_w: gemm::PackedW::new(&l.in_w, 3 * h, h),
                q_w: gemm::PackedW::new(&l.in_w[..h * h], h, h),
                kv_w: gemm::PackedW::new(&l.in_w[h * h..], 2 * h, h),
                out_w: gemm::PackedW::new(&l.out_w, h, h),
                lin1: gemm::PackedW::new(&l.lin1_w, cfg.head_ff, h),
                lin2: gemm::PackedW::new(&l.lin2_w, h, cfg.head_ff),
            })
            .collect();
        let packed_scorer = gemm::PackedW::new(&w.scorer.w1, h, h);
        Ok(Self {
            cfg,
            w,
            rope,
            pool,
            threads,
            ws: Mutex::new(Workspace::default()),
            packed,
            packed_head,
            packed_scorer,
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Scores for every marker of every sequence, grouped per sequence.
    pub fn forward(&self, batch: &Batch) -> Result<Vec<Vec<f32>>> {
        let mut ws = self.ws.lock().unwrap();
        let ws: &mut Workspace = &mut ws;
        let scores = self.pool.install(|| self.forward_packed(ws, batch))?;
        Ok(batch.split_scores(&scores))
    }

    fn forward_packed(&self, ws: &mut Workspace, b: &Batch) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let (h, inter) = (cfg.hidden, cfg.intermediate);
        let t = b.tokens();
        let eps = cfg.norm_eps;
        grow(&mut ws.x, t * h);
        grow(&mut ws.xn, t * h);
        grow(&mut ws.qkv, t * 3 * h);
        grow(&mut ws.ctx, t * h);
        grow(&mut ws.h, t * cfg.head_ff);
        grow(&mut ws.g, t * inter);
        ws.pos.clear();
        for &len in &b.lens {
            ws.pos.extend(0..len as u32);
        }
        let Workspace { x, xn, qkv, ctx, h: hbuf, g, pos, kv: kvp } = ws;
        let (x, xn, qkv, ctx) = (&mut x[..t * h], &mut xn[..t * h], &mut qkv[..t * 3 * h], &mut ctx[..t * h]);

        // Embeddings: gather + LayerNorm (no bias).
        let table = self.w.tensors.f32(EMBEDDING, &[cfg.vocab, h])?;
        x.par_chunks_mut(h).zip(b.ids.par_iter()).for_each(|(row, &id)| {
            layer_norm(&table[id as usize * h..(id as usize + 1) * h], &self.w.emb_norm, None, eps, row);
        });

        let seqs: Vec<AttnSeq> = b
            .starts
            .iter()
            .zip(&b.lens)
            .map(|(&s, &len)| AttnSeq { q_row: s, nq: len, kv_row: s, len })
            .collect();

        for (l, layer) in self.w.layers.iter().enumerate() {
            let p = &self.packed[l];
            let normed: &[f32] = match &layer.attn_norm {
                Some(wn) => {
                    prof::time(1, || rows_layer_norm(x, wn, None, eps, xn, h));
                    xn
                }
                None => x,
            };
            prof::time(2, || gemm::linear(normed, t, &p.wqkv, qkv, 3 * h, 0, Epi::Store));
            prof::time(3, || self.rope_inplace(qkv, pos, l));
            let window = if cfg.global[l] { None } else { Some(cfg.half_window) };
            prof::time(if window.is_none() { 4 } else { 5 }, || attention(kvp, qkv, 3 * h, 0, qkv, 3 * h, h, 2 * h, ctx, h, &seqs, cfg.heads, window, 0.125));
            prof::time(6, || gemm::linear(ctx, t, &p.wo, x, h, 0, Epi::Add));
            prof::time(1, || rows_layer_norm(x, &layer.mlp_norm, None, eps, xn, h));
            let gb = &mut g[..t * inter];
            prof::time(7, || gemm::linear_geglu(xn, t, &p.wi, gb));
            prof::time(9, || gemm::linear(gb, t, &p.wo2, x, h, 0, Epi::Add));
        }

        // Final norm + question-type embedding.
        let type_emb = &self.w.type_emb;
        {
            let xs = SyncPtr(x.as_mut_ptr());
            let fnorm = &self.w.final_norm;
            (0..b.starts.len()).into_par_iter().for_each(|s| {
                let te = &type_emb[b.qtypes[s] as usize * h..][..h];
                for r in b.starts[s]..b.starts[s] + b.lens[s] {
                    // SAFETY: each sequence owns disjoint rows.
                    let row = unsafe { std::slice::from_raw_parts_mut(xs.get().add(r * h), h) };
                    let mut tmp = [0f32; 1024];
                    let tmp = &mut tmp[..h];
                    layer_norm(row, fnorm, None, eps, tmp);
                    for ((o, &v), &e) in row.iter_mut().zip(tmp.iter()).zip(te) {
                        *o = v + e;
                    }
                }
            });
        }

        // Decision head: all but the last layer over every token.
        let nh = self.w.head.len();
        let ff = cfg.head_ff;
        for (l, layer) in self.w.head.iter().enumerate().take(nh.saturating_sub(1)) {
            let p = &self.packed_head[l];
            rows_layer_norm(x, &layer.norm1_w, Some(&layer.norm1_b), 1e-5, xn, h);
            gemm::linear(xn, t, &p.in_w, qkv, 3 * h, 0, Epi::Bias(&layer.in_b));
            attention(kvp, qkv, 3 * h, 0, qkv, 3 * h, h, 2 * h, ctx, h, &seqs, cfg.head_heads, None, 0.125);
            gemm::linear(ctx, t, &p.out_w, x, h, 0, Epi::AddBias(&layer.out_b));
            rows_layer_norm(x, &layer.norm2_w, Some(&layer.norm2_b), 1e-5, xn, h);
            let hb = &mut hbuf[..t * ff];
            gemm::linear(xn, t, &p.lin1, hb, ff, 0, Epi::BiasRelu(&layer.lin1_b));
            gemm::linear(hb, t, &p.lin2, x, h, 0, Epi::AddBias(&layer.lin2_b));
        }

        // Gather marker rows; the final head layer (if any) only needs their outputs.
        let m = b.marker_rows.len();
        let mut y = vec![0f32; m * h];
        for (dst, &r) in y.chunks_mut(h).zip(&b.marker_rows) {
            dst.copy_from_slice(&x[r * h..(r + 1) * h]);
        }
        if let Some(layer) = self.w.head.last() {
            let p = &self.packed_head[nh - 1];
            rows_layer_norm(x, &layer.norm1_w, Some(&layer.norm1_b), 1e-5, xn, h);
            // Keys/values for every token: columns h..3h of the in-projection.
            gemm::linear(xn, t, &p.kv_w, qkv, 3 * h, h, Epi::Bias(&layer.in_b[h..]));
            // Queries for markers only.
            let mut xm = vec![0f32; m * h];
            for (dst, &r) in xm.chunks_mut(h).zip(&b.marker_rows) {
                dst.copy_from_slice(&xn[r * h..(r + 1) * h]);
            }
            let mut qm = vec![0f32; m * h];
            gemm::linear(&xm, m, &p.q_w, &mut qm, h, 0, Epi::Bias(&layer.in_b[..h]));
            let mseqs: Vec<AttnSeq> = (0..b.starts.len())
                .map(|s| AttnSeq {
                    q_row: b.marker_starts[s],
                    nq: b.marker_starts[s + 1] - b.marker_starts[s],
                    kv_row: b.starts[s],
                    len: b.lens[s],
                })
                .collect();
            let mut cm = vec![0f32; m * h];
            attention(kvp, &qm, h, 0, qkv, 3 * h, h, 2 * h, &mut cm, h, &mseqs, cfg.head_heads, None, 0.125);
            gemm::linear(&cm, m, &p.out_w, &mut y, h, 0, Epi::AddBias(&layer.out_b));
            let mut yn = vec![0f32; m * h];
            rows_layer_norm(&y, &layer.norm2_w, Some(&layer.norm2_b), 1e-5, &mut yn, h);
            let mut hm = vec![0f32; m * ff];
            gemm::linear(&yn, m, &p.lin1, &mut hm, ff, 0, Epi::BiasRelu(&layer.lin1_b));
            gemm::linear(&hm, m, &p.lin2, &mut y, h, 0, Epi::AddBias(&layer.lin2_b));
        }

        // Scorer.
        let sc = &self.w.scorer;
        let mut yn = vec![0f32; m * h];
        rows_layer_norm(&y, &sc.norm_w, Some(&sc.norm_b), 1e-5, &mut yn, h);
        let mut z = vec![0f32; m * h];
        gemm::linear(&yn, m, &self.packed_scorer, &mut z, h, 0, Epi::Bias(&sc.b1));
        let scores = z
            .chunks(h)
            .map(|row| {
                let mut acc = 0f32;
                for (&v, &w2) in row.iter().zip(&sc.w2) {
                    acc += fastmath::gelu(v) * w2;
                }
                acc + sc.b2
            })
            .collect();
        Ok(scores)
    }

    /// Rotary embedding on the q and k parts of packed qkv rows (rotate_half layout).
    fn rope_inplace(&self, qkv: &mut [f32], pos: &[u32], layer: usize) {
        let cfg = &self.cfg;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let half = hd / 2;
        let (cos, sin) = &*self.rope[layer];
        qkv.par_chunks_mut(3 * h).zip(pos.par_iter()).for_each(|(row, &p)| {
            let c = &cos[p as usize * half..][..half];
            let s = &sin[p as usize * half..][..half];
            for head in row[..2 * h].chunks_mut(hd) {
                let (lo, hi) = head.split_at_mut(half);
                for i in 0..half {
                    let (a, b) = (lo[i], hi[i]);
                    lo[i] = a * c[i] - b * s[i];
                    hi[i] = b * c[i] + a * s[i];
                }
            }
        });
    }
}

/// Scaled dot-product attention over packed sequences, parallel over (sequence, head, query block).
#[allow(clippy::too_many_arguments)]
fn attention(
    kvp: &mut Vec<f32>,
    q: &[f32],
    ldq: usize,
    q_off: usize,
    kv: &[f32],
    ldkv: usize,
    k_off: usize,
    v_off: usize,
    out: &mut [f32],
    ldo: usize,
    seqs: &[AttnSeq],
    heads: usize,
    window: Option<usize>,
    scale: f32,
) {
    #[cfg(target_arch = "x86_64")]
    if gemm::avx512() {
        return attention_packed(kvp, q, ldq, q_off, kv, ldkv, k_off, v_off, out, ldo, seqs, heads, window, scale);
    }
    let _ = kvp;
    let hd = 64;
    let mut tasks = Vec::new();
    for (si, s) in seqs.iter().enumerate() {
        for head in 0..heads {
            for q0 in (0..s.nq).step_by(QBLOCK) {
                tasks.push((si, head, q0));
            }
        }
    }
    let out_ptr = SyncPtr(out.as_mut_ptr());
    tasks.into_par_iter().for_each_init(Vec::<f32>::new, |scratch, (si, head, q0)| {
        let s = seqs[si];
        let q1 = (q0 + QBLOCK).min(s.nq);
        let nq = q1 - q0;
        let (k0, k1) = match window {
            Some(w) => (q0.saturating_sub(w), (q1 + w).min(s.len)),
            None => (0, s.len),
        };
        let nk = k1 - k0;
        if scratch.len() < nq * nk {
            scratch.resize(nq * nk, 0.0);
        }
        let sc = &mut scratch[..nq * nk];
        let qp = q[(s.q_row + q0) * ldq + q_off + head * hd..].as_ptr();
        let kp = kv[(s.kv_row + k0) * ldkv + k_off + head * hd..].as_ptr();
        let vp = kv[(s.kv_row + k0) * ldkv + v_off + head * hd..].as_ptr();
        // S = scale * Q K^T
        gemm::small(nq, nk, hd, sc.as_mut_ptr(), nk, 1, qp, ldq, 1, kp, 1, ldkv, scale);
        for (i, row) in sc.chunks_mut(nk).enumerate() {
            let (lo, hi) = match window {
                Some(w) => {
                    let qi = q0 + i;
                    (qi.saturating_sub(w).max(k0) - k0, (qi + w + 1).min(k1) - k0)
                }
                None => (0, nk),
            };
            softmax_range(row, lo, hi, 1.0);
        }
        // O = P V
        // SAFETY: tasks write disjoint (rows, head) blocks of `out`.
        let op = unsafe { out_ptr.get().add((s.q_row + q0) * ldo + head * hd) };
        gemm::small(nq, hd, nk, op, ldo, 1, sc.as_ptr(), nk, 1, vp, ldkv, 1, 1.0);
    });
}

/// AVX-512 attention: Kᵀ and V of every (sequence, head) are packed once into
/// 32-wide panels, then both products run on the 8×32 register-tiled GEMM kernel.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn attention_packed(
    kvp: &mut Vec<f32>,
    q: &[f32],
    ldq: usize,
    q_off: usize,
    kv: &[f32],
    ldkv: usize,
    k_off: usize,
    v_off: usize,
    out: &mut [f32],
    ldo: usize,
    seqs: &[AttnSeq],
    heads: usize,
    window: Option<usize>,
    scale: f32,
) {
    const HD: usize = 64;
    const P: usize = 32;
    // Per (sequence, head): Kᵀ panels [npk][64][32] then V panels [2][npk*32][32].
    let mut bases = Vec::with_capacity(seqs.len() * heads);
    let mut total = 0;
    for s in seqs {
        let block = s.len.div_ceil(P) * P * HD * 2;
        for _ in 0..heads {
            bases.push(total);
            total += block;
        }
    }
    if kvp.len() < total {
        kvp.resize(total, 0.0);
    }
    let kvp_ptr = SyncPtr(kvp.as_mut_ptr());
    (0..seqs.len() * heads).into_par_iter().for_each(|sh| {
        let (s, head) = (seqs[sh / heads], sh % heads);
        let npk = s.len.div_ceil(P);
        // SAFETY: each (sequence, head) owns its disjoint block of the packed buffer.
        let block = unsafe { std::slice::from_raw_parts_mut(kvp_ptr.get().add(bases[sh]), npk * P * HD * 2) };
        let (kt, vp) = block.split_at_mut(npk * P * HD);
        let vpanel = npk * P * P;
        for j in 0..npk * P {
            let (panel, lane) = (j / P, j % P);
            if j < s.len {
                let row = &kv[(s.kv_row + j) * ldkv..];
                let (kr, vr) = (&row[k_off + head * HD..][..HD], &row[v_off + head * HD..][..HD]);
                for d in 0..HD {
                    kt[(panel * HD + d) * P + lane] = kr[d];
                }
                vp[j * P..(j + 1) * P].copy_from_slice(&vr[..P]);
                vp[vpanel + j * P..vpanel + (j + 1) * P].copy_from_slice(&vr[P..]);
            } else {
                for d in 0..HD {
                    kt[(panel * HD + d) * P + lane] = 0.0;
                }
                vp[j * P..(j + 1) * P].fill(0.0);
                vp[vpanel + j * P..vpanel + (j + 1) * P].fill(0.0);
            }
        }
    });
    let kvp = &*kvp;
    let mut tasks = Vec::new();
    for (si, s) in seqs.iter().enumerate() {
        for head in 0..heads {
            for q0 in (0..s.nq).step_by(QBLOCK) {
                tasks.push((si, head, q0));
            }
        }
    }
    let out_ptr = SyncPtr(out.as_mut_ptr());
    tasks.into_par_iter().for_each_init(
        || (Vec::<f32>::new(), [0f32; 8 * HD], [0f32; 8 * P]),
        |(sc, qpad, tile), (si, head, q0)| {
            let s = seqs[si];
            let npk = s.len.div_ceil(P);
            let q1 = (q0 + QBLOCK).min(s.nq);
            let nq = q1 - q0;
            let nqp = nq.next_multiple_of(8);
            let (k0, k1) = match window {
                Some(w) => (q0.saturating_sub(w), (q1 + w).min(s.len)),
                None => (0, s.len),
            };
            let (k0a, k1a) = (k0 / P * P, k1.div_ceil(P) * P);
            let nk = k1a - k0a;
            if sc.len() < nqp * nk {
                sc.resize(nqp * nk, 0.0);
            }
            let sc = &mut sc[..nqp * nk];
            let base = bases[si * heads + head];
            let kt = &kvp[base..base + npk * P * HD];
            let vp = &kvp[base + npk * P * HD..];
            let vpanel = npk * P * P;
            // S = Q Kᵀ (unscaled), one 8-row strip at a time.
            for strip in (0..nqp).step_by(8) {
                let rows = 8.min(nq - strip);
                let (qptr, lda) = if rows == 8 {
                    (q[(s.q_row + q0 + strip) * ldq + q_off + head * HD..].as_ptr(), ldq)
                } else {
                    qpad.fill(0.0);
                    for i in 0..rows {
                        let src = &q[(s.q_row + q0 + strip + i) * ldq + q_off + head * HD..][..HD];
                        qpad[i * HD..(i + 1) * HD].copy_from_slice(src);
                    }
                    (qpad.as_ptr(), HD)
                };
                for kp in k0a / P..k1a / P {
                    // SAFETY: AVX-512 checked by the caller; operands are in bounds.
                    unsafe { gemm::tile8x32(HD, qptr, lda, kt[kp * HD * P..].as_ptr(), tile.as_mut_ptr()) };
                    let c0 = kp * P - k0a;
                    for i in 0..8 {
                        sc[(strip + i) * nk + c0..][..P].copy_from_slice(&tile[i * P..(i + 1) * P]);
                    }
                }
            }
            for (i, row) in sc.chunks_mut(nk).enumerate() {
                if i >= nq {
                    row.fill(0.0);
                    continue;
                }
                let qi = q0 + i;
                let (lo, hi) = match window {
                    Some(w) => (qi.saturating_sub(w).max(k0a) - k0a, (qi + w + 1).min(s.len) - k0a),
                    None => (0, s.len - k0a),
                };
                softmax_range(row, lo, hi, scale);
            }
            // O = P V for both 32-wide halves of the head.
            for half in 0..2 {
                let vb = vp[half * vpanel + k0a * P..].as_ptr();
                for strip in (0..nqp).step_by(8) {
                    // SAFETY: AVX-512 checked by the caller; operands are in bounds.
                    unsafe { gemm::tile8x32(nk, sc[strip * nk..].as_ptr(), nk, vb, tile.as_mut_ptr()) };
                    for i in 0..8.min(nq - strip) {
                        // SAFETY: tasks write disjoint (rows, head) blocks of `out`.
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(out_ptr.get().add((s.q_row + q0 + strip + i) * ldo + head * HD + half * P), P)
                        };
                        dst.copy_from_slice(&tile[i * P..(i + 1) * P]);
                    }
                }
            }
        },
    );
}

/// Softmax of `scale · row[lo..hi]`; entries outside the range become 0.
fn softmax_range(row: &mut [f32], lo: usize, hi: usize, scale: f32) {
    row[..lo].fill(0.0);
    row[hi..].fill(0.0);
    let r = &mut row[lo..hi];
    let max = max_f32(r);
    for v in r.iter_mut() {
        *v = fastmath::exp((*v - max) * scale);
    }
    let inv = 1.0 / sum_f32(r);
    for v in r.iter_mut() {
        *v *= inv;
    }
}

#[inline]
fn sum_f32(x: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let chunks = x.chunks_exact(16);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..16 {
            acc[i] += c[i];
        }
    }
    acc.iter().sum::<f32>() + rest.iter().sum::<f32>()
}

#[inline]
fn max_f32(x: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 16];
    let chunks = x.chunks_exact(16);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..16 {
            acc[i] = acc[i].max(c[i]);
        }
    }
    acc.iter().chain(rest).copied().fold(f32::NEG_INFINITY, f32::max)
}

#[inline]
fn layer_norm(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = sum_f32(x) / n;
    let mut acc = [0f32; 16];
    let chunks = x.chunks_exact(16);
    let rest = chunks.remainder();
    for c in chunks {
        for i in 0..16 {
            let d = c[i] - mean;
            acc[i] += d * d;
        }
    }
    let var = (acc.iter().sum::<f32>() + rest.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()) / n;
    let rstd = 1.0 / (var + eps).sqrt();
    match b {
        Some(b) => {
            for (((o, &v), &g), &bb) in out.iter_mut().zip(x).zip(w).zip(b) {
                *o = (v - mean) * rstd * g + bb;
            }
        }
        None => {
            for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
                *o = (v - mean) * rstd * g;
            }
        }
    }
}

fn rows_layer_norm(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f32, out: &mut [f32], width: usize) {
    let rows = x.len() / width;
    out[..rows * width].par_chunks_mut(width).zip(x.par_chunks(width)).with_min_len(16).for_each(|(o, r)| {
        layer_norm(r, w, b, eps, o);
    });
}
