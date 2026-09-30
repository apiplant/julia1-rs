//! Matrix products for the CPU backend.
//!
//! `linear` multiplies activations by constant weights. Weights are pre-packed once
//! into 32-column panels (`[k][32]` per panel); an AVX-512 kernel keeps an 8×32
//! tile in registers across the whole K dimension (B streams from L2) and hands the
//! finished tile to a fused epilogue (bias, ReLU, residual add, or GEGLU).
//! CPUs without AVX-512 fall back to the `gemm` crate plus a separate epilogue pass.
use super::fastmath;
use gemm::Parallelism;
use rayon::prelude::*;
use std::sync::LazyLock;

const NR: usize = 32;
const MR: usize = 8;
const MB: usize = 256;
const MAX_K: usize = 4096;

static AVX512: LazyLock<bool> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        std::env::var_os("JULIA_NO_AVX512").is_none() && std::arch::is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
});

/// What to do with each finished `x · Wᵀ` tile.
#[derive(Clone, Copy)]
pub enum Epi<'a> {
    /// out = xWᵀ
    Store,
    /// out += xWᵀ (residual)
    Add,
    /// out = xWᵀ + b
    Bias(&'a [f32]),
    /// out = relu(xWᵀ + b)
    BiasRelu(&'a [f32]),
    /// out += xWᵀ + b (residual with bias)
    AddBias(&'a [f32]),
}

impl Epi<'_> {
    #[inline(always)]
    fn apply(self, tile: &[f32], out: &mut [f32], col: usize) {
        match self {
            Epi::Store => out.copy_from_slice(tile),
            Epi::Add => out.iter_mut().zip(tile).for_each(|(o, t)| *o += t),
            Epi::Bias(b) => out.iter_mut().zip(tile).zip(&b[col..]).for_each(|((o, t), b)| *o = t + b),
            Epi::BiasRelu(b) => out.iter_mut().zip(tile).zip(&b[col..]).for_each(|((o, t), b)| *o = (t + b).max(0.0)),
            Epi::AddBias(b) => out.iter_mut().zip(tile).zip(&b[col..]).for_each(|((o, t), b)| *o += t + b),
        }
    }
}

/// A `[n, k]` row-major weight (PyTorch `Linear.weight`) prepared for `x · Wᵀ`.
pub struct PackedW {
    w: Vec<f32>,
    /// Panel p holds columns p*NR..(p+1)*NR as `[k][NR]` (zero padded past n).
    panels: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

impl PackedW {
    pub fn new(w: &[f32], n: usize, k: usize) -> Self {
        assert_eq!(w.len(), n * k);
        assert!(k <= MAX_K);
        let np = n.div_ceil(NR);
        let mut panels = vec![0f32; np * k * NR];
        for p in 0..np {
            for kk in 0..k {
                for j in 0..NR {
                    let col = p * NR + j;
                    if col < n {
                        panels[(p * k + kk) * NR + j] = w[col * k + kk];
                    }
                }
            }
        }
        Self { w: if *AVX512 { Vec::new() } else { w.to_vec() }, panels, n, k }
    }

    fn panel(&self, p: usize) -> *const f32 {
        self.panels[p * self.k * NR..].as_ptr()
    }
}

#[derive(Clone, Copy)]
struct Ptr(*mut f32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}
impl Ptr {
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// Work split: `blocks` row blocks of `block_rows` rows times `units` column units,
/// handed out as one contiguous chunk of (row block, unit) pairs per worker.
struct Plan {
    units: usize,
    block_rows: usize,
    chunk: usize,
    tasks: usize,
}

impl Plan {
    /// Prefer whole-row tasks (weights stream once); add row blocks only when that
    /// balances the workers better or M is large.
    fn new(rows: usize, units: usize) -> Self {
        let threads = rayon::current_num_threads().max(1);
        let min_blocks = rows.div_ceil(MB);
        let max_blocks = min_blocks.max(rows / 32).min(min_blocks + 8);
        let mut best = (f64::INFINITY, min_blocks);
        for blocks in min_blocks..=max_blocks {
            let total = blocks * units;
            let tasks = threads.min(total);
            let per_thread = total.div_ceil(tasks) as f64;
            // Cost: slowest worker's share of the work (+ a small penalty per extra block).
            let cost = per_thread / blocks as f64 * (1.0 + 0.02 * (blocks - min_blocks) as f64);
            if cost < best.0 - 1e-9 {
                best = (cost, blocks);
            }
        }
        let blocks = best.1;
        let block_rows = rows.div_ceil(blocks).next_multiple_of(MR);
        let blocks = rows.div_ceil(block_rows);
        let total = blocks * units;
        let tasks = threads.min(total);
        let chunk = total.div_ceil(tasks);
        Self { units, block_rows, chunk, tasks: total.div_ceil(chunk) }
    }

    /// (row range, unit) pairs of task `t`.
    fn items(&self, t: usize, rows: usize) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        let total = rows.div_ceil(self.block_rows) * self.units;
        (t * self.chunk..((t + 1) * self.chunk).min(total)).map(move |u| {
            let (b, unit) = (u / self.units, u % self.units);
            (b * self.block_rows, ((b + 1) * self.block_rows).min(rows), unit)
        })
    }
}

fn run_tasks(tasks: usize, flops: usize, task: impl Fn(usize) + Sync + Send) {
    if tasks == 1 || flops < 1 << 17 {
        (0..tasks).for_each(task);
    } else {
        (0..tasks).into_par_iter().with_max_len(1).for_each(task);
    }
}

/// Compute an MR×NR tile of A·B for rows r..r+mr into `tile` (row stride NR).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn tile_product(x: &[f32], k: usize, r: usize, mr: usize, b: *const f32, tile: &mut [f32; MR * NR], pad: &mut [f32]) {
    // SAFETY: rows r..r+mr of x exist; short strips are staged into a zero-padded copy.
    unsafe {
        if mr == MR {
            kernel(k, x.as_ptr().add(r * k), k, b, tile.as_mut_ptr());
        } else {
            pad[..MR * k].fill(0.0);
            pad[..mr * k].copy_from_slice(&x[r * k..(r + mr) * k]);
            kernel(k, pad.as_ptr(), k, b, tile.as_mut_ptr());
        }
    }
}

/// out[r, col..col+n] = epi(x[r, :k] · Wᵀ) for r in 0..rows; x is dense with row stride k.
pub fn linear(x: &[f32], rows: usize, w: &PackedW, out: &mut [f32], ldo: usize, col: usize, epi: Epi) {
    if rows == 0 {
        return;
    }
    assert!(x.len() >= rows * w.k);
    assert!(out.len() >= (rows - 1) * ldo + col + w.n);
    #[cfg(target_arch = "x86_64")]
    if *AVX512 {
        let np = w.n.div_ceil(NR);
        let plan = Plan::new(rows, np);
        let out_ptr = Ptr(out.as_mut_ptr());
        let k = w.k;
        run_tasks(plan.tasks, rows * w.n * k, |t| {
            let mut tile = [0f32; MR * NR];
            let mut pad = vec![0f32; if !rows.is_multiple_of(MR) { MR * k } else { 0 }];
            for (r0, r1, p) in plan.items(t, rows) {
                let ncols = NR.min(w.n - p * NR);
                let mut r = r0;
                while r < r1 {
                    let mr = MR.min(r1 - r);
                    tile_product(x, k, r, mr, w.panel(p), &mut tile, &mut pad);
                    for i in 0..mr {
                        // SAFETY: tasks own disjoint (row block, panel) tiles of `out`.
                        let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.get().add((r + i) * ldo + col + p * NR), ncols) };
                        epi.apply(&tile[i * NR..i * NR + ncols], dst, p * NR);
                    }
                    r += mr;
                }
            }
        });
        return;
    }
    let mut tmp = vec![0f32; rows * w.n];
    fallback_gemm(x, rows, w, &mut tmp);
    out.par_chunks_mut(ldo).take(rows).zip(tmp.par_chunks(w.n)).for_each(|(o, t)| epi.apply(t, &mut o[col..col + w.n], 0));
}

/// g[r, :] = gelu(x·Wᵀ[:, :inter]) * (x·Wᵀ[:, inter:]) for a `[2·inter, k]` weight,
/// without materializing the `[rows, 2·inter]` product.
pub fn linear_geglu(x: &[f32], rows: usize, w: &PackedW, g: &mut [f32]) {
    let inter = w.n / 2;
    assert!(w.n.is_multiple_of(2 * NR) && g.len() >= rows * inter);
    #[cfg(target_arch = "x86_64")]
    if *AVX512 {
        let half = inter / NR; // panels per half
        let plan = Plan::new(rows, half);
        let out_ptr = Ptr(g.as_mut_ptr());
        let k = w.k;
        run_tasks(plan.tasks, rows * w.n * k, |t| {
            let mut a = [0f32; MR * NR];
            let mut b = [0f32; MR * NR];
            let mut pad = vec![0f32; if !rows.is_multiple_of(MR) { MR * k } else { 0 }];
            for (r0, r1, p) in plan.items(t, rows) {
                let mut r = r0;
                while r < r1 {
                    let mr = MR.min(r1 - r);
                    tile_product(x, k, r, mr, w.panel(p), &mut a, &mut pad);
                    tile_product(x, k, r, mr, w.panel(half + p), &mut b, &mut pad);
                    for i in 0..mr {
                        // SAFETY: tasks own disjoint (row block, panel) tiles of `g`.
                        let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.get().add((r + i) * inter + p * NR), NR) };
                        for ((o, &u), &v) in dst.iter_mut().zip(&a[i * NR..(i + 1) * NR]).zip(&b[i * NR..(i + 1) * NR]) {
                            *o = fastmath::gelu(u) * v;
                        }
                    }
                    r += mr;
                }
            }
        });
        return;
    }
    let mut tmp = vec![0f32; rows * w.n];
    fallback_gemm(x, rows, w, &mut tmp);
    g.par_chunks_mut(inter).take(rows).zip(tmp.par_chunks(w.n)).for_each(|(o, t)| {
        for ((o, &u), &v) in o.iter_mut().zip(&t[..inter]).zip(&t[inter..]) {
            *o = fastmath::gelu(u) * v;
        }
    });
}

fn fallback_gemm(x: &[f32], rows: usize, w: &PackedW, out: &mut [f32]) {
    let par = if rows * w.n * w.k < 1 << 18 { Parallelism::None } else { Parallelism::Rayon(rayon::current_num_threads()) };
    // SAFETY: out is [rows, n], x is [rows, k], w is [n, k].
    unsafe {
        gemm::gemm(
            rows,
            w.n,
            w.k,
            out.as_mut_ptr(),
            1,
            w.n as isize,
            false,
            x.as_ptr(),
            1,
            w.k as isize,
            w.w.as_ptr(),
            w.k as isize,
            1,
            0.0,
            1.0,
            false,
            false,
            false,
            par,
        );
    }
}

pub(super) fn avx512() -> bool {
    *AVX512
}

/// 8×32 tile = A[8×k] (row stride lda) · B[k×32] (contiguous); AVX-512 only.
///
/// # Safety
/// `avx512()` must be true; A rows and B must be readable, `tile` holds 256 floats.
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn tile8x32(k: usize, a: *const f32, lda: usize, b: *const f32, tile: *mut f32) {
    unsafe { kernel(k, a, lda, b, tile) }
}

/// tile[MR×NR] = A[MR×k] (row stride lda) · B[k×NR] (packed, contiguous).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn kernel(k: usize, a: *const f32, lda: usize, b: *const f32, tile: *mut f32) {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = [[_mm512_setzero_ps(); 2]; MR];
        for kk in 0..k {
            let b0 = _mm512_loadu_ps(b.add(kk * NR));
            let b1 = _mm512_loadu_ps(b.add(kk * NR + 16));
            for (i, row) in acc.iter_mut().enumerate() {
                let av = _mm512_set1_ps(*a.add(i * lda + kk));
                row[0] = _mm512_fmadd_ps(av, b0, row[0]);
                row[1] = _mm512_fmadd_ps(av, b1, row[1]);
            }
        }
        for (i, row) in acc.iter().enumerate() {
            _mm512_storeu_ps(tile.add(i * NR), row[0]);
            _mm512_storeu_ps(tile.add(i * NR + 16), row[1]);
        }
    }
}

/// dst = beta · lhs · rhs on strided operands, single-threaded (used inside parallel tasks).
#[allow(clippy::too_many_arguments)]
pub fn small(
    m: usize,
    n: usize,
    k: usize,
    dst: *mut f32,
    dst_rs: usize,
    dst_cs: usize,
    lhs: *const f32,
    lhs_rs: usize,
    lhs_cs: usize,
    rhs: *const f32,
    rhs_rs: usize,
    rhs_cs: usize,
    beta: f32,
) {
    // SAFETY: callers pass in-bounds strided views.
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            dst,
            dst_cs as isize,
            dst_rs as isize,
            false,
            lhs,
            lhs_cs as isize,
            lhs_rs as isize,
            rhs,
            rhs_cs as isize,
            rhs_rs as isize,
            0.0,
            beta,
            false,
            false,
            false,
            Parallelism::None,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }

    fn naive(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize) -> Vec<f64> {
        let mut out = vec![0f64; rows * n];
        for r in 0..rows {
            for j in 0..n {
                out[r * n + j] = (0..k).map(|kk| x[r * k + kk] as f64 * w[j * k + kk] as f64).sum();
            }
        }
        out
    }

    #[test]
    fn linear_matches_naive() {
        let mut seed = 1u32;
        for (rows, n, k) in [(1, 32, 7), (13, 96, 384), (70, 40, 300), (129, 1152, 384)] {
            let x: Vec<f32> = (0..rows * k).map(|_| rnd(&mut seed)).collect();
            let w: Vec<f32> = (0..n * k).map(|_| rnd(&mut seed)).collect();
            let bias: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
            let packed = PackedW::new(&w, n, k);
            let ldo = n + 5;
            let base: Vec<f32> = (0..rows * ldo).map(|_| rnd(&mut seed)).collect();
            let prod = naive(&x, &w, rows, n, k);
            for mode in 0..5 {
                let epi = [Epi::Store, Epi::Add, Epi::Bias(&bias), Epi::BiasRelu(&bias), Epi::AddBias(&bias)][mode];
                let mut out = base.clone();
                linear(&x, rows, &packed, &mut out, ldo, 3, epi);
                for r in 0..rows {
                    for j in 0..ldo {
                        let old = base[r * ldo + j] as f64;
                        let expect = if (3..3 + n).contains(&j) {
                            let (v, b) = (prod[r * n + j - 3], bias[j - 3] as f64);
                            [v, old + v, v + b, (v + b).max(0.0), old + v + b][mode]
                        } else {
                            old
                        };
                        assert!((out[r * ldo + j] as f64 - expect).abs() < 1e-4, "{rows}x{n}x{k} mode {mode} at {r},{j}");
                    }
                }
            }
        }
    }

    #[test]
    fn geglu_matches_naive() {
        let mut seed = 7u32;
        for (rows, inter, k) in [(1, 64, 384), (21, 128, 100), (300, 1152, 384)] {
            let x: Vec<f32> = (0..rows * k).map(|_| rnd(&mut seed)).collect();
            let w: Vec<f32> = (0..2 * inter * k).map(|_| rnd(&mut seed)).collect();
            let packed = PackedW::new(&w, 2 * inter, k);
            let prod = naive(&x, &w, rows, 2 * inter, k);
            let mut g = vec![0f32; rows * inter];
            linear_geglu(&x, rows, &packed, &mut g);
            for r in 0..rows {
                for j in 0..inter {
                    let (u, v) = (prod[r * 2 * inter + j], prod[r * 2 * inter + inter + j]);
                    let expect = 0.5 * u * (1.0 + libm::erf(u / std::f64::consts::SQRT_2)) * v;
                    assert!((g[r * inter + j] as f64 - expect).abs() < 1e-4, "{rows}x{inter}x{k} at {r},{j}");
                }
            }
        }
    }
}

#[cfg(test)]
mod perf {
    use super::*;

    /// cargo test --release --lib gemm_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn gemm_speed() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        for (rows, n, k) in [(160, 2304, 384), (1024, 2304, 384), (1024, 384, 1152), (4096, 1152, 384)] {
            let x = vec![0.5f32; rows * k];
            let w = vec![0.25f32; n * k];
            let packed = PackedW::new(&w, n, k);
            let mut out = vec![0f32; rows * n];
            pool.install(|| {
                linear(&x, rows, &packed, &mut out, n, 0, Epi::Store);
                let reps = (2e10 / (2.0 * (rows * n * k) as f64)).ceil() as usize;
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    linear(&x, rows, &packed, &mut out, n, 0, Epi::Store);
                }
                let s = t.elapsed().as_secs_f64();
                println!("{rows}x{n}x{k}: {:.1} GFLOP/s (1 thread)", 2.0 * (rows * n * k * reps) as f64 / s / 1e9);
            });
        }
    }
}
