//! CUDA backend: BF16 tensor-core GEMMs (cuBLAS, FP32 accumulate), FP32 residual
//! stream and norms, fused flash attention with RoPE / sliding window, packed
//! sequences without padding. Precision matches the Python runtime's CUDA path
//! (autocast BF16), with FP32 logits.
use crate::config::{ModelConfig, rope_table};
use crate::model::{Batch, EMBEDDING, HostWeights};
use anyhow::{Context, Result, bail, ensure};
use cudarc::cublas::{CudaBlas, result as blas, sys as bsys};
use cudarc::driver::{CudaContext, CudaStream, result as cu, sys};
use std::ffi::{CString, c_void};
use std::sync::{Arc, Mutex};

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.fatbin"));

const BR: usize = 64;

/// Owned device allocation.
struct DevBuf {
    ptr: sys::CUdeviceptr,
}

impl DevBuf {
    fn new(bytes: usize) -> Result<Self> {
        // SAFETY: plain allocation; freed in Drop.
        let ptr = unsafe { cu::malloc_sync(bytes.max(256))? };
        Ok(Self { ptr })
    }

    fn upload<T>(data: &[T]) -> Result<Self> {
        let buf = Self::new(std::mem::size_of_val(data))?;
        // SAFETY: buffer was sized for `data`.
        unsafe { cu::memcpy_htod_sync(buf.ptr, data)? };
        Ok(buf)
    }

    fn bf16(data: &[f32]) -> Result<Self> {
        let converted: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
        Self::upload(&converted)
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: allocated by malloc_sync and freed once.
        unsafe {
            let _ = cu::free_sync(self.ptr);
        }
    }
}

/// Page-locked host buffer for async transfers.
struct Pinned {
    ptr: *mut u8,
    bytes: usize,
}

impl Pinned {
    fn new(bytes: usize) -> Result<Self> {
        // SAFETY: plain allocation; freed in Drop.
        let ptr = unsafe { cu::malloc_host(bytes.max(256), 0)? } as *mut u8;
        Ok(Self { ptr, bytes })
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: allocated by malloc_host and freed once.
        unsafe {
            let _ = cu::free_host(self.ptr as *mut c_void);
        }
    }
}

struct Kernels {
    embed_ln: sys::CUfunction,
    ln_bf16: sys::CUfunction,
    final_norm: sys::CUfunction,
    geglu: sys::CUfunction,
    bias_act: sys::CUfunction,
    gather_rows: sys::CUfunction,
    scorer: sys::CUfunction,
    attention_rope: sys::CUfunction,
    attention_plain: sys::CUfunction,
}

struct EncLayer {
    attn_norm: Option<DevBuf>,
    wqkv: DevBuf,
    wo: DevBuf,
    mlp_norm: DevBuf,
    wi: DevBuf,
    wo2: DevBuf,
    rope: usize,
    global: bool,
}

struct HeadLayer {
    norm1_w: DevBuf,
    norm1_b: DevBuf,
    in_w: DevBuf,
    in_b: DevBuf,
    out_w: DevBuf,
    out_b: DevBuf,
    norm2_w: DevBuf,
    norm2_b: DevBuf,
    lin1_w: DevBuf,
    lin1_b: DevBuf,
    lin2_w: DevBuf,
    lin2_b: DevBuf,
}

#[derive(Default)]
struct Workspace {
    tokens: usize,
    markers: usize,
    meta_len: usize,
    bufs: Vec<DevBuf>,
    meta: Option<DevBuf>,
    host: Option<Pinned>,
    scores_host: Option<Pinned>,
}

/// Device pointers into the current workspace.
#[derive(Clone, Copy)]
struct Ws {
    x: u64,
    xn: u64,
    qkv: u64,
    ctx: u64,
    h: u64,
    g: u64,
    f32s: u64,
    ym: u64,
    ymb: u64,
    cm: u64,
    delta: u64,
    dm: u64,
    scores: u64,
}

pub struct CudaModel {
    cfg: ModelConfig,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Kernels,
    _module: sys::CUmodule,
    emb: DevBuf,
    emb_norm: DevBuf,
    layers: Vec<EncLayer>,
    final_norm: DevBuf,
    type_emb: DevBuf,
    head: Vec<HeadLayer>,
    s_norm_w: DevBuf,
    s_norm_b: DevBuf,
    s_w1: DevBuf,
    s_b1: DevBuf,
    s_w2: DevBuf,
    s_b2: f32,
    ropes: Vec<(DevBuf, DevBuf)>,
    ws: Mutex<Workspace>,
    name: String,
}

// SAFETY: raw CUDA handles are only used under the workspace mutex with the context bound.
unsafe impl Send for CudaModel {}
unsafe impl Sync for CudaModel {}

fn p<T>(v: &T) -> *mut c_void {
    v as *const T as *mut c_void
}

impl CudaModel {
    pub fn new(cfg: ModelConfig, w: &HostWeights, ordinal: usize) -> Result<Self> {
        ensure!(cfg.hidden == 384 && cfg.head_dim == 64, "CUDA kernels are specialized for hidden 384 / head_dim 64");
        ensure!(cfg.head_heads == cfg.heads && cfg.head_layers >= 1, "Unsupported decision head layout");
        let ctx = CudaContext::new(ordinal).context("CUDA device unavailable")?;
        let (major, _) = ctx.compute_capability()?;
        if major < 8 {
            bail!("CUDA backend requires BF16 tensor cores (compute capability 8.0+)");
        }
        // SAFETY: we manage all synchronization explicitly on one stream.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        let blas = CudaBlas::new(stream.clone())?;
        // SAFETY: FATBIN is a valid image produced by nvcc in build.rs.
        let module = unsafe { cu::module::load_data(FATBIN.as_ptr() as *const c_void)? };
        let f = |name: &str| -> Result<sys::CUfunction> {
            // SAFETY: module is loaded; name is a kernel in kernels.cu.
            Ok(unsafe { cu::module::get_function(module, CString::new(name)?)? })
        };
        let k = Kernels {
            embed_ln: f("embed_ln")?,
            ln_bf16: f("ln_bf16")?,
            final_norm: f("final_norm")?,
            geglu: f("geglu")?,
            bias_act: f("bias_act")?,
            gather_rows: f("gather_rows")?,
            scorer: f("scorer")?,
            attention_rope: f("attention_rope")?,
            attention_plain: f("attention_plain")?,
        };
        let (h, i) = (cfg.hidden, cfg.intermediate);
        let emb = DevBuf::upload(&w.tensors.f32(EMBEDDING, &[cfg.vocab, h])?)?;
        let mut thetas: Vec<f32> = Vec::new();
        let mut ropes = Vec::new();
        let mut layers = Vec::new();
        for (l, layer) in w.layers.iter().enumerate() {
            let theta = cfg.rope_theta[l];
            let rope = match thetas.iter().position(|&t| t == theta) {
                Some(r) => r,
                None => {
                    let (c, s) = rope_table(theta, cfg.head_dim, cfg.max_positions);
                    ropes.push((DevBuf::upload(&c)?, DevBuf::upload(&s)?));
                    thetas.push(theta);
                    thetas.len() - 1
                }
            };
            layers.push(EncLayer {
                attn_norm: layer.attn_norm.as_deref().map(DevBuf::upload).transpose()?,
                wqkv: DevBuf::bf16(&layer.wqkv)?,
                wo: DevBuf::bf16(&layer.wo)?,
                mlp_norm: DevBuf::upload(&layer.mlp_norm)?,
                wi: DevBuf::bf16(&layer.wi)?,
                wo2: DevBuf::bf16(&layer.wo2)?,
                rope,
                global: cfg.global[l],
            });
        }
        debug_assert_eq!(w.layers[0].wi.len(), 2 * i * h);
        let head = w
            .head
            .iter()
            .map(|l| {
                Ok(HeadLayer {
                    norm1_w: DevBuf::upload(&l.norm1_w)?,
                    norm1_b: DevBuf::upload(&l.norm1_b)?,
                    in_w: DevBuf::bf16(&l.in_w)?,
                    in_b: DevBuf::upload(&l.in_b)?,
                    out_w: DevBuf::bf16(&l.out_w)?,
                    out_b: DevBuf::upload(&l.out_b)?,
                    norm2_w: DevBuf::upload(&l.norm2_w)?,
                    norm2_b: DevBuf::upload(&l.norm2_b)?,
                    lin1_w: DevBuf::bf16(&l.lin1_w)?,
                    lin1_b: DevBuf::upload(&l.lin1_b)?,
                    lin2_w: DevBuf::bf16(&l.lin2_w)?,
                    lin2_b: DevBuf::upload(&l.lin2_b)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let sc = &w.scorer;
        let name = format!("cuda:{ordinal} ({})", ctx.name().unwrap_or_default());
        Ok(Self {
            emb,
            emb_norm: DevBuf::upload(&w.emb_norm)?,
            layers,
            final_norm: DevBuf::upload(&w.final_norm)?,
            type_emb: DevBuf::upload(&w.type_emb)?,
            head,
            s_norm_w: DevBuf::upload(&sc.norm_w)?,
            s_norm_b: DevBuf::upload(&sc.norm_b)?,
            s_w1: DevBuf::bf16(&sc.w1)?,
            s_b1: DevBuf::upload(&sc.b1)?,
            s_w2: DevBuf::upload(&sc.w2)?,
            s_b2: sc.b2,
            ropes,
            cfg,
            ctx,
            stream,
            blas,
            k,
            _module: module,
            ws: Mutex::new(Workspace::default()),
            name,
        })
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    fn ensure_workspace(&self, ws: &mut Workspace, tokens: usize, markers: usize, meta_len: usize) -> Result<Ws> {
        let cfg = &self.cfg;
        let (h, inter, ff) = (cfg.hidden, cfg.intermediate, cfg.head_ff);
        if tokens > ws.tokens || markers > ws.markers {
            let t = tokens.max(ws.tokens).next_power_of_two().max(1024);
            let m = markers.max(ws.markers).next_power_of_two().max(256);
            ws.bufs.clear();
            for bytes in [
                t * h * 4,                      // x
                t * h * 2,                      // xn
                t * 3 * h * 2,                  // qkv
                t * h * 2,                      // ctx
                t * (2 * inter).max(ff) * 2,    // h
                t * inter * 2,                  // g
                t * (3 * h).max(ff) * 4,        // f32 scratch
                m * h * 4,                      // ym
                m * h * 2,                      // ymb
                m * h * 2,                      // cm
                t * h * 2,                      // delta (bf16 GEMM output added in the next norm)
                m * h * 2,                      // dm (marker delta)
                m * 4,                          // scores
            ] {
                ws.bufs.push(DevBuf::new(bytes)?);
            }
            ws.tokens = t;
            ws.markers = m;
            ws.scores_host = Some(Pinned::new(m * 4)?);
        }
        if meta_len > ws.meta_len {
            let n = meta_len.next_power_of_two().max(4096);
            ws.meta = Some(DevBuf::new(n * 4)?);
            ws.host = Some(Pinned::new(n * 4)?);
            ws.meta_len = n;
        }
        let b: Vec<u64> = ws.bufs.iter().map(|x| x.ptr).collect();
        Ok(Ws {
            x: b[0],
            xn: b[1],
            qkv: b[2],
            ctx: b[3],
            h: b[4],
            g: b[5],
            f32s: b[6],
            ym: b[7],
            ymb: b[8],
            cm: b[9],
            delta: b[10],
            dm: b[11],
            scores: b[12],
        })
    }

    fn launch(&self, f: sys::CUfunction, grid: (u32, u32, u32), block: u32, params: &mut [*mut c_void]) -> Result<()> {
        // SAFETY: params match the kernel signatures in kernels.cu; buffers are sized by ensure_workspace.
        unsafe { cu::launch_kernel(f, grid, (block, 1, 1), 0, self.stream.cu_stream(), params)? };
        Ok(())
    }

    /// One warp per 384-wide row, 8 rows per block.
    fn rows_grid(rows: usize) -> (u32, u32, u32) {
        (rows.div_ceil(8) as u32, 1, 1)
    }

    /// c[rows, n] = a[rows, k] · Wᵀ (+ beta·c). `a`, `w` BF16; `c` FP32 or BF16.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, a: u64, w: u64, c: u64, rows: usize, n: usize, k: usize, c_f32: bool, beta: f32) -> Result<()> {
        let alpha = 1f32;
        let ctype = if c_f32 { bsys::cudaDataType::CUDA_R_32F } else { bsys::cudaDataType::CUDA_R_16BF };
        // SAFETY: device pointers are valid for the given shapes.
        unsafe {
            blas::gemm_ex(
                *self.blas.handle(),
                bsys::cublasOperation_t::CUBLAS_OP_T,
                bsys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                rows as i32,
                k as i32,
                p(&alpha),
                w as *const c_void,
                bsys::cudaDataType::CUDA_R_16BF,
                k as i32,
                a as *const c_void,
                bsys::cudaDataType::CUDA_R_16BF,
                k as i32,
                p(&beta),
                c as *mut c_void,
                ctype,
                n as i32,
                bsys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                bsys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
            )?;
        }
        Ok(())
    }

    /// x += delta + pending (each optional, 0 = none), then out = bf16(LayerNorm(x)).
    #[allow(clippy::too_many_arguments)]
    fn ln(&self, x: u64, delta: u64, pending: u64, w: &DevBuf, b: u64, out: u64, rows: usize, eps: f32) -> Result<()> {
        let rows_i = rows as i32;
        self.launch(self.k.ln_bf16, Self::rows_grid(rows), 256, &mut [p(&x), p(&delta), p(&pending), p(&w.ptr), p(&b), p(&out), p(&rows_i), p(&eps)])
    }

    fn attention(&self, rope: Option<usize>, qkv: u64, out: u64, tiles_ptr: u64, ntiles: usize, window: i32) -> Result<()> {
        let (f, cos, sin) = match rope {
            Some(r) => (self.k.attention_rope, self.ropes[r].0.ptr, self.ropes[r].1.ptr),
            None => (self.k.attention_plain, 0, 0),
        };
        let ld = (3 * self.cfg.hidden) as i32;
        let ldo = self.cfg.hidden as i32;
        let scale = 0.125f32 * std::f32::consts::LOG2_E;
        self.launch(
            f,
            (ntiles as u32, self.cfg.heads as u32, 1),
            128,
            &mut [p(&qkv), p(&ld), p(&out), p(&ldo), p(&tiles_ptr), p(&cos), p(&sin), p(&window), p(&scale)],
        )
    }

    pub fn forward(&self, batch: &Batch) -> Result<Vec<Vec<f32>>> {
        let cfg = &self.cfg;
        let (h, inter, ff) = (cfg.hidden, cfg.intermediate, cfg.head_ff);
        let eps = cfg.norm_eps;
        let t = batch.tokens();
        let m = batch.marker_rows.len();
        let mut tiles: Vec<i32> = Vec::new();
        for (&s, &len) in batch.starts.iter().zip(&batch.lens) {
            for q0 in (0..len).step_by(BR) {
                tiles.extend([s as i32, len as i32, q0 as i32]);
            }
        }
        let ntiles = tiles.len() / 3;
        let meta_len = 2 * t + tiles.len() + m;
        self.ctx.bind_to_thread()?;
        let mut guard = self.ws.lock().unwrap();
        let ws = &mut *guard;
        let d = self.ensure_workspace(ws, t, m, meta_len)?;

        // Upload [ids | per-token qtype | tiles | marker rows] in one transfer.
        let meta_dev = ws.meta.as_ref().unwrap().ptr;
        let host = ws.host.as_ref().unwrap();
        // SAFETY: pinned buffer holds meta_len i32 values (checked by ensure_workspace).
        let meta = unsafe { std::slice::from_raw_parts_mut(host.ptr as *mut i32, meta_len) };
        debug_assert!(meta_len * 4 <= host.bytes);
        for (dst, &id) in meta[..t].iter_mut().zip(&batch.ids) {
            *dst = id as i32;
        }
        let mut o = t;
        for (s, &len) in batch.lens.iter().enumerate() {
            meta[o..o + len].fill(batch.qtypes[s] as i32);
            o += len;
        }
        meta[2 * t..2 * t + tiles.len()].copy_from_slice(&tiles);
        for (dst, &r) in meta[2 * t + tiles.len()..].iter_mut().zip(&batch.marker_rows) {
            *dst = r as i32;
        }
        // SAFETY: device meta buffer is at least meta_len i32 long.
        unsafe { cu::memcpy_htod_async(meta_dev, &meta[..], self.stream.cu_stream())? };
        let ids_ptr = meta_dev;
        let qtype_ptr = meta_dev + (t * 4) as u64;
        let tiles_ptr = meta_dev + (2 * t * 4) as u64;
        let markers_ptr = meta_dev + ((2 * t + tiles.len()) * 4) as u64;
        let ti = t as i32;
        let null = 0u64;

        self.launch(self.k.embed_ln, Self::rows_grid(t), 256, &mut [p(&ids_ptr), p(&self.emb.ptr), p(&self.emb_norm.ptr), p(&d.x), p(&d.xn), p(&ti), p(&eps)])?;
        // Linear outputs are BF16 (as under autocast); each residual add happens in the next norm.
        let mut delta = null;
        for layer in &self.layers {
            if let Some(norm) = &layer.attn_norm {
                self.ln(d.x, delta, null, norm, null, d.xn, t, eps)?;
            }
            self.gemm(d.xn, layer.wqkv.ptr, d.qkv, t, 3 * h, h, false, 0.0)?;
            let window = if layer.global { -1 } else { cfg.half_window as i32 };
            self.attention(Some(layer.rope), d.qkv, d.ctx, tiles_ptr, ntiles, window)?;
            self.gemm(d.ctx, layer.wo.ptr, d.delta, t, h, h, false, 0.0)?;
            self.ln(d.x, d.delta, null, &layer.mlp_norm, null, d.xn, t, eps)?;
            self.gemm(d.xn, layer.wi.ptr, d.h, t, 2 * inter, h, false, 0.0)?;
            let (inter_i, n8) = (inter as i32, (t * inter).div_ceil(8));
            self.launch(self.k.geglu, (n8.div_ceil(256) as u32, 1, 1), 256, &mut [p(&d.h), p(&d.g), p(&ti), p(&inter_i)])?;
            self.gemm(d.g, layer.wo2.ptr, d.delta, t, h, inter, false, 0.0)?;
            delta = d.delta;
        }
        let h0 = &self.head[0];
        self.launch(
            self.k.final_norm,
            Self::rows_grid(t),
            256,
            &mut [p(&d.x), p(&d.delta), p(&self.final_norm.ptr), p(&self.type_emb.ptr), p(&qtype_ptr), p(&h0.norm1_w.ptr), p(&h0.norm1_b.ptr), p(&d.xn), p(&ti), p(&eps)],
        )?;
        let bias_act = |input: u64, bias: u64, out: u64, rows: usize, cols: usize, relu: i32| -> Result<()> {
            let (r, c) = (rows as i32, cols as i32);
            let n4 = (rows * cols).div_ceil(4);
            self.launch(self.k.bias_act, (n4.div_ceil(256) as u32, 1, 1), 256, &mut [p(&input), p(&bias), p(&out), p(&r), p(&c), p(&relu)])
        };
        let gather = |src: u64, dst: u64, elem: i32| -> Result<()> {
            let mi = m as i32;
            self.launch(self.k.gather_rows, Self::rows_grid(m), 256, &mut [p(&src), p(&markers_ptr), p(&dst), p(&mi), p(&elem)])
        };
        let (mut delta, mut pending) = (null, null);
        let nh = self.head.len();
        for (j, layer) in self.head.iter().enumerate() {
            if j > 0 {
                self.ln(d.x, delta, pending, &layer.norm1_w, layer.norm1_b.ptr, d.xn, t, 1e-5)?;
            }
            self.gemm(d.xn, layer.in_w.ptr, d.f32s, t, 3 * h, h, true, 0.0)?;
            bias_act(d.f32s, layer.in_b.ptr, d.qkv, t, 3 * h, 0)?;
            self.attention(None, d.qkv, d.ctx, tiles_ptr, ntiles, -1)?;
            if j + 1 < nh {
                self.gemm(d.ctx, layer.out_w.ptr, d.delta, t, h, h, false, 0.0)?;
                self.ln(d.x, d.delta, layer.out_b.ptr, &layer.norm2_w, layer.norm2_b.ptr, d.xn, t, 1e-5)?;
                self.gemm(d.xn, layer.lin1_w.ptr, d.f32s, t, ff, h, true, 0.0)?;
                bias_act(d.f32s, layer.lin1_b.ptr, d.h, t, ff, 1)?;
                self.gemm(d.h, layer.lin2_w.ptr, d.delta, t, h, ff, false, 0.0)?;
                (delta, pending) = (d.delta, layer.lin2_b.ptr);
            } else {
                // Last layer: only option markers feed the scorer.
                gather(d.ctx, d.cm, 2)?;
                gather(d.x, d.ym, 4)?;
                self.gemm(d.cm, layer.out_w.ptr, d.dm, m, h, h, false, 0.0)?;
                self.ln(d.ym, d.dm, layer.out_b.ptr, &layer.norm2_w, layer.norm2_b.ptr, d.ymb, m, 1e-5)?;
                self.gemm(d.ymb, layer.lin1_w.ptr, d.f32s, m, ff, h, true, 0.0)?;
                bias_act(d.f32s, layer.lin1_b.ptr, d.h, m, ff, 1)?;
                self.gemm(d.h, layer.lin2_w.ptr, d.dm, m, h, ff, false, 0.0)?;
                (delta, pending) = (d.dm, layer.lin2_b.ptr);
            }
        }
        self.ln(d.ym, delta, pending, &self.s_norm_w, self.s_norm_b.ptr, d.ymb, m, 1e-5)?;
        self.gemm(d.ymb, self.s_w1.ptr, d.f32s, m, h, h, true, 0.0)?;
        let mi = m as i32;
        self.launch(self.k.scorer, Self::rows_grid(m), 256, &mut [p(&d.f32s), p(&self.s_b1.ptr), p(&self.s_w2.ptr), p(&self.s_b2), p(&d.scores), p(&mi)])?;
        let sh = ws.scores_host.as_ref().unwrap();
        // SAFETY: pinned scores buffer holds at least m floats.
        let scores = unsafe { std::slice::from_raw_parts_mut(sh.ptr as *mut f32, m) };
        debug_assert!(m * 4 <= sh.bytes);
        unsafe {
            cu::memcpy_dtoh_async(scores, d.scores, self.stream.cu_stream())?;
        }
        self.stream.synchronize()?;
        Ok(batch.split_scores(scores))
    }
}
