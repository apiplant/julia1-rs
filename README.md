# julia1 — Rust runtime for Julia-1

A Rust port of the inference runtime for [Supersonic Labs **Julia-1**](https://huggingface.co/SupersonicLabs/Julia-1)
(the `julia` Python package: `FastEngine`, the named typed-question API and the
hierarchical `Router`), with two backends:

- **CPU, FP32.** AVX-512 register-tiled GEMM over pre-packed weights with fused
  epilogues (bias, ReLU, residual add, GEGLU), packed tensor attention, and a portable
  fallback through the `gemm` crate.
- **CUDA, BF16 tensor cores** (feature `cuda`). cuBLAS GEMMs with FP32 accumulation,
  an FP32 residual stream, and a FlashAttention-2-style `mma.sync` kernel with
  in-kernel RoPE and sliding window. This is the same precision as the Python
  runtime's CUDA path (autocast BF16).

It reads the unmodified checkpoint directory (`model.safetensors`, `encoder/`,
`tokenizer/`, `julia_config.json`).

## Build

```bash
cargo build --release                    # CPU only
cargo build --release --features cuda    # + CUDA (needs nvcc; compute capability 8.0+)
```

- `.cargo/config.toml` builds with `-C target-cpu=native`. Remove that line for
  portable binaries. The AVX-512 path is selected at runtime either way, and
  `JULIA_NO_AVX512=1` forces the fallback.
- `JULIA_CUDA_ARCHS` (default `80,86,89,90,120`) picks the SASS targets. PTX for
  `compute_80` is always embedded.

## Use

```bash
# Legacy rows or {"state", "questions"} objects, one JSON per line.
echo '{"state": "I was charged twice for the same order.", "questions": {"team": {"type": "choice",
  "instructions": "Which team should handle this request?",
  "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery",
               "access": "Account access and login"}}}}' \
  | ./target/release/julia1 predict --checkpoint /path/to/Julia-1 --device cuda
```

```rust
use julia1::{Engine, EngineOptions, Device, Request, QType, State};

let engine = Engine::load("Julia-1", EngineOptions {
    device: Device::Cuda(0),          // or Device::Cpu (threads: JULIA_CPU_THREADS, default 4)
    strict_encoding: true,
    head_length: 512,                 // max_length defaults to the checkpoint's 8,192
    ..Default::default()
})?;
let answers = engine.predict_typed(&State::from("I was charged twice."), &questions_json)?;
let legacy = engine.predict(&[Request::new("I was charged twice.", "Which team?",
    vec!["Billing".into(), "Shipping".into()], QType::Choice)], true)?;
let raw = engine.logits(&rows)?;                        // raw option logits
let routed = julia1::Router::with_defaults(&engine).route(&row_with_4096_options)?;
```

The API follows the Python runtime:

- `predict_typed` returns full softmax probabilities keyed by caller IDs, plus
  `choice`, `score` (expected index) or `noul` (P(true)), and `max_probability`.
- `predict` returns `index` and display-rounded `probabilities`.
- `logits` returns raw scores.
- Validation errors match Python's `ValueError` cases: 2–20 options, strict-encoding
  rejections, noul criteria, and the rest.

## Parity with the Python runtime

`julia1 check` compares against reference dumps produced by `bench/py_bench.py
reference`. It uses the 2,000 typed-decision test questions (pinned dataset revision)
plus 933, 3,928 and 7,556-token requests.

| | token IDs | argmax vs Python FP32 CPU | logit error vs Python FP32 CPU |
| --- | --- | ---: | ---: |
| Rust CPU (FP32) | identical (all rows, incl. JSON states) | **2000 / 2000** | max 6e-4, mean 3e-5 |
| Rust CUDA (BF16) | identical | 1984 / 2000 | mean 0.166 |
| *Python CUDA (BF16), for reference* | | *1974 / 2000* | *mean 0.173* |

Matching tokenization required reproducing Python's `json.dumps` byte for byte for
JSON states: separators, float `repr`, escaping and key order (`src/pyjson.rs`).

## Benchmarks

Measured with `bench/run_all.sh`, Python and Rust run back to back on the same idle
machine:

- Hardware: AMD Ryzen 9 7950X, RTX 4090.
- Python: torch 2.12 + transformers 5.0, the unmodified `FastEngine` via
  `julia.load_model(..., strict_encoding=True, max_length=8192, head_length=512)`.
- Same rows and the same protocol on both sides.

| Config | Metric | Python | Rust | Speedup |
| --- | --- | ---: | ---: | ---: |
| CUDA | single request, mean | 6.4 ms | 0.9 ms | **6.85×** |
| CUDA | 2,000 rows, batch 16 | 1,434 rows/s | 3,202 rows/s | **2.23×** |
| CUDA | 933-token request | 7.6 ms | 2.0 ms | **3.83×** |
| CUDA | 3,928-token request | 36.8 ms | 7.5 ms | **4.93×** |
| CUDA | 7,556-token request | 114 ms | 20.5 ms | **5.56×** |
| CPU 4 threads | single request, mean | 37.0 ms | 27.6 ms | **1.34×** |
| CPU 4 threads | 2,000 rows, batch 16 | 14.6 rows/s | 19.7 rows/s | **1.35×** |
| CPU 4 threads | 933-token request | 247 ms | 176 ms | **1.40×** |
| CPU 4 threads | 3,928-token request | 2,176 ms | 1,119 ms | **1.94×** |
| CPU 4 threads | 7,556-token request | 6,742 ms | 3,042 ms | **2.22×** |
| CPU 16 threads | single request, mean | 26.3 ms | 19.4 ms | **1.35×** |
| CPU 16 threads | 2,000 rows, batch 16 | 32.9 rows/s | 43.6 rows/s | **1.33×** |
| CPU 16 threads | 933-token request | 125 ms | 88.8 ms | **1.41×** |
| CPU 16 threads | 3,928-token request | 875 ms | 439 ms | **1.99×** |
| CPU 16 threads | 7,556-token request | 2,852 ms | 1,080 ms | **2.64×** |

**What each metric measures**

- *Single request*: one `predict([row])` per call over the first rows of the typed
  set, about 160 tokens each, tokenization included.
- *Batch*: `predict(all 2,000 rows)` with `batch_size=16`, about 293 tokens per row on
  average and 586k tokens total, encoding caches cleared first.
- *N-token request*: one long request, best of 3.
- CPU thread counts are equal on both sides. Python's default is 4
  (`JULIA_CPU_THREADS`).

### Why it is faster

- **No padding.** Each forward pass packs sequences back to back (varlen), so every
  GEMM and attention tile runs on real tokens only.
- **CPU GEMM.** Weights are pre-packed once into 32-column panels. An 8×32 AVX-512 tile
  stays in registers across all of K, reaching ~92–95% of single-core FP32 peak.
  Epilogues fuse bias, ReLU, residual adds and GEGLU, so the `[T, 2304]` MLP
  intermediate is never materialized. Work is split evenly per thread, which matters
  for single requests.
- **Sliding-window layers** (15 of 22) only compute their ±64 band. Python's SDPA path
  masks a full L×L matrix.
- **The last head layer** computes queries, out-projection and FFN for option markers
  only (keys/values stay full). Python does the same on CPU, but not on CUDA.
- **CUDA.** About 200 kernel launches per forward with no Python dispatch. BF16 GEMMs
  run at ~136 TFLOP/s on the 4090. Residual adds are fused into the following
  LayerNorm, and attention is a single tensor-core kernel per layer with RoPE applied
  on load.

## Layout

| Path | Contents |
| --- | --- |
| `src/encode.rs` | `data.sequence` marker serialization, strict checks, LRU token/encoding caches |
| `src/pyjson.rs` | Python-compatible `json.dumps` for JSON states |
| `src/engine.rs` | `FastEngine`: validation, length-sorted micro-batches, `logits` / `predict` |
| `src/typed.rs`, `src/router.rs` | named typed questions; hierarchical router for >20 options |
| `src/cpu/` | FP32 forward pass, AVX-512 GEMM, vectorized `exp`/`erf` |
| `src/cuda/` | CUDA forward pass and `kernels.cu` |
| `bench/` | Python reference/baseline harness, `run_all.sh`, results |
| `tests/` | ports of the Python unit tests; checkpoint tests skip if weights are absent |

```bash
cargo test --release --features cuda   # JULIA_CHECKPOINT=/path/to/Julia-1 for checkpoint tests
python3 bench/py_bench.py export && python3 bench/py_bench.py reference --device cpu
./target/release/julia1 check --device cpu      # parity vs bench/data/ref-cpu.json
```

## Not ported

- Training and distillation code: the `Decisions` dataset, collator targets,
  `act_head` training, and `save_pretrained`.
- INT8 (`quantization.json`) checkpoints.
- The optional Bend native library. Its operations (argmax, softmax, LayerNorm,
  dense projections) are native Rust here.
- `torch.compile`.

`act_head` is unused at inference (`return_actions=False`), as in Python.

`JULIA_PROFILE=1 julia1 bench ...` prints per-op CPU timings.
