// Julia-1 CUDA kernels. Residual stream FP32, GEMM operands BF16 (FP32 accumulate,
// cuBLAS), attention on tensor cores (mma.sync m16n8k16, FlashAttention-2 style).
// Sequences are packed back to back without padding.
#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
#define HID 384
#define HD 64
#define PER_LANE (HID / 32)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Two-pass LayerNorm over one 384-wide row held as 3 float4 per lane.
__device__ __forceinline__ void ln_row(float4 (&v)[3], const float* __restrict__ w, const float* __restrict__ b, float eps, int lane) {
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) s += v[j].x + v[j].y + v[j].z + v[j].w;
    float mean = warp_sum(s) * (1.f / HID);
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        float a = v[j].x - mean, c = v[j].y - mean, d = v[j].z - mean, e = v[j].w - mean;
        q += a * a + c * c + d * d + e * e;
    }
    float rstd = rsqrtf(warp_sum(q) * (1.f / HID) + eps);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        int i = j * 128 + lane * 4;
        float4 g = *reinterpret_cast<const float4*>(w + i);
        float4 bb = b ? *reinterpret_cast<const float4*>(b + i) : make_float4(0.f, 0.f, 0.f, 0.f);
        v[j].x = (v[j].x - mean) * rstd * g.x + bb.x;
        v[j].y = (v[j].y - mean) * rstd * g.y + bb.y;
        v[j].z = (v[j].z - mean) * rstd * g.z + bb.z;
        v[j].w = (v[j].w - mean) * rstd * g.w + bb.w;
    }
}

__device__ __forceinline__ void load_row(float4 (&v)[3], const float* __restrict__ row, int lane) {
#pragma unroll
    for (int j = 0; j < 3; ++j) v[j] = *reinterpret_cast<const float4*>(row + j * 128 + lane * 4);
}

__device__ __forceinline__ void store_row(float* __restrict__ row, const float4 (&v)[3], int lane) {
#pragma unroll
    for (int j = 0; j < 3; ++j) *reinterpret_cast<float4*>(row + j * 128 + lane * 4) = v[j];
}

__device__ __forceinline__ void store_row_bf16(bf16* __restrict__ row, const float4 (&v)[3], int lane) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        __nv_bfloat162 a = __floats2bfloat162_rn(v[j].x, v[j].y);
        __nv_bfloat162 b = __floats2bfloat162_rn(v[j].z, v[j].w);
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&a);
        packed.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(row + j * 128 + lane * 4) = packed;
    }
}

// x = LayerNorm(table[id]) (no bias); xb = bf16(x) (layer 0 attention input is the identity).
extern "C" __global__ void embed_ln(const uint32_t* __restrict__ ids, const float* __restrict__ table, const float* __restrict__ w,
                                    float* __restrict__ x, bf16* __restrict__ xb, int rows, float eps) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    float4 v[3];
    load_row(v, table + (size_t)ids[row] * HID, lane);
    ln_row(v, w, nullptr, eps, lane);
    store_row(x + (size_t)row * HID, v, lane);
    store_row_bf16(xb + (size_t)row * HID, v, lane);
}

__device__ __forceinline__ void add_bf16(float4 (&v)[3], const bf16* __restrict__ row, int lane) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        uint2 raw = *reinterpret_cast<const uint2*>(row + j * 128 + lane * 4);
        float2 a = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.x));
        float2 b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&raw.y));
        v[j].x += a.x; v[j].y += a.y; v[j].z += b.x; v[j].w += b.y;
    }
}

__device__ __forceinline__ void add_f32(float4 (&v)[3], const float* __restrict__ p, int lane) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        float4 q = *reinterpret_cast<const float4*>(p + j * 128 + lane * 4);
        v[j].x += q.x; v[j].y += q.y; v[j].z += q.z; v[j].w += q.w;
    }
}

// Residual update x += delta (bf16 GEMM output) + pending (bias), written back when
// either is given, then out = bf16(LayerNorm(x; w, b)).
extern "C" __global__ void ln_bf16(float* __restrict__ x, const bf16* __restrict__ delta, const float* __restrict__ pending,
                                   const float* __restrict__ w, const float* __restrict__ b, bf16* __restrict__ out, int rows, float eps) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    float4 v[3];
    float* xr = x + (size_t)row * HID;
    load_row(v, xr, lane);
    if (delta) add_bf16(v, delta + (size_t)row * HID, lane);
    if (pending) add_f32(v, pending, lane);
    if (delta || pending) store_row(xr, v, lane);
    ln_row(v, w, b, eps, lane);
    store_row_bf16(out + (size_t)row * HID, v, lane);
}

// Encoder final norm + question-type embedding (written back to x), fused with the
// first head layer's norm1 -> out (bf16).
extern "C" __global__ void final_norm(float* __restrict__ x, const bf16* __restrict__ delta, const float* __restrict__ fw, const float* __restrict__ type_emb,
                                      const int32_t* __restrict__ qtype, const float* __restrict__ hw, const float* __restrict__ hb,
                                      bf16* __restrict__ out, int rows, float eps) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    float4 v[3];
    float* xr = x + (size_t)row * HID;
    load_row(v, xr, lane);
    add_bf16(v, delta + (size_t)row * HID, lane);
    ln_row(v, fw, nullptr, eps, lane);
    const float* te = type_emb + qtype[row] * HID;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        float4 p = *reinterpret_cast<const float4*>(te + j * 128 + lane * 4);
        v[j].x += p.x; v[j].y += p.y; v[j].z += p.z; v[j].w += p.w;
    }
    store_row(xr, v, lane);
    ln_row(v, hw, hb, 1e-5f, lane);
    store_row_bf16(out + (size_t)row * HID, v, lane);
}

__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); }

// g[r, i] = gelu(h[r, i]) * h[r, inter + i]
extern "C" __global__ void geglu(const bf16* __restrict__ h, bf16* __restrict__ g, int rows, int inter) {
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 8;
    size_t total = (size_t)rows * inter;
    if (idx >= total) return;
    size_t r = idx / inter, c = idx % inter;
    const bf16* src = h + r * 2 * inter + c;
    uint4 a = *reinterpret_cast<const uint4*>(src);
    uint4 b = *reinterpret_cast<const uint4*>(src + inter);
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
    uint4 o;
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 x = __bfloat1622float2(a2[i]), y = __bfloat1622float2(b2[i]);
        o2[i] = __floats2bfloat162_rn(gelu(x.x) * y.x, gelu(x.y) * y.y);
    }
    *reinterpret_cast<uint4*>(g + idx) = o;
}

// out = bf16(act(in + bias)) for an FP32 [rows, cols] GEMM result.
extern "C" __global__ void bias_act(const float* __restrict__ in, const float* __restrict__ bias, bf16* __restrict__ out,
                                    int rows, int cols, int relu) {
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= (size_t)rows * cols) return;
    int c = idx % cols;
    float4 v = *reinterpret_cast<const float4*>(in + idx);
    float4 b = *reinterpret_cast<const float4*>(bias + c);
    v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w;
    if (relu) { v.x = fmaxf(v.x, 0.f); v.y = fmaxf(v.y, 0.f); v.z = fmaxf(v.z, 0.f); v.w = fmaxf(v.w, 0.f); }
    __nv_bfloat162 p = __floats2bfloat162_rn(v.x, v.y), q = __floats2bfloat162_rn(v.z, v.w);
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&p);
    packed.y = *reinterpret_cast<uint32_t*>(&q);
    *reinterpret_cast<uint2*>(out + idx) = packed;
}

// dst[i] = src[idx[i]] for 384-wide rows; one warp per row. elem_bytes is 4 (f32) or 2 (bf16).
extern "C" __global__ void gather_rows(const char* __restrict__ src, const int32_t* __restrict__ idx, char* __restrict__ dst,
                                       int rows, int elem_bytes) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    size_t bytes = (size_t)HID * elem_bytes;
    const uint4* s = reinterpret_cast<const uint4*>(src + (size_t)idx[row] * bytes);
    uint4* d = reinterpret_cast<uint4*>(dst + (size_t)row * bytes);
    for (int i = lane; i < (int)(bytes / 16); i += 32) d[i] = s[i];
}

// score[r] = sum_i gelu(z[r, i] + b1[i]) * w2[i] + b2
extern "C" __global__ void scorer(const float* __restrict__ z, const float* __restrict__ b1, const float* __restrict__ w2, float b2,
                                  float* __restrict__ out, int rows) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    float s = 0.f;
    for (int i = lane; i < HID; i += 32) s += gelu(z[(size_t)row * HID + i] + b1[i]) * w2[i];
    s = warp_sum(s);
    if (lane == 0) out[row] = s + b2;
}

// ---------------------------------------------------------------- attention
#define BR 64
#define BC 64
#define SPAD 72  // shared row stride (bf16): conflict-free fragment loads

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Load 64 rows x 64 dims of one head into shared memory (zero rows past len),
// applying rotate-half RoPE at sequence position `pos0 + r` when ROPE.
template <bool ROPE>
__device__ __forceinline__ void load_tile(bf16 (*dst)[SPAD], const bf16* __restrict__ base, int ld, int pos0, int len,
                                          const float* __restrict__ cos_t, const float* __restrict__ sin_t) {
    int r = threadIdx.x >> 1, half = threadIdx.x & 1;  // 2 threads per row, 16 rotary pairs each
    int p = pos0 + r;
    uint4 lo[2], hi[2];
    if (p < len) {
        const bf16* src = base + (size_t)p * ld + half * 16;
        lo[0] = *reinterpret_cast<const uint4*>(src);
        lo[1] = *reinterpret_cast<const uint4*>(src + 8);
        hi[0] = *reinterpret_cast<const uint4*>(src + 32);
        hi[1] = *reinterpret_cast<const uint4*>(src + 40);
        if (ROPE) {
            const float* c = cos_t + (size_t)p * 32 + half * 16;
            const float* s = sin_t + (size_t)p * 32 + half * 16;
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                __nv_bfloat162* a = reinterpret_cast<__nv_bfloat162*>(&lo[k]);
                __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&hi[k]);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float2 x = __bfloat1622float2(a[i]), y = __bfloat1622float2(b[i]);
                    int d = k * 8 + i * 2;
                    float c0 = c[d], c1 = c[d + 1], s0 = s[d], s1 = s[d + 1];
                    a[i] = __floats2bfloat162_rn(x.x * c0 - y.x * s0, x.y * c1 - y.y * s1);
                    b[i] = __floats2bfloat162_rn(y.x * c0 + x.x * s0, y.y * c1 + x.y * s1);
                }
            }
        }
    } else {
        lo[0] = lo[1] = hi[0] = hi[1] = make_uint4(0, 0, 0, 0);
    }
    bf16* row = dst[r] + half * 16;
    *reinterpret_cast<uint4*>(row) = lo[0];
    *reinterpret_cast<uint4*>(row + 8) = lo[1];
    *reinterpret_cast<uint4*>(row + 32) = hi[0];
    *reinterpret_cast<uint4*>(row + 40) = hi[1];
}

// V tile stored transposed: dst[d][key].
__device__ __forceinline__ void load_vt(bf16 (*dst)[SPAD], const bf16* __restrict__ base, int ld, int pos0, int len) {
    int r = threadIdx.x >> 1, half = threadIdx.x & 1;
    int p = pos0 + r;
    uint4 v[4];
    if (p < len) {
        const uint4* src = reinterpret_cast<const uint4*>(base + (size_t)p * ld + half * 32);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = src[i];
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = make_uint4(0, 0, 0, 0);
    }
    const bf16* e = reinterpret_cast<const bf16*>(v);
#pragma unroll
    for (int i = 0; i < 32; ++i) dst[half * 32 + i][r] = e[i];
}

// One block = (64-query tile, head). tiles[3*t] = {sequence start row, length, first query}.
// qkv rows are [q | k | v] (each heads*64 wide); out rows are heads*64 wide.
template <bool ROPE>
__device__ void attention_impl(const bf16* __restrict__ qkv, int ld, bf16* __restrict__ out, int ldo,
                               const int32_t* __restrict__ tiles, const float* __restrict__ cos_t,
                               const float* __restrict__ sin_t, int window, float scale_log2) {
    __shared__ __align__(16) bf16 sQ[BR][SPAD];
    __shared__ __align__(16) bf16 sK[BC][SPAD];
    __shared__ __align__(16) bf16 sV[HD][SPAD];
    const int tile = blockIdx.x, head = blockIdx.y;
    const int start = tiles[3 * tile], len = tiles[3 * tile + 1], q0 = tiles[3 * tile + 2];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int hidden = gridDim.y * HD;
    const bf16* seq = qkv + (size_t)start * ld + head * HD;

    load_tile<ROPE>(sQ, seq, ld, q0, len, cos_t, sin_t);
    __syncthreads();
    uint32_t qf[4][4];
    const int qr = warp * 16 + g;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        qf[kk][0] = *reinterpret_cast<const uint32_t*>(&sQ[qr][kk * 16 + t * 2]);
        qf[kk][1] = *reinterpret_cast<const uint32_t*>(&sQ[qr + 8][kk * 16 + t * 2]);
        qf[kk][2] = *reinterpret_cast<const uint32_t*>(&sQ[qr][kk * 16 + t * 2 + 8]);
        qf[kk][3] = *reinterpret_cast<const uint32_t*>(&sQ[qr + 8][kk * 16 + t * 2 + 8]);
    }
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const int row0 = q0 + qr, row1 = row0 + 8;

    int kbeg = 0, kend = len;
    if (window >= 0) {
        kbeg = max(0, q0 - window);
        kend = min(len, q0 + BR + window);
    }
    for (int k0 = kbeg; k0 < kend; k0 += BC) {
        __syncthreads();
        load_tile<ROPE>(sK, seq + hidden, ld, k0, len, cos_t, sin_t);
        load_vt(sV, seq + 2 * hidden, ld, k0, len);
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sK[n * 8 + g][kk * 16 + t * 2]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sK[n * 8 + g][kk * 16 + t * 2 + 8]);
                mma_bf16(s[n], qf[kk], b0, b1);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int key = k0 + n * 8 + t * 2 + (e & 1);
                int row = e < 2 ? row0 : row1;
                bool ok = key < kend && (window < 0 || abs(row - key) <= window);
                s[n][e] = ok ? s[n][e] * scale_log2 : -INFINITY;
            }
            mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
            mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        float u0 = n0 == -INFINITY ? 0.f : n0, u1 = n1 == -INFINITY ? 0.f : n1;
        float a0 = exp2f(m0 - u0), a1 = exp2f(m1 - u1);
        m0 = n0;
        m1 = n1;
        l0 *= a0;
        l1 *= a1;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1;
            s[n][0] = exp2f(s[n][0] - u0); s[n][1] = exp2f(s[n][1] - u0);
            s[n][2] = exp2f(s[n][2] - u1); s[n][3] = exp2f(s[n][3] - u1);
            l0 += s[n][0] + s[n][1];
            l1 += s[n][2] + s[n][3];
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            uint32_t pa[4] = {pack_bf16(s[2 * kk][0], s[2 * kk][1]), pack_bf16(s[2 * kk][2], s[2 * kk][3]),
                              pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sV[n * 8 + g][kk * 16 + t * 2]);
                uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sV[n * 8 + g][kk * 16 + t * 2 + 8]);
                mma_bf16(o[n], pa, b0, b1);
            }
        }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
    bf16* dst = out + (size_t)start * ldo + head * HD;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        int c = n * 8 + t * 2;
        if (row0 < len) *reinterpret_cast<uint32_t*>(dst + (size_t)row0 * ldo + c) = pack_bf16(o[n][0] * i0, o[n][1] * i0);
        if (row1 < len) *reinterpret_cast<uint32_t*>(dst + (size_t)row1 * ldo + c) = pack_bf16(o[n][2] * i1, o[n][3] * i1);
    }
}

extern "C" __global__ void __launch_bounds__(128) attention_rope(const bf16* qkv, int ld, bf16* out, int ldo, const int32_t* tiles,
                                                                 const float* cos_t, const float* sin_t, int window, float scale_log2) {
    attention_impl<true>(qkv, ld, out, ldo, tiles, cos_t, sin_t, window, scale_log2);
}

extern "C" __global__ void __launch_bounds__(128) attention_plain(const bf16* qkv, int ld, bf16* out, int ldo, const int32_t* tiles,
                                                                  const float* cos_t, const float* sin_t, int window, float scale_log2) {
    attention_impl<false>(qkv, ld, out, ldo, tiles, cos_t, sin_t, window, scale_log2);
}
