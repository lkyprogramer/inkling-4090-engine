// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//
// Qwen3.8-27B (`qwen3_5` text model) operations: Gated DeltaNet linear attention, gated full
// attention with partial RoPE, SwiGLU MLP. Maths follows the Hugging Face `qwen3_5` reference
// and llama.cpp's qwen35 graph (MIT). All buffers are token-major ([t][...]); T tokens per
// launch (decode is T = 1). Parameter lists are mirrored by crates/kernels/src/qwen.rs.

#include <cuda_fp16.h>

// Programmatic dependent launch: on Hopper every kernel may be launched before its predecessor
// finishes (overlapping launch latency); griddepcontrol.wait blocks until the predecessor's
// results are visible, so it is the first thing every kernel does. A no-op otherwise.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if !defined(ASTRAL_PDL_LATE)  // small, mostly single-wave kernels: let the successor start scheduling now
#define PDL_WAIT() asm volatile("griddepcontrol.wait;\n griddepcontrol.launch_dependents;" ::: "memory")
#else
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#endif
#else
#define PDL_WAIT() do {} while (0)
#endif

typedef unsigned long long u64;
typedef unsigned short u16;

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    v = warp_sum(v);
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = lane < (blockDim.x >> 5) ? part[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    return total;
}
__device__ __forceinline__ float bf16f(u16 b) { return __uint_as_float((unsigned)b << 16); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// int8 per-32 quantisation (llama.cpp q8_1 scheme) plus an f16 copy; one value per lane.
__device__ __forceinline__ void q8_store(float v, signed char* __restrict__ xq, float* __restrict__ xd, __half* __restrict__ xh,
                                         u64 blk, unsigned lane) {
    if (xh) xh[blk * 32u + lane] = __float2half_rn(v);
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float d = a * (1.0f / 127.0f);
    xq[blk * 32u + lane] = (signed char)(a == 0.0f ? 0 : __float2int_rn(v / d));
    if (lane == 0) xd[blk] = d;
}

// out[t] = table[token[t]] (BF16 rows of `dim`), clamped row index. One block per token.
extern "C" __global__ void astral_qw_embed_bf16(const u16* __restrict__ table, const int* __restrict__ token, float* __restrict__ out,
                                                unsigned dim, unsigned rows) { PDL_WAIT();
    const unsigned t = blockIdx.x;
    const u64 row = (u64)min((unsigned)token[t], rows - 1u);
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) out[(u64)t * dim + i] = bf16f(table[row * dim + i]);
}

// ---------------------------------------------------------------------------------------------
// Causal depthwise conv (4 taps, no bias) + SiLU over T tokens, every (token, channel) in
// parallel: dst[t][c] = silu(sum_j w[c][j] x[t-3+j][c]), inputs before t = 0 from `state`
// ([c][3], oldest first). Then astral_qw_conv_state stores the last three inputs.
__device__ __forceinline__ float cin(const float* __restrict__ src, unsigned stride, const float* __restrict__ state, int t, unsigned c) {
    return t >= 0 ? src[(u64)t * stride + c] : state[c * 3 + (unsigned)(3 + t)];
}
extern "C" __global__ void astral_qw_conv_silu(const float* __restrict__ src, unsigned src_stride, float* __restrict__ dst, unsigned ntok,
                                               const float* __restrict__ state, const float* __restrict__ w, unsigned channels) { PDL_WAIT();
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= channels || t >= ntok) return;
    const int ti = (int)t;
    const float* wc = w + (u64)c * 4;
    float o = cin(src, src_stride, state, ti - 3, c) * wc[0];
    o = fmaf(cin(src, src_stride, state, ti - 2, c), wc[1], o);
    o = fmaf(cin(src, src_stride, state, ti - 1, c), wc[2], o);
    o = fmaf(src[(u64)t * src_stride + c], wc[3], o);
    dst[(u64)t * channels + c] = silu(o);
}
extern "C" __global__ void astral_qw_conv_state(const float* __restrict__ src, unsigned src_stride, unsigned ntok, float* __restrict__ state,
                                                unsigned channels) { PDL_WAIT();
    const unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int n = (int)ntok;
    const float s0 = cin(src, src_stride, state, n - 3, c), s1 = cin(src, src_stride, state, n - 2, c), s2 = cin(src, src_stride, state, n - 1, c);
    state[c * 3 + 0] = s0;
    state[c * 3 + 1] = s1;
    state[c * 3 + 2] = s2;
}

// ---------------------------------------------------------------------------------------------
// Gated DeltaNet prep, in place on the convolved qkv rows ([t][q (kh*dk) | k (kh*dk) | v]):
//   q, k: L2-normalised per head (x / sqrt(sum x^2 + 1e-6)); q also scaled by 1/sqrt(dk).
//   beta[t][h] = sigmoid(b), g[t][h] = neg_a[h] * softplus(a + dt_bias[h])   (neg_a = -exp(A_log)).
// grid = (kh, T), dk threads (128).
extern "C" __global__ void astral_qw_gdn_prep(float* __restrict__ qkv, unsigned qkv_stride, const float* __restrict__ b,
                                              const float* __restrict__ a, unsigned ab_stride, const float* __restrict__ neg_a,
                                              const float* __restrict__ dt_bias, float* __restrict__ beta, float* __restrict__ g,
                                              unsigned kh, unsigned dk, unsigned vh) { PDL_WAIT();
    const unsigned h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    float* row = qkv + (u64)t * qkv_stride;
    float* q = row + (u64)h * dk;
    float* k = row + (u64)kh * dk + (u64)h * dk;
    const float qv = i < dk ? q[i] : 0.0f, kv = i < dk ? k[i] : 0.0f;
    const float sq = block_sum(qv * qv), sk = block_sum(kv * kv);
    if (i < dk) {
        q[i] = qv * rsqrtf(sq + 1e-6f) * rsqrtf((float)dk);
        k[i] = kv * rsqrtf(sk + 1e-6f);
    }
    if (h == 0) {
        for (unsigned j = i; j < vh; j += blockDim.x) {
            beta[(u64)t * vh + j] = 1.0f / (1.0f + expf(-b[(u64)t * ab_stride + j]));
            const float x = a[(u64)t * ab_stride + j] + dt_bias[j];
            const float sp = x > 20.0f ? x : log1pf(expf(x));   // softplus (torch threshold 20)
            g[(u64)t * vh + j] = neg_a[j] * sp;
        }
    }
}

// Gated delta rule over T tokens. Value head h (key head h / (vh / kh)); state S[i][j]
// (i = key dim, j = value dim) stored [h][i][j]. Each column j is split over 4 lanes (lane & 3
// = quarter q owning rows 32q..32q+31), reduced with shuffles, so a warp covers 8 columns:
//   S *= exp(g); delta_j = (v_j - sum_i S[i][j] k_i) * beta; S[i][j] += k_i delta_j;
//   out_j = sum_i S[i][j] q_i            (q pre-scaled by 1/sqrt(dk))
// Tokens are staged GDN_TC at a time in shared memory (q, k, this block's v, g, beta) so the
// per-token steps need neither barriers nor global loads. dk = dv = 128.
// grid = (vh, 128 / 32), 128 threads (32 columns) per block.
#define GDN_D 128
#define GDN_Q 32
#define GDN_TC 16
extern "C" __global__ void __launch_bounds__(128) astral_qw_gdn_recur(const float* __restrict__ qkv, unsigned qkv_stride,
                                                                     const float* __restrict__ beta, const float* __restrict__ g,
                                                                     float* __restrict__ state, float* __restrict__ out,
                                                                     unsigned ntok, unsigned kh, unsigned vh) { PDL_WAIT();
    const unsigned h = blockIdx.x, lane = threadIdx.x & 31u, q = lane & 3u;
    const unsigned jl = (threadIdx.x >> 5) * 8u + (lane >> 2);   // column within the block
    const unsigned j = blockIdx.y * 32u + jl;
    const unsigned hk = h / (vh / kh);
    __shared__ float qs[GDN_TC][GDN_D], ks[GDN_TC][GDN_D], vs[GDN_TC][32], gs[GDN_TC], bs[GDN_TC];
    float s[GDN_Q];
    float* st = state + (u64)h * GDN_D * GDN_D + (u64)(q * GDN_Q) * GDN_D + j;
#pragma unroll
    for (unsigned i = 0; i < GDN_Q; ++i) s[i] = st[(u64)i * GDN_D];
    for (unsigned t0 = 0; t0 < ntok; t0 += GDN_TC) {
        const unsigned tc = min((unsigned)GDN_TC, ntok - t0);
        __syncthreads();
        for (unsigned x = threadIdx.x; x < tc * GDN_D; x += blockDim.x) {
            const unsigned tt = x / GDN_D, i = x % GDN_D;
            const float* row = qkv + (u64)(t0 + tt) * qkv_stride;
            qs[tt][i] = row[(u64)hk * GDN_D + i];
            ks[tt][i] = row[(u64)kh * GDN_D + (u64)hk * GDN_D + i];
        }
        for (unsigned x = threadIdx.x; x < tc * 32u; x += blockDim.x) {
            const unsigned tt = x / 32u, c = x % 32u;
            vs[tt][c] = qkv[(u64)(t0 + tt) * qkv_stride + 2u * (u64)kh * GDN_D + (u64)h * GDN_D + blockIdx.y * 32u + c];
        }
        if (threadIdx.x < tc) {
            gs[threadIdx.x] = expf(g[(u64)(t0 + threadIdx.x) * vh + h]);
            bs[threadIdx.x] = beta[(u64)(t0 + threadIdx.x) * vh + h];
        }
        __syncthreads();
        for (unsigned tt = 0; tt < tc; ++tt) {
            const float decay = gs[tt], bt = bs[tt];
            const float* kq = ks[tt] + q * GDN_Q;
            const float* qq = qs[tt] + q * GDN_Q;
            float kd = 0.0f;
#pragma unroll
            for (unsigned i = 0; i < GDN_Q; ++i) {
                s[i] *= decay;
                kd = fmaf(s[i], kq[i], kd);
            }
            kd += __shfl_xor_sync(0xffffffffu, kd, 1);
            kd += __shfl_xor_sync(0xffffffffu, kd, 2);
            const float delta = (vs[tt][jl] - kd) * bt;
            float o = 0.0f;
#pragma unroll
            for (unsigned i = 0; i < GDN_Q; ++i) {
                s[i] = fmaf(kq[i], delta, s[i]);
                o = fmaf(s[i], qq[i], o);
            }
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            if (q == 0) out[(u64)(t0 + tt) * vh * GDN_D + (u64)h * GDN_D + j] = o;
        }
    }
#pragma unroll
    for (unsigned i = 0; i < GDN_Q; ++i) st[(u64)i * GDN_D] = s[i];
}

// Chunked gated delta rule (prefill): the same recurrence as astral_qw_gdn_recur, exactly
// rearranged per chunk of C = 64 tokens (WY form). With cumulative log decay G (A = exp G),
// state S0 at the chunk start and per-token corrections D:
//   (I + M) D = diag(beta) (V - diag(A) K S0),  M[t][s] = beta_t e^{G_t - G_s} (k_t . k_s), s < t
//   => D = U - W S0 with U = (I+M)^-1 diag(beta) V and W = (I+M)^-1 diag(beta A) K (forward subst.)
//   O = diag(A) Q S0 + (P o L) D,               P[t][s] = e^{G_t - G_s} (q_t . k_s), s <= t
//   S_C = A_C S0 + sum_s e^{G_C - G_s} k_s D_s^T
// Decay ratios are always exp(G_t - G_s) (never divided), so strong decay underflows to 0
// cleanly. Tokens past the end are padded with beta = 0, g = 0 (no effect). Each block handles
// one value head and GDN_CB value columns; grid = (vh, 128 / GDN_CB), 256 threads, dynamic
// shared memory GDN_CHUNK_SMEM bytes (state S[128][CB], K and Q [64][129], V, U/D, M, P
// [64][CB + 1] / [64][65], W [64][129], G / beta / ratios).
#define GDN_C 64
#define GDN_CB 64
#define GDN_LK 132   // K / Q / W rows (16-byte aligned)
#define GDN_LU 68    // V / U rows
#define GDN_LM 65    // M / P rows
extern "C" __global__ void __launch_bounds__(256) astral_qw_gdn_chunk(const float* __restrict__ qkv, unsigned qkv_stride,
                                                                     const float* __restrict__ beta, const float* __restrict__ g,
                                                                     float* __restrict__ state, float* __restrict__ out,
                                                                     unsigned ntok, unsigned kh, unsigned vh) { PDL_WAIT();
    extern __shared__ __align__(16) float sm[];
    float* S = sm;                                   // [128][64]
    float* Ks = S + GDN_D * GDN_CB;                  // [64][132]
    float* Qs = Ks + GDN_C * GDN_LK;
    float* Ws = Qs + GDN_C * GDN_LK;
    float* Vs = Ws + GDN_C * GDN_LK;                 // [64][68]
    float* Us = Vs + GDN_C * GDN_LU;                 // [64][68]  (U, then D)
    float* Ms = Us + GDN_C * GDN_LU;                 // [64][65]
    float* Ps = Ms + GDN_C * GDN_LM;                 // [64][65]
    float* Gs = Ps + GDN_C * GDN_LM;                 // cumulative log decay
    float* Bs = Gs + GDN_C;                          // beta
    float* Rs = Bs + GDN_C;                          // e^{G_C - G_s}
    float* As = Rs + GDN_C;                          // e^{G_t}
    const unsigned h = blockIdx.x, c0 = blockIdx.y * GDN_CB, tid = threadIdx.x;
    const unsigned hk = h / (vh / kh);
    const unsigned ty = tid >> 4, tx = tid & 15u;    // 16 x 16 thread grid for the tiled phases
    float* st = state + (u64)h * GDN_D * GDN_D;
    for (unsigned x = tid; x < GDN_D * GDN_CB; x += blockDim.x) S[x] = st[(u64)(x / GDN_CB) * GDN_D + c0 + x % GDN_CB];
    for (unsigned t0 = 0; t0 < ntok; t0 += GDN_C) {
        const unsigned tc = min((unsigned)GDN_C, ntok - t0);
        __syncthreads();
        for (unsigned x = tid; x < GDN_C * GDN_D; x += blockDim.x) {
            const unsigned t = x / GDN_D, i = x % GDN_D;
            float qv = 0.0f, kv = 0.0f;
            if (t < tc) {
                const float* row = qkv + (u64)(t0 + t) * qkv_stride;
                qv = row[(u64)hk * GDN_D + i];
                kv = row[(u64)kh * GDN_D + (u64)hk * GDN_D + i];
            }
            Qs[t * GDN_LK + i] = qv;
            Ks[t * GDN_LK + i] = kv;
        }
        for (unsigned x = tid; x < GDN_C * GDN_CB; x += blockDim.x) {
            const unsigned t = x / GDN_CB, c = x % GDN_CB;
            Vs[t * GDN_LU + c] = t < tc ? qkv[(u64)(t0 + t) * qkv_stride + 2u * (u64)kh * GDN_D + (u64)h * GDN_D + c0 + c] : 0.0f;
        }
        if (tid < 32) {
            const unsigned a = 2u * tid, b = a + 1u;
            const float ga = a < tc ? g[(u64)(t0 + a) * vh + h] : 0.0f, gb0 = b < tc ? g[(u64)(t0 + b) * vh + h] : 0.0f;
            const float gb = ga + gb0;
            float sum = gb;
            for (int o = 1; o < 32; o <<= 1) {
                const float n = __shfl_up_sync(0xffffffffu, sum, o);
                if ((int)tid >= o) sum += n;
            }
            const float excl = sum - gb;
            Gs[a] = excl + ga;
            Gs[b] = excl + gb;
            Bs[a] = a < tc ? beta[(u64)(t0 + a) * vh + h] : 0.0f;
            Bs[b] = b < tc ? beta[(u64)(t0 + b) * vh + h] : 0.0f;
        }
        __syncthreads();
        const float GC = Gs[GDN_C - 1];
        if (tid < GDN_C) {
            Rs[tid] = expf(GC - Gs[tid]);
            As[tid] = expf(Gs[tid]);
        }
        // M and P: thread tile rows 4ty..4ty+3, columns tx + 16b.
        {
            float kk[4][4], qk[4][4];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 4; ++b) kk[a][b] = qk[a][b] = 0.0f;
            for (unsigned i = 0; i < GDN_D; ++i) {
                float kt[4], qt[4], ks[4];
#pragma unroll
                for (unsigned a = 0; a < 4; ++a) {
                    kt[a] = Ks[(4u * ty + a) * GDN_LK + i];
                    qt[a] = Qs[(4u * ty + a) * GDN_LK + i];
                }
#pragma unroll
                for (unsigned b = 0; b < 4; ++b) ks[b] = Ks[(tx + 16u * b) * GDN_LK + i];
#pragma unroll
                for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                    for (unsigned b = 0; b < 4; ++b) {
                        kk[a][b] = fmaf(kt[a], ks[b], kk[a][b]);
                        qk[a][b] = fmaf(qt[a], ks[b], qk[a][b]);
                    }
            }
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 4; ++b) {
                    const unsigned t = 4u * ty + a, s2 = tx + 16u * b;
                    const float dec = s2 <= t ? expf(Gs[t] - Gs[s2]) : 0.0f;
                    Ms[t * GDN_LM + s2] = s2 < t ? Bs[t] * dec * kk[a][b] : 0.0f;
                    Ps[t * GDN_LM + s2] = dec * qk[a][b];
                }
        }
        __syncthreads();
        // Forward substitution, one column per thread, the column in registers.
        if (tid < GDN_CB + GDN_D) {
            const bool isu = tid < GDN_CB;
            const unsigned c = isu ? tid : tid - GDN_CB;
            float X[GDN_C];
#pragma unroll
            for (unsigned t = 0; t < GDN_C; ++t) {
                float v = isu ? Bs[t] * Vs[t * GDN_LU + c] : Bs[t] * As[t] * Ks[t * GDN_LK + c];
                const float* mr = Ms + t * GDN_LM;
#pragma unroll
                for (unsigned s2 = 0; s2 < t; ++s2) v = fmaf(-mr[s2], X[s2], v);
                X[t] = v;
            }
            float* dst = isu ? Us + c : Ws + c;
            const unsigned ld = isu ? GDN_LU : GDN_LK;
#pragma unroll
            for (unsigned t = 0; t < GDN_C; ++t) dst[t * ld] = X[t];
        }
        __syncthreads();
        // D = U - W S0: thread tile rows 4ty.., columns 4tx..4tx+3.
        {
            float acc[4][4];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.0f;
            for (unsigned i = 0; i < GDN_D; ++i) {
                const float4 s4 = *reinterpret_cast<const float4*>(S + i * GDN_CB + 4u * tx);
#pragma unroll
                for (unsigned a = 0; a < 4; ++a) {
                    const float w = Ws[(4u * ty + a) * GDN_LK + i];
                    acc[a][0] = fmaf(w, s4.x, acc[a][0]); acc[a][1] = fmaf(w, s4.y, acc[a][1]);
                    acc[a][2] = fmaf(w, s4.z, acc[a][2]); acc[a][3] = fmaf(w, s4.w, acc[a][3]);
                }
            }
#pragma unroll
            for (unsigned a = 0; a < 4; ++a) {
                float4* u = reinterpret_cast<float4*>(Us + (4u * ty + a) * GDN_LU + 4u * tx);
                float4 v = *u;
                v.x -= acc[a][0]; v.y -= acc[a][1]; v.z -= acc[a][2]; v.w -= acc[a][3];
                *u = v;
            }
        }
        __syncthreads();
        // O = diag(A) Q S0 + (P o L) D.
        {
            float qa[4][4], pd[4][4];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 4; ++b) qa[a][b] = pd[a][b] = 0.0f;
            for (unsigned i = 0; i < GDN_D; ++i) {
                const float4 s4 = *reinterpret_cast<const float4*>(S + i * GDN_CB + 4u * tx);
#pragma unroll
                for (unsigned a = 0; a < 4; ++a) {
                    const float q = Qs[(4u * ty + a) * GDN_LK + i];
                    qa[a][0] = fmaf(q, s4.x, qa[a][0]); qa[a][1] = fmaf(q, s4.y, qa[a][1]);
                    qa[a][2] = fmaf(q, s4.z, qa[a][2]); qa[a][3] = fmaf(q, s4.w, qa[a][3]);
                }
            }
            for (unsigned s2 = 0; s2 < GDN_C; ++s2) {
                const float4 d4 = *reinterpret_cast<const float4*>(Us + s2 * GDN_LU + 4u * tx);
#pragma unroll
                for (unsigned a = 0; a < 4; ++a) {
                    const float pv = Ps[(4u * ty + a) * GDN_LM + s2];
                    pd[a][0] = fmaf(pv, d4.x, pd[a][0]); pd[a][1] = fmaf(pv, d4.y, pd[a][1]);
                    pd[a][2] = fmaf(pv, d4.z, pd[a][2]); pd[a][3] = fmaf(pv, d4.w, pd[a][3]);
                }
            }
#pragma unroll
            for (unsigned a = 0; a < 4; ++a) {
                const unsigned t = 4u * ty + a;
                if (t < tc) {
                    float* o = out + (u64)(t0 + t) * vh * GDN_D + (u64)h * GDN_D + c0 + 4u * tx;
#pragma unroll
                    for (unsigned b = 0; b < 4; ++b) o[b] = fmaf(As[t], qa[a][b], pd[a][b]);
                }
            }
        }
        __syncthreads();
        // S = A_C S0 + sum_s e^{G_C - G_s} k_s D_s^T: thread tile rows 8ty..8ty+7, columns 4tx...
        {
            const float AC = expf(GC);
            float acc[8][4];
#pragma unroll
            for (unsigned a = 0; a < 8; ++a) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.0f;
            for (unsigned s2 = 0; s2 < GDN_C; ++s2) {
                const float r = Rs[s2];
                const float4 d4 = *reinterpret_cast<const float4*>(Us + s2 * GDN_LU + 4u * tx);
                const float4 k0 = *reinterpret_cast<const float4*>(Ks + s2 * GDN_LK + 8u * ty);
                const float4 k1 = *reinterpret_cast<const float4*>(Ks + s2 * GDN_LK + 8u * ty + 4u);
                const float kr[8] = {k0.x * r, k0.y * r, k0.z * r, k0.w * r, k1.x * r, k1.y * r, k1.z * r, k1.w * r};
#pragma unroll
                for (unsigned a = 0; a < 8; ++a) {
                    acc[a][0] = fmaf(kr[a], d4.x, acc[a][0]); acc[a][1] = fmaf(kr[a], d4.y, acc[a][1]);
                    acc[a][2] = fmaf(kr[a], d4.z, acc[a][2]); acc[a][3] = fmaf(kr[a], d4.w, acc[a][3]);
                }
            }
#pragma unroll
            for (unsigned a = 0; a < 8; ++a) {
                float4* sp = reinterpret_cast<float4*>(S + (8u * ty + a) * GDN_CB + 4u * tx);
                float4 v = *sp;
                v.x = fmaf(AC, v.x, acc[a][0]); v.y = fmaf(AC, v.y, acc[a][1]);
                v.z = fmaf(AC, v.z, acc[a][2]); v.w = fmaf(AC, v.w, acc[a][3]);
                *sp = v;
            }
        }
    }
    __syncthreads();
    for (unsigned x = tid; x < GDN_D * GDN_CB; x += blockDim.x) st[(u64)(x / GDN_CB) * GDN_D + c0 + x % GDN_CB] = S[x];
}

// Gated RMSNorm per value head: y = rms_norm(x_h) * w * silu(z_h), quantised (int8 + f16) per 32.
// x, z: [t][vh][dv]; grid = (vh, T), dv threads (128 = 4 warps).
extern "C" __global__ void astral_qw_gated_norm_q8(const float* __restrict__ x, const float* __restrict__ z, unsigned z_stride,
                                                   const float* __restrict__ w,
                                                   signed char* __restrict__ xq, float* __restrict__ xd, __half* __restrict__ xh,
                                                   unsigned vh, unsigned dv, float eps) { PDL_WAIT();
    const unsigned h = blockIdx.x, t = blockIdx.y, i = threadIdx.x, lane = i & 31u;
    const u64 base = ((u64)t * vh + h) * dv;
    const float v = x[base + i];
    const float inv = rsqrtf(block_sum(v * v) / (float)dv + eps);
    const float y = v * inv * w[i] * silu(z[(u64)t * z_stride + (u64)h * dv + i]);
    q8_store(y, xq, xd, xh, (base + i) >> 5, lane);
}

// ---------------------------------------------------------------------------------------------
// Full-attention prep for T tokens at positions pos[0] + t. qg: [t][heads][2 * hd] (query then
// gate per head). One block of hd threads per job:
//   jobs [0, heads):        q = rope(rms(q) * qn); gate copied out
//   jobs [heads, +kvh):     k = rope(rms(k) * kn) -> f16 cache
//   jobs [heads+kvh, +kvh): v -> f16 cache
// RoPE: NeoX style on the first `rot` dims (pairs i, i + rot/2), inv_freq_i = theta^(-2i/rot).
extern "C" __global__ void astral_qw_attn_prep(const float* __restrict__ qg, const float* __restrict__ k, const float* __restrict__ v,
                                               unsigned row_stride,
                                               const float* __restrict__ qn, const float* __restrict__ kn, float* __restrict__ q_out,
                                               float* __restrict__ gate_out, __half* __restrict__ kc, __half* __restrict__ vc,
                                               const int* __restrict__ pos_p, unsigned heads, unsigned kvh, unsigned hd,
                                               unsigned rot, float theta, float eps, unsigned capacity) { PDL_WAIT();
    const unsigned job = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const unsigned pos = min((unsigned)pos_p[0] + t, capacity - 1u);
    __shared__ float xs[512];
    const bool is_v = job >= heads + kvh;
    if (is_v) {
        const unsigned h = job - heads - kvh;
        vc[(u64)pos * kvh * hd + (u64)h * hd + i] = __float2half_rn(v[(u64)t * row_stride + (u64)h * hd + i]);
        return;
    }
    const bool is_k = job >= heads;
    const unsigned h = is_k ? job - heads : job;
    const float x = is_k ? k[(u64)t * row_stride + (u64)h * hd + i] : qg[(u64)t * row_stride + (u64)h * 2u * hd + i];
    if (!is_k) gate_out[((u64)t * heads + h) * hd + i] = qg[(u64)t * row_stride + (u64)h * 2u * hd + hd + i];
    const float inv = rsqrtf(block_sum(x * x) / (float)hd + eps);
    const float y = x * inv * (is_k ? kn[i] : qn[i]);
    xs[i] = y;
    __syncthreads();
    float r = y;
    if (i < rot) {
        const unsigned half = rot / 2u, f = i % half;
        const float inv_freq = powf(theta, -2.0f * (float)f / (float)rot);
        float sn, cs;
        sincosf((float)pos * inv_freq, &sn, &cs);
        r = i < half ? y * cs - xs[i + half] * sn : y * cs + xs[i - half] * sn;
    }
    if (is_k) kc[(u64)pos * kvh * hd + (u64)h * hd + i] = __float2half_rn(r);
    else q_out[((u64)t * heads + h) * hd + i] = r;
}

// Split-K causal attention, head_dim 256, no sink: grid (heads, max_splits, T), 4 warps, one
// warp per key (lane = 8 dims), online softmax per warp, merged per block into part[t][h][split]
// = {m, l, acc[256]}; astral_qw_attn_merge combines the splits.
#define QA_HD 256
#define QA_CHUNK 64
extern "C" __global__ void __launch_bounds__(128) astral_qw_attn_split(const float* __restrict__ q, const __half* __restrict__ kc,
                                                                       const __half* __restrict__ vc, float* __restrict__ part,
                                                                       const int* __restrict__ pos_p, unsigned heads, unsigned kvh,
                                                                       float scale, unsigned capacity, unsigned max_splits) { PDL_WAIT();
    const unsigned h = blockIdx.x, sp = blockIdx.y, z = blockIdx.z;
    const unsigned g = h / (heads / kvh);
    const unsigned pos = min((unsigned)pos_p[0] + z, capacity - 1u);
    const unsigned start = sp * QA_CHUNK;
    if (start > pos) return;
    const unsigned end = min(pos + 1, start + QA_CHUNK);
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned width = kvh * QA_HD;
    const float* qh = q + ((u64)z * heads + h) * QA_HD + lane * 8u;
    const float4 q0 = reinterpret_cast<const float4*>(qh)[0], q1 = reinterpret_cast<const float4*>(qh)[1];
    float m = -INFINITY, l = 0.0f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    // 4 keys per iteration as independent chains, one online-softmax update per group.
    constexpr unsigned GK = 4;
    for (unsigned j0 = start + warp; j0 < end; j0 += 4u * GK) {
        float sc[GK];
        uint4 vr[GK];
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            const unsigned j = j0 + 4u * u;
            const unsigned jj = j < end ? j : j0;
            const uint4 kr = *reinterpret_cast<const uint4*>(kc + (u64)jj * width + (u64)g * QA_HD + lane * 8u);
            vr[u] = *reinterpret_cast<const uint4*>(vc + (u64)jj * width + (u64)g * QA_HD + lane * 8u);
            const __half2* kh2 = reinterpret_cast<const __half2*>(&kr);
            const float2 k0 = __half22float2(kh2[0]), k1 = __half22float2(kh2[1]), k2 = __half22float2(kh2[2]), k3 = __half22float2(kh2[3]);
            sc[u] = q0.x * k0.x + q0.y * k0.y + q0.z * k1.x + q0.w * k1.y + q1.x * k2.x + q1.y * k2.y + q1.z * k3.x + q1.w * k3.y;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
            for (unsigned u = 0; u < GK; ++u) sc[u] += __shfl_xor_sync(0xffffffffu, sc[u], o);
        }
        float mx = m;
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            sc[u] = (j0 + 4u * u < end) ? sc[u] * scale : -INFINITY;
            mx = fmaxf(mx, sc[u]);
        }
        const float c = expf(m - mx);
        l *= c;
#pragma unroll
        for (unsigned d = 0; d < 8; ++d) acc[d] *= c;
#pragma unroll
        for (unsigned u = 0; u < GK; ++u) {
            const float p = expf(sc[u] - mx);
            l += p;
            const __half2* vh2 = reinterpret_cast<const __half2*>(&vr[u]);
#pragma unroll
            for (unsigned d = 0; d < 4; ++d) {
                const float2 vv = __half22float2(vh2[d]);
                acc[2 * d] = fmaf(p, vv.x, acc[2 * d]);
                acc[2 * d + 1] = fmaf(p, vv.y, acc[2 * d + 1]);
            }
        }
        m = mx;
    }
    __shared__ float wm[4], wl[4], wacc[4][QA_HD];
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
#pragma unroll
    for (unsigned d = 0; d < 8; ++d) wacc[warp][lane * 8u + d] = acc[d];
    __syncthreads();
    const float M = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
    float* out = part + (((u64)z * heads + h) * max_splits + sp) * (QA_HD + 2);
    for (unsigned d = threadIdx.x; d < QA_HD; d += blockDim.x) {
        float a = 0.0f;
#pragma unroll
        for (unsigned w = 0; w < 4; ++w) a = fmaf(wacc[w][d], wm[w] == -INFINITY ? 0.0f : expf(wm[w] - M), a);
        out[2 + d] = a;
    }
    if (threadIdx.x == 0) {
        float L = 0.0f;
        for (unsigned w = 0; w < 4; ++w) L = fmaf(wl[w], wm[w] == -INFINITY ? 0.0f : expf(wm[w] - M), L);
        out[0] = M;
        out[1] = L;
    }
}
extern "C" __global__ void astral_qw_attn_merge(const float* __restrict__ part, float* __restrict__ out, const int* __restrict__ pos_p,
                                                unsigned capacity, unsigned max_splits) { PDL_WAIT();
    const unsigned h = blockIdx.x, z = blockIdx.y, heads = gridDim.x;
    const unsigned pos = min((unsigned)pos_p[0] + z, capacity - 1u);
    const unsigned n_splits = min((pos + QA_CHUNK) / QA_CHUNK, max_splits);
    const float* p = part + ((u64)z * heads + h) * max_splits * (QA_HD + 2);
    float M = -INFINITY;
    for (unsigned s = 0; s < n_splits; ++s) M = fmaxf(M, p[(u64)s * (QA_HD + 2)]);
    for (unsigned d = threadIdx.x; d < QA_HD; d += blockDim.x) {
        float L = 0.0f, acc = 0.0f;
        for (unsigned s = 0; s < n_splits; ++s) {
            const float* ps = p + (u64)s * (QA_HD + 2);
            const float f = expf(ps[0] - M);
            L = fmaf(ps[1], f, L);
            acc = fmaf(ps[2 + d], f, acc);
        }
        out[((u64)z * heads + h) * QA_HD + d] = acc / L;
    }
}

// ---------------------------------------------------------------------------------------------
// SwiGLU: y = silu(g) * u for T rows of n, quantised (int8 + f16) per 32. One warp per block of 32.
extern "C" __global__ void __launch_bounds__(256) astral_qw_swiglu_q8(const float* __restrict__ g, const float* __restrict__ u,
                                                                      unsigned n, unsigned stride, signed char* __restrict__ xq,
                                                                      float* __restrict__ xd, __half* __restrict__ xh, u64 total) { PDL_WAIT();
    // grid = (ceil(n / 256), T): one warp per 32-value block of row t (n % 32 == 0); g and u
    // rows are `stride` apart. (`total` = T * n, kept for the launcher's checks.)
    (void)total;
    const unsigned t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31u;
    if (c >= n) return;
    const u64 r = (u64)t * stride + c;
    const u64 i = (u64)t * n + c;
    q8_store(silu(g[r]) * u[r], xq, xd, xh, i >> 5, lane);
}

// x += y
extern "C" __global__ void astral_qw_add(float* __restrict__ x, const float* __restrict__ y, u64 n) { PDL_WAIT();
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) x[i] += y[i];
}

// =============================================================================================
// Flash-attention prefill for head_dim 256, causal, no sink, no bias (tensor cores, sm_80+).
// grid = (ceil(T / 64), heads, 2): 4 warps x 16 queries of one head; blockIdx.z = which half of
// the 256 value dims this block accumulates (QK^T is computed in full by both halves, keeping
// the output accumulator at 64 registers). Keys in tiles of 64 staged in dynamic shared memory
// (K [64][264] f16, V half [64][136] f16). Q rounded to f16 (as flash attention does).
#define QF_Q 64
#define QF_K 64
#define QF_KROW 264
#define QF_VROW 136
__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
    const unsigned sp = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sp), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void mma_f16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

extern "C" __global__ void __launch_bounds__(128) astral_qw_attn_prefill(const float* __restrict__ q, const __half* __restrict__ kc,
                                                                        const __half* __restrict__ vc, float* __restrict__ out,
                                                                        const int* __restrict__ pos_p, unsigned ntok, unsigned heads,
                                                                        unsigned kvh, float scale, unsigned capacity) { PDL_WAIT();
    constexpr unsigned HD = 256, HV = 128;
    extern __shared__ __align__(16) __half fsm[];
    __half* Ks = fsm;                       // [64][264]
    __half* Vs = fsm + QF_K * QF_KROW;      // [64][136]
    const unsigned h = blockIdx.y, qt = blockIdx.x, vh0 = blockIdx.z * HV;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    const unsigned kvhh = h / (heads / kvh), width = kvh * HD;
    const unsigned pos0 = (unsigned)pos_p[0];
    const unsigned ta = qt * QF_Q + warp * 16u + g, tb = ta + 8u;
    const bool oka = ta < ntok, okb = tb < ntok;
    const unsigned la = min(ta, ntok - 1u), lb = min(tb, ntok - 1u);
    const unsigned pa = min(pos0 + la, capacity - 1u), pb = min(pos0 + lb, capacity - 1u);
    unsigned qa[16][4];
    {
        const float* qra = q + ((u64)la * heads + h) * HD;
        const float* qrb = q + ((u64)lb * heads + h) * HD;
#pragma unroll
        for (unsigned ks = 0; ks < 16; ++ks) {
            const unsigned c = ks * 16u + 2u * t;
            qa[ks][0] = pack_h2(qra[c], qra[c + 1]);
            qa[ks][1] = pack_h2(qrb[c], qrb[c + 1]);
            qa[ks][2] = pack_h2(qra[c + 8], qra[c + 9]);
            qa[ks][3] = pack_h2(qrb[c + 8], qrb[c + 9]);
        }
    }
    const unsigned last = min(pos0 + min(qt * QF_Q + QF_Q - 1u, ntok - 1u), capacity - 1u);
    float ma = -1e30f, mb = -1e30f, sa = 0.0f, sb = 0.0f;
    float o[16][4];
#pragma unroll
    for (unsigned i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    for (unsigned kb = 0; kb <= last; kb += QF_K) {
        __syncthreads();
        for (unsigned c = tid; c < QF_K * 32u; c += blockDim.x) {           // K: 32 chunks of 16 B per row
            const unsigned row = c >> 5, part = c & 31u;
            const unsigned kp = min(kb + row, last);
            cp_async16(Ks + row * QF_KROW + part * 8u, kc + (u64)kp * width + (u64)kvhh * HD + part * 8u);
        }
        for (unsigned c = tid; c < QF_K * 16u; c += blockDim.x) {           // V half: 16 chunks per row
            const unsigned row = c >> 4, part = c & 15u;
            const unsigned kp = min(kb + row, last);
            cp_async16(Vs + row * QF_VROW + part * 8u, vc + (u64)kp * width + (u64)kvhh * HD + vh0 + part * 8u);
        }
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            const __half* kr = Ks + (nt * 8u + g) * QF_KROW + 2u * t;
#pragma unroll
            for (unsigned ks = 0; ks < 16; ++ks)
                mma_f16(s[nt], qa[ks], *reinterpret_cast<const unsigned*>(kr + ks * 16u), *reinterpret_cast<const unsigned*>(kr + ks * 16u + 8u));
        }
        float tma = -INFINITY, tmb = -INFINITY;
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt)
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const bool rowb = i >= 2;
                const unsigned key = kb + nt * 8u + 2u * t + (i & 1u);
                const bool valid = (rowb ? okb : oka) && key <= (rowb ? pb : pa);
                const float l = valid ? s[nt][i] * scale : -INFINITY;
                s[nt][i] = l;
                if (rowb) tmb = fmaxf(tmb, l); else tma = fmaxf(tma, l);
            }
        tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, 1));
        tma = fmaxf(tma, __shfl_xor_sync(0xffffffffu, tma, 2));
        tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, 1));
        tmb = fmaxf(tmb, __shfl_xor_sync(0xffffffffu, tmb, 2));
        const float na = fmaxf(ma, tma), nb = fmaxf(mb, tmb);
        const float ca = expf(ma - na), cb = expf(mb - nb);
        ma = na;
        mb = nb;
        float psa = 0.0f, psb = 0.0f;
#pragma unroll
        for (unsigned nt = 0; nt < 8; ++nt) {
            s[nt][0] = expf(s[nt][0] - na); s[nt][1] = expf(s[nt][1] - na);
            s[nt][2] = expf(s[nt][2] - nb); s[nt][3] = expf(s[nt][3] - nb);
            psa += s[nt][0] + s[nt][1];
            psb += s[nt][2] + s[nt][3];
        }
        psa += __shfl_xor_sync(0xffffffffu, psa, 1);
        psa += __shfl_xor_sync(0xffffffffu, psa, 2);
        psb += __shfl_xor_sync(0xffffffffu, psb, 1);
        psb += __shfl_xor_sync(0xffffffffu, psb, 2);
        sa = sa * ca + psa;
        sb = sb * cb + psb;
#pragma unroll
        for (unsigned dn = 0; dn < 16; ++dn) { o[dn][0] *= ca; o[dn][1] *= ca; o[dn][2] *= cb; o[dn][3] *= cb; }
#pragma unroll
        for (unsigned kk = 0; kk < 4; ++kk) {
            const unsigned pa_[4] = {pack_h2(s[2 * kk][0], s[2 * kk][1]), pack_h2(s[2 * kk][2], s[2 * kk][3]),
                                     pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (unsigned dn = 0; dn < 16; ++dn) {
                unsigned b0, b1;
                const __half* vp = Vs + (kk * 16u + (lane & 15u)) * QF_VROW + dn * 8u;
                const unsigned sp = (unsigned)__cvta_generic_to_shared(vp);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(sp));
                mma_f16(o[dn], pa_, b0, b1);
            }
        }
    }
    const float ia = 1.0f / sa, ib = 1.0f / sb;
#pragma unroll
    for (unsigned dn = 0; dn < 16; ++dn) {
        const unsigned c = vh0 + dn * 8u + 2u * t;
        if (oka) {
            float* orow = out + ((u64)ta * heads + h) * HD;
            orow[c] = o[dn][0] * ia;
            orow[c + 1] = o[dn][1] * ia;
        }
        if (okb) {
            float* orow = out + ((u64)tb * heads + h) * HD;
            orow[c] = o[dn][2] * ib;
            orow[c + 1] = o[dn][3] * ib;
        }
    }
}

// x += add (written back), then y = rms_norm(x) * w (y optional) quantised (int8 + f16) per 32,
// for T rows of dim (one block per row): the residual add fused into the following norm.
extern "C" __global__ void __launch_bounds__(1024) astral_qw_add_rms_q8(float* __restrict__ x, const float* __restrict__ add,
                                                                        const float* __restrict__ w, float* __restrict__ y,
                                                                        signed char* __restrict__ xq, float* __restrict__ xd,
                                                                        __half* __restrict__ xh, unsigned dim, float eps) { PDL_WAIT();
    const unsigned t = blockIdx.x;
    x += (u64)t * dim;
    add += (u64)t * dim;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = x[i] + add[i];
        x[i] = v;
        ss = fmaf(v, v, ss);
    }
    const float inv = rsqrtf(block_sum(ss) / (float)dim + eps);   // block_sum syncs: x is complete
    const unsigned lane = threadIdx.x & 31u, nw = blockDim.x >> 5;
    for (unsigned b = threadIdx.x >> 5; b < (dim >> 5); b += nw) {
        const unsigned i = b * 32u + lane;
        const float v = x[i] * inv * w[i];
        if (y) y[(u64)t * dim + i] = v;
        q8_store(v, xq + (u64)t * dim, xd + (u64)t * (dim >> 5), xh ? xh + (u64)t * dim : nullptr, b, lane);
    }
}

// =============================================================================================
// Two-phase chunked gated delta rule (prefill). Same maths as astral_qw_gdn_chunk, split by
// what depends on the carried state:
//   prep (parallel over head x chunk x column half): G, M, P, U = (I+M)^-1 beta V and
//     W = (I+M)^-1 beta A K (both independent of S) -> scratch.
//   scan (per head x 32-column block, sequential over chunks): D = U - W S, O, S update.
// Scratch per (head, chunk): W [64][128], U [64][128], P [64][64], G [64].
#define G2_CH 64
extern "C" __global__ void __launch_bounds__(256) astral_qw_gdn_prep2(const float* __restrict__ qkv, unsigned qkv_stride,
                                                                     const float* __restrict__ beta, const float* __restrict__ g,
                                                                     float* __restrict__ wsc, float* __restrict__ usc, float* __restrict__ psc,
                                                                     float* __restrict__ gsc, unsigned ntok, unsigned kh, unsigned vh) { PDL_WAIT();
    extern __shared__ __align__(16) float sm2[];
    float* Ks = sm2;                         // [64][132]
    float* Qs = Ks + G2_CH * GDN_LK;
    float* Vs = Qs + G2_CH * GDN_LK;         // [64][68]  (this half's value columns)
    float* Ms = Vs + G2_CH * GDN_LU;         // [64][65]
    float* Gs = Ms + G2_CH * GDN_LM;
    float* Bs = Gs + G2_CH;
    float* As = Bs + G2_CH;
    const unsigned h = blockIdx.x, ch = blockIdx.y, half = blockIdx.z, tid = threadIdx.x;
    const unsigned nch = gridDim.y;
    const unsigned t0 = ch * G2_CH;
    if (t0 >= ntok) return;
    const unsigned tc = min((unsigned)G2_CH, ntok - t0);
    const unsigned hk = h / (vh / kh);
    const unsigned ty = tid >> 4, tx = tid & 15u;
    const u64 cbase = (u64)h * nch + ch;
    for (unsigned x = tid; x < G2_CH * GDN_D; x += blockDim.x) {
        const unsigned t = x / GDN_D, i = x % GDN_D;
        float qv = 0.0f, kv = 0.0f;
        if (t < tc) {
            const float* row = qkv + (u64)(t0 + t) * qkv_stride;
            qv = row[(u64)hk * GDN_D + i];
            kv = row[(u64)kh * GDN_D + (u64)hk * GDN_D + i];
        }
        Qs[t * GDN_LK + i] = qv;
        Ks[t * GDN_LK + i] = kv;
    }
    for (unsigned x = tid; x < G2_CH * 64u; x += blockDim.x) {
        const unsigned t = x / 64u, c = x % 64u;
        Vs[t * GDN_LU + c] = t < tc ? qkv[(u64)(t0 + t) * qkv_stride + 2u * (u64)kh * GDN_D + (u64)h * GDN_D + half * 64u + c] : 0.0f;
    }
    if (tid < 32) {
        const unsigned a = 2u * tid, b = a + 1u;
        const float ga = a < tc ? g[(u64)(t0 + a) * vh + h] : 0.0f, gb0 = b < tc ? g[(u64)(t0 + b) * vh + h] : 0.0f;
        const float gb = ga + gb0;
        float sum = gb;
        for (int o = 1; o < 32; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, sum, o);
            if ((int)tid >= o) sum += n;
        }
        const float excl = sum - gb;
        Gs[a] = excl + ga;
        Gs[b] = excl + gb;
        Bs[a] = a < tc ? beta[(u64)(t0 + a) * vh + h] : 0.0f;
        Bs[b] = b < tc ? beta[(u64)(t0 + b) * vh + h] : 0.0f;
    }
    __syncthreads();
    if (tid < G2_CH) {
        As[tid] = expf(Gs[tid]);
        if (half == 0) gsc[cbase * G2_CH + tid] = Gs[tid];
    }
    {
        float kk[4][4], qk[4][4];
#pragma unroll
        for (unsigned a = 0; a < 4; ++a)
#pragma unroll
            for (unsigned b = 0; b < 4; ++b) kk[a][b] = qk[a][b] = 0.0f;
        for (unsigned i = 0; i < GDN_D; ++i) {
            float kt[4], qt[4], ks[4];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a) { kt[a] = Ks[(4u * ty + a) * GDN_LK + i]; qt[a] = Qs[(4u * ty + a) * GDN_LK + i]; }
#pragma unroll
            for (unsigned b = 0; b < 4; ++b) ks[b] = Ks[(tx + 16u * b) * GDN_LK + i];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a)
#pragma unroll
                for (unsigned b = 0; b < 4; ++b) { kk[a][b] = fmaf(kt[a], ks[b], kk[a][b]); qk[a][b] = fmaf(qt[a], ks[b], qk[a][b]); }
        }
#pragma unroll
        for (unsigned a = 0; a < 4; ++a)
#pragma unroll
            for (unsigned b = 0; b < 4; ++b) {
                const unsigned t = 4u * ty + a, s2 = tx + 16u * b;
                const float dec = s2 <= t ? expf(Gs[t] - Gs[s2]) : 0.0f;
                Ms[t * GDN_LM + s2] = s2 < t ? Bs[t] * dec * kk[a][b] : 0.0f;
                if (half == 0) psc[(cbase * G2_CH + t) * G2_CH + s2] = dec * qk[a][b];
            }
    }
    __syncthreads();
    // Forward substitution: half 0 solves W (128 columns) and U cols 0..63; half 1 U cols 64..127.
    const unsigned ncols = half == 0 ? 192u : 64u;
    if (tid < ncols) {
        const bool isu = tid < 64u;
        const unsigned c = isu ? tid : tid - 64u;
        float X[G2_CH];
#pragma unroll
        for (unsigned t = 0; t < G2_CH; ++t) {
            float v = isu ? Bs[t] * Vs[t * GDN_LU + c] : Bs[t] * As[t] * Ks[t * GDN_LK + c];
            const float* mr = Ms + t * GDN_LM;
#pragma unroll
            for (unsigned s2 = 0; s2 < t; ++s2) v = fmaf(-mr[s2], X[s2], v);
            X[t] = v;
        }
        if (isu) {
            float* dst = usc + cbase * G2_CH * GDN_D + half * 64u + c;
#pragma unroll
            for (unsigned t = 0; t < G2_CH; ++t) dst[(u64)t * GDN_D] = X[t];
        } else {
            float* dst = wsc + cbase * G2_CH * GDN_D + c;
#pragma unroll
            for (unsigned t = 0; t < G2_CH; ++t) dst[(u64)t * GDN_D] = X[t];
        }
    }
}

// Scan: grid (vh, 128 / 32), 256 threads; per chunk D = U - W S, O = A Q S + P D, S update.
#define G2_CB 32
extern "C" __global__ void __launch_bounds__(256) astral_qw_gdn_scan2(const float* __restrict__ qkv, unsigned qkv_stride,
                                                                     const float* __restrict__ wsc, const float* __restrict__ usc,
                                                                     const float* __restrict__ psc, const float* __restrict__ gsc,
                                                                     float* __restrict__ state, float* __restrict__ out, unsigned ntok,
                                                                     unsigned kh, unsigned vh, unsigned nch) { PDL_WAIT();
    extern __shared__ __align__(16) float sm3[];
    float* S = sm3;                          // [128][32]
    float* Ks = S + GDN_D * G2_CB;           // [64][132]
    float* Qs = Ks + G2_CH * GDN_LK;
    float* Ws = Qs + G2_CH * GDN_LK;
    float* Us = Ws + G2_CH * GDN_LK;         // [64][36] (U, then D)
    float* Ps = Us + G2_CH * 36;             // [64][65]
    float* Gs = Ps + G2_CH * GDN_LM;
    const unsigned h = blockIdx.x, c0 = blockIdx.y * G2_CB, tid = threadIdx.x;
    const unsigned hk = h / (vh / kh);
    float* st = state + (u64)h * GDN_D * GDN_D;
    for (unsigned x = tid; x < GDN_D * G2_CB; x += blockDim.x) S[x] = st[(u64)(x / G2_CB) * GDN_D + c0 + x % G2_CB];
    // Thread tiles: D / O rows 4 * (tid / 8) .., cols 4 * (tid % 8) ..; S rows 4 * (tid / 8) .. (+64), cols 4 * (tid % 8).
    const unsigned ry = tid >> 3, cx = (tid & 7u) * 4u;
    for (unsigned ch = 0; ch < nch; ++ch) {
        const unsigned t0 = ch * G2_CH;
        const unsigned tc = min((unsigned)G2_CH, ntok - t0);
        const u64 cbase = (u64)h * nch + ch;
        __syncthreads();
        for (unsigned x = tid; x < G2_CH * GDN_D; x += blockDim.x) {
            const unsigned t = x / GDN_D, i = x % GDN_D;
            float qv = 0.0f, kv = 0.0f;
            if (t < tc) {
                const float* row = qkv + (u64)(t0 + t) * qkv_stride;
                qv = row[(u64)hk * GDN_D + i];
                kv = row[(u64)kh * GDN_D + (u64)hk * GDN_D + i];
            }
            Qs[t * GDN_LK + i] = qv;
            Ks[t * GDN_LK + i] = kv;
            Ws[t * GDN_LK + i] = wsc[(cbase * G2_CH + t) * GDN_D + i];
        }
        for (unsigned x = tid; x < G2_CH * G2_CB; x += blockDim.x) {
            const unsigned t = x / G2_CB, c = x % G2_CB;
            Us[t * 36 + c] = usc[(cbase * G2_CH + t) * GDN_D + c0 + c];
        }
        for (unsigned x = tid; x < G2_CH * G2_CH; x += blockDim.x) Ps[(x / G2_CH) * GDN_LM + x % G2_CH] = psc[cbase * G2_CH * G2_CH + x];
        if (tid < G2_CH) Gs[tid] = gsc[cbase * G2_CH + tid];
        __syncthreads();
        const float GC = Gs[G2_CH - 1], AC = expf(GC);
        // D = U - W S
        {
            float acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            for (unsigned i = 0; i < GDN_D; ++i) {
                const float4 s4 = *reinterpret_cast<const float4*>(S + i * G2_CB + cx);
#pragma unroll
                for (unsigned a = 0; a < 2; ++a) {
                    const float w = Ws[(2u * ry + a) * GDN_LK + i];
                    acc[a][0] = fmaf(w, s4.x, acc[a][0]); acc[a][1] = fmaf(w, s4.y, acc[a][1]);
                    acc[a][2] = fmaf(w, s4.z, acc[a][2]); acc[a][3] = fmaf(w, s4.w, acc[a][3]);
                }
            }
#pragma unroll
            for (unsigned a = 0; a < 2; ++a) {
                float* u = Us + (2u * ry + a) * 36 + cx;
                u[0] -= acc[a][0]; u[1] -= acc[a][1]; u[2] -= acc[a][2]; u[3] -= acc[a][3];
            }
        }
        __syncthreads();
        // O = diag(A) Q S + (P o L) D
        {
            float qa[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}}, pd[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            for (unsigned i = 0; i < GDN_D; ++i) {
                const float4 s4 = *reinterpret_cast<const float4*>(S + i * G2_CB + cx);
#pragma unroll
                for (unsigned a = 0; a < 2; ++a) {
                    const float q = Qs[(2u * ry + a) * GDN_LK + i];
                    qa[a][0] = fmaf(q, s4.x, qa[a][0]); qa[a][1] = fmaf(q, s4.y, qa[a][1]);
                    qa[a][2] = fmaf(q, s4.z, qa[a][2]); qa[a][3] = fmaf(q, s4.w, qa[a][3]);
                }
            }
            for (unsigned s2 = 0; s2 < G2_CH; ++s2) {
                const float* dr = Us + s2 * 36 + cx;
                const float d0 = dr[0], d1 = dr[1], d2 = dr[2], d3 = dr[3];
#pragma unroll
                for (unsigned a = 0; a < 2; ++a) {
                    const float pv = Ps[(2u * ry + a) * GDN_LM + s2];
                    pd[a][0] = fmaf(pv, d0, pd[a][0]); pd[a][1] = fmaf(pv, d1, pd[a][1]);
                    pd[a][2] = fmaf(pv, d2, pd[a][2]); pd[a][3] = fmaf(pv, d3, pd[a][3]);
                }
            }
#pragma unroll
            for (unsigned a = 0; a < 2; ++a) {
                const unsigned t = 2u * ry + a;
                if (t < tc) {
                    const float At = expf(Gs[t]);
                    float* o = out + (u64)(t0 + t) * vh * GDN_D + (u64)h * GDN_D + c0 + cx;
#pragma unroll
                    for (unsigned b = 0; b < 4; ++b) o[b] = fmaf(At, qa[a][b], pd[a][b]);
                }
            }
        }
        __syncthreads();
        // S = A_C S + sum_s e^{G_C - G_s} k_s D_s^T  (rows 4ry..4ry+3, cols cx..cx+3)
        {
            float acc[4][4];
#pragma unroll
            for (unsigned a = 0; a < 4; ++a) acc[a][0] = acc[a][1] = acc[a][2] = acc[a][3] = 0.0f;
            for (unsigned s2 = 0; s2 < G2_CH; ++s2) {
                const float r = expf(GC - Gs[s2]);
                const float* dr = Us + s2 * 36 + cx;
                const float d0 = dr[0], d1 = dr[1], d2 = dr[2], d3 = dr[3];
                const float4 k4 = *reinterpret_cast<const float4*>(Ks + s2 * GDN_LK + 4u * ry);
                const float kr[4] = {k4.x * r, k4.y * r, k4.z * r, k4.w * r};
#pragma unroll
                for (unsigned a = 0; a < 4; ++a) {
                    acc[a][0] = fmaf(kr[a], d0, acc[a][0]); acc[a][1] = fmaf(kr[a], d1, acc[a][1]);
                    acc[a][2] = fmaf(kr[a], d2, acc[a][2]); acc[a][3] = fmaf(kr[a], d3, acc[a][3]);
                }
            }
#pragma unroll
            for (unsigned a = 0; a < 4; ++a) {
                float* sp = S + (4u * ry + a) * G2_CB + cx;
                sp[0] = fmaf(AC, sp[0], acc[a][0]); sp[1] = fmaf(AC, sp[1], acc[a][1]);
                sp[2] = fmaf(AC, sp[2], acc[a][2]); sp[3] = fmaf(AC, sp[3], acc[a][3]);
            }
        }
    }
    __syncthreads();
    for (unsigned x = tid; x < GDN_D * G2_CB; x += blockDim.x) st[(u64)(x / G2_CB) * GDN_D + c0 + x % G2_CB] = S[x];
}
