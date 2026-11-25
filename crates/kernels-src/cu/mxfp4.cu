// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//
// MXFP4 weight-only GEMV for decode (ADR-0007 step 1; spec 03 §3 "MMVQ-style").
// Runs natively on any CUDA GPU: weights stay 4-bit in memory and are unpacked in
// registers. No FP4 tensor cores are needed, so this is the H100 path for MXFP4.
//
//   y[b, n] = sum_k W[n, k] * x[b, k]        b < B, B in 1..=4 (one kernel per B), K % 32 == 0
//
// Weight layout (Astral repacked, see astral_quant::repack_mxfp4):
//   qs[n * K/2 + blk*16 + j] : 16 bytes per 32-element block; low nibble -> element j,
//                              high nibble -> element j + 16 (same nibble order as GGUF MXFP4)
//   es[n * K/32 + blk]       : E8M0 block scale
// Result equals ggml's dequantize_row_mxfp4 followed by an f32 dot product (up to f32
// summation order); astral_quant::dequant_mxfp4 is the bit-exact reference.
//
// v1 (2026-11-24): compile-time batch, bit-trick E2M1 -> f16x2 unpack, 128-bit padded
// shared-memory reads, two blocks in flight per lane. Parameter list mirrored by
// GemvMxfp4F32 in crates/kernels/src/ops.rs.

#include <cuda_fp16.h>

// Programmatic dependent launch: on Hopper every kernel may be launched before its predecessor
// finishes (overlapping launch latency); griddepcontrol.wait blocks until the predecessor's
// results are visible, so it is the first thing every kernel does. A no-op otherwise.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if defined(ASTRAL_PDL_EARLY)  // multi-wave GEMV/GEMM: waiting successors would hold SM slots
#define PDL_WAIT() asm volatile("griddepcontrol.wait;\n griddepcontrol.launch_dependents;" ::: "memory")
#else
#define PDL_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#endif
#else
#define PDL_WAIT() do {} while (0)
#endif

typedef unsigned long long u64;
typedef unsigned char u8;

#define TILE_K 2048
// 4 pad floats per 32 => lane stride 144 B: quarter-warp float4 reads hit 32 distinct banks.
#define TILE_PAD (TILE_K + TILE_K / 8)

// E2M1 nibble n (s e1 e0 m) -> f16 bits: s at bit 15, (e1 e0 m) at bits 11..9.
// That f16 equals E2M1(n) * 2^-14 exactly (normal for e >= 1, subnormal 0.5 * 2^-14 for e = 0),
// so the 2^14 is folded into the block scale below.
// Returns f16x2 {lo nibble, hi nibble} of byte b.
__device__ __forceinline__ __half2 e2m1x2_scaled(unsigned b) {
    const unsigned bits = ((b & 0x08u) << 12) | ((b & 0x07u) << 9)     // low nibble -> low half
                        | ((b & 0x80u) << 24) | ((b & 0x70u) << 21);   // high nibble -> high half
    return *reinterpret_cast<const __half2*>(&bits);
}

// Block scale: E8M0 value 2^(e-127), times 2^14 to undo the f16 placement above.
// (Equivalent to ggml's halved scale applied to doubled kvalues.) For every e in 0..=240
// this is an exact normal f32 (exponent e - 113 >= -113). e >= 241 would need a scale of
// 2^128 or more: such blocks (never produced by real quantizers) decode to +inf here,
// whereas ggml returns finite values; the loader rejects them before upload.
__device__ __forceinline__ float block_scale(unsigned e) {
    return e <= 240u ? __uint_as_float((e + 14u) << 23) : __uint_as_float(0x7f800000u);
}

__device__ __forceinline__ unsigned xs_idx(unsigned b, unsigned k) {
    return b * TILE_PAD + k + ((k >> 5) << 2);
}

template <int B>
__device__ __forceinline__ void block_dot(uint4 q4, const float* __restrict__ xs, unsigned kl, float d, float (&acc)[B]) {
    const unsigned w[4] = {q4.x, q4.y, q4.z, q4.w};
    float p[B];
#pragma unroll
    for (int b = 0; b < B; ++b) p[b] = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; ++g) {        // 4 words -> elements 4g..4g+3 and 16+4g..16+4g+3
        float lo[4], hi[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float2 f = __half22float2(e2m1x2_scaled((w[g] >> (8 * t)) & 0xffu));
            lo[t] = f.x;
            hi[t] = f.y;
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const float4 xa = *reinterpret_cast<const float4*>(&xs[xs_idx(b, kl + 4 * g)]);
            const float4 xb = *reinterpret_cast<const float4*>(&xs[xs_idx(b, kl + 16 + 4 * g)]);
            p[b] = fmaf(lo[0], xa.x, p[b]); p[b] = fmaf(lo[1], xa.y, p[b]);
            p[b] = fmaf(lo[2], xa.z, p[b]); p[b] = fmaf(lo[3], xa.w, p[b]);
            p[b] = fmaf(hi[0], xb.x, p[b]); p[b] = fmaf(hi[1], xb.y, p[b]);
            p[b] = fmaf(hi[2], xb.z, p[b]); p[b] = fmaf(hi[3], xb.w, p[b]);
        }
    }
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = fmaf(d, p[b], acc[b]);
}

// One warp per output row; blockDim.x = 32 * rows_per_block.
template <int B>
__device__ __forceinline__ void gemv_mxfp4(const u8* __restrict__ qs, const u8* __restrict__ es,
                                           const float* __restrict__ x, float* __restrict__ y,
                                           unsigned N, unsigned K) {
    __shared__ __align__(16) float xs[B * TILE_PAD];
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned n = blockIdx.x * (blockDim.x >> 5) + warp;
    const unsigned blocks_per_row = K >> 5;

    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.0f;

    for (unsigned k0 = 0; k0 < K; k0 += TILE_K) {
        const unsigned tk = min((unsigned)TILE_K, K - k0);
        __syncthreads();
        // x tile -> shared, 128-bit global loads (K % 32 == 0 keeps rows 16-byte aligned).
        for (unsigned i = threadIdx.x; i < (unsigned)B * (tk >> 2); i += blockDim.x) {
            const unsigned b = i / (tk >> 2), k = (i - b * (tk >> 2)) << 2;
            *reinterpret_cast<float4*>(&xs[xs_idx(b, k)]) = __ldg(reinterpret_cast<const float4*>(x + (u64)b * K + k0 + k));
        }
        __syncthreads();
        if (n < N) {
            const uint4* qrow = reinterpret_cast<const uint4*>(qs + (u64)n * (K >> 1));
            const u8* erow = es + (u64)n * blocks_per_row;
            const unsigned blk_end = (k0 + tk) >> 5;
            unsigned blk = (k0 >> 5) + lane;
            for (; blk + 32u < blk_end; blk += 64u) {       // two blocks in flight per lane
                const uint4 q0 = __ldg(qrow + blk);
                const uint4 q1 = __ldg(qrow + blk + 32u);
                const float d0 = block_scale(erow[blk]);
                const float d1 = block_scale(erow[blk + 32u]);
                block_dot<B>(q0, xs, (blk << 5) - k0, d0, acc);
                block_dot<B>(q1, xs, ((blk + 32u) << 5) - k0, d1, acc);
            }
            if (blk < blk_end) {
                block_dot<B>(__ldg(qrow + blk), xs, (blk << 5) - k0, block_scale(erow[blk]), acc);
            }
        }
    }
    if (n < N) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float v = acc[b];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) y[(u64)b * N + n] = v;
        }
    }
}

#define GEMV_ENTRY(BB)                                                                                 \
    extern "C" __global__ void __launch_bounds__(256) astral_gemv_mxfp4_f32_b##BB(                     \
        const u8* __restrict__ qs, const u8* __restrict__ es, const float* __restrict__ x,             \
        float* __restrict__ y, unsigned N, unsigned K) { PDL_WAIT();                                               \
        gemv_mxfp4<BB>(qs, es, x, y, N, K);                                                            \
    }
GEMV_ENTRY(1)
GEMV_ENTRY(2)
GEMV_ENTRY(3)
GEMV_ENTRY(4)
