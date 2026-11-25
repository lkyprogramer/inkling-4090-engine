// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//
// MXFP4 x FP16 GEMV/skinny-GEMM on tensor cores (ADR-0007 step 1; sm_80 and newer).
//
//   y[b, n] (+)= sum_k W[n, k] * f16(x[b, k])    b < B <= 8, N % 16 == 0, K % 64 == 0
//
// Weights stay 4-bit in memory, in the tensor-core layout of astral_quant::repack_mxfp4_mma:
// 16-row x 64-column tiles, 512 value bytes + 32 scale bytes, one 16-byte chunk per lane.
// Each lane unpacks its E2M1 nibbles to f16 in registers (value * 2^-14, exact) and feeds
// mma.sync.m16n8k16 with fp32 accumulation. MMA rows are weight rows and its 8 columns are
// the batch, so batches 1..8 cost the same tensor-core work. Each 32-element block goes into
// a fresh fp32 fragment that is then scaled by its E8M0 scale (times 2^14) in fp32, so the
// scales are applied exactly. Activations are rounded to f16 when staged (W4A16 contract).
//
// v2.1: a block owns 4 row tiles (64 rows) so activations are staged once per 64 rows, only
// the B real batch rows are staged, each lane keeps 4 weight loads in flight, and
// gridDim.y > 1 splits K across blocks (results combined with fp32 atomicAdd into a zeroed y;
// summation order then varies between runs). Mirrored by GemmMxfp4MmaF32 in ops.rs.
//
// v2.4: f16-activation entry stages with cp.async into two shared buffers (copy of the next
// tile overlaps compute). 2048-column tiles were dropped: double-buffered they exceed the
// 48 KB static shared-memory limit, and they were never the fastest variant.
//
// v2.3: templated on warps per block, weight loads in flight per lane, and activation tile
// size; several instantiations are exported (astral_gemm_mxfp4_mma_f32_w{W}u{U}t{T}) so the
// launcher can pick the best per GPU. ncu on v2.2 showed ~60% of stalls waiting on global
// loads with ~4 active warps per scheduler, so more bytes in flight is the lever.

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
typedef unsigned short u16;


__device__ __forceinline__ unsigned lo_pair(unsigned v) {  // v = byte0 | byte1 << 16
    return ((v & 0x00080008u) << 12) | ((v & 0x00070007u) << 9);
}
__device__ __forceinline__ unsigned hi_pair(unsigned v) {
    return ((v & 0x00800080u) << 8) | ((v & 0x00700070u) << 5);
}

// Measured and rejected (H100 PCIe, 2026-11-24): staging with 4 or 8 loads in flight per
// thread (slower: the registers cost occupancy), and removing the split-K atomics (no change).

// Weight loads: read-only path, streamed once. L2 prefetch (L2::256B) and eviction hints
// were measured on the H100 PCIe (2026-11-24) and changed nothing, so plain __ldg is used.
__device__ __forceinline__ uint4 load_w(const uint4* p) { return __ldg(p); }

__device__ __forceinline__ float block_scale(unsigned e) {  // 2^(e-127) * 2^14, e <= 240
    return e <= 240u ? __uint_as_float((e + 14u) << 23) : __uint_as_float(0x7f800000u);
}

__device__ __forceinline__ void mma16816(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 16x64 weight tile (two MXFP4 blocks) against the staged activations.
__device__ __forceinline__ void tile_pair(float (&acc)[4], uint4 w, unsigned s, const u16* __restrict__ xr,
                                          unsigned kl, unsigned t) {
    const unsigned words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const unsigned wr0 = words[2 * blk], wr8 = words[2 * blk + 1];   // rows g, g+8
        const unsigned r0_01 = __byte_perm(wr0, 0, 0x4140), r0_23 = __byte_perm(wr0, 0, 0x4342);
        const unsigned r8_01 = __byte_perm(wr8, 0, 0x4140), r8_23 = __byte_perm(wr8, 0, 0x4342);
        const unsigned kb = kl + 32u * blk;
        const unsigned bl0 = *reinterpret_cast<const unsigned*>(&xr[kb + 2 * t]);
        const unsigned bl1 = *reinterpret_cast<const unsigned*>(&xr[kb + 2 * t + 8]);
        const unsigned bh0 = *reinterpret_cast<const unsigned*>(&xr[kb + 16 + 2 * t]);
        const unsigned bh1 = *reinterpret_cast<const unsigned*>(&xr[kb + 16 + 2 * t + 8]);
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(c, lo_pair(r0_01), lo_pair(r8_01), lo_pair(r0_23), lo_pair(r8_23), bl0, bl1);
        mma16816(c, hi_pair(r0_01), hi_pair(r8_01), hi_pair(r0_23), hi_pair(r8_23), bh0, bh1);
        const float d0 = block_scale((s >> (16 * blk)) & 0xffu);        // row g
        const float d8 = block_scale((s >> (16 * blk + 8)) & 0xffu);    // row g + 8
        acc[0] = fmaf(d0, c[0], acc[0]);
        acc[1] = fmaf(d0, c[1], acc[1]);
        acc[2] = fmaf(d8, c[2], acc[2]);
        acc[3] = fmaf(d8, c[3], acc[3]);
    }
}

// WARPS: one 16-row tile per warp. UNROLL: 64-column weight tiles loaded per lane before use.
// TILE_K: activation columns staged in shared memory per pass (stride +8 halves avoids bank
// conflicts on fragment reads).
// Writes this lane's C fragment: rows (g, g+8) of the tile, batch columns (2t, 2t+1).
__device__ __forceinline__ void store_tile(float* __restrict__ y, const float (&acc)[4], unsigned tile_r, unsigned g,
                                           unsigned t, unsigned N, unsigned B, bool split) {
    const unsigned row = tile_r * 16u + g, col = 2u * t;
    float* o0 = y + (u64)col * N + row;
    float* o1 = y + (u64)(col + 1) * N + row;
    if (col < B) {
        if (split) { atomicAdd(o0, acc[0]); atomicAdd(o0 + 8, acc[2]); }
        else { *o0 = acc[0]; o0[8] = acc[2]; }
    }
    if (col + 1 < B) {
        if (split) { atomicAdd(o1, acc[1]); atomicAdd(o1 + 8, acc[3]); }
        else { *o1 = acc[1]; o1[8] = acc[3]; }
    }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_PENDING>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N_PENDING)); }

// f32 activations, converted to f16 while staging (synchronous staging).
template <int WARPS, int UNROLL, int TILE_K>
__device__ __forceinline__ void gemm_mxfp4_mma(const u8* __restrict__ q, const u8* __restrict__ sc,
                                               const float* __restrict__ x, float* __restrict__ y,
                                               unsigned N, unsigned K, unsigned B) {
    constexpr unsigned XS_STRIDE = TILE_K + 8;
    __shared__ __align__(16) u16 xs[8 * XS_STRIDE];

    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned tiles_n = N >> 4, tiles_k = K >> 6;
    const unsigned tile_r = blockIdx.x * WARPS + warp;
    const bool active = tile_r < tiles_n;
    // K range of this split, in 64-column pairs.
    const unsigned p_begin = (unsigned)(((u64)tiles_k * blockIdx.y) / gridDim.y);
    const unsigned p_end = (unsigned)(((u64)tiles_k * (blockIdx.y + 1)) / gridDim.y);

    const uint4* qtile = reinterpret_cast<const uint4*>(q) + (u64)tile_r * tiles_k * 32u + lane;
    const unsigned* stile = reinterpret_cast<const unsigned*>(sc) + (u64)tile_r * tiles_k * 8u + g;
    const u16* xr = xs + (g < B ? g : 0u) * XS_STRIDE;   // columns >= B read row 0 and are discarded

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (unsigned pk0 = p_begin; pk0 < p_end; pk0 += TILE_K / 64) {
        const unsigned pk1 = min(p_end, pk0 + TILE_K / 64);
        const unsigned k0 = pk0 << 6, tk = (pk1 - pk0) << 6;
        __syncthreads();
        for (unsigned i = threadIdx.x; i < B * (tk >> 2); i += blockDim.x) {
            const unsigned b = i / (tk >> 2), k = (i - b * (tk >> 2)) << 2;
            const float4 f = __ldg(reinterpret_cast<const float4*>(x + (u64)b * K + k0 + k));
            const __half2 h01 = __floats2half2_rn(f.x, f.y), h23 = __floats2half2_rn(f.z, f.w);
            uint2 packed;
            packed.x = *reinterpret_cast<const unsigned*>(&h01);
            packed.y = *reinterpret_cast<const unsigned*>(&h23);
            *reinterpret_cast<uint2*>(&xs[b * XS_STRIDE + k]) = packed;
        }
        __syncthreads();
        if (active) {
            unsigned p = pk0;
            for (; p + UNROLL <= pk1; p += UNROLL) {
                uint4 w[UNROLL];
                unsigned s[UNROLL];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    w[u] = load_w(qtile + (u64)(p + u) * 32u);
                    s[u] = __ldg(stile + (u64)(p + u) * 8u);
                }
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) tile_pair(acc, w[u], s[u], xr, ((p + u) << 6) - k0, t);
            }
            for (; p < pk1; ++p) tile_pair(acc, load_w(qtile + (u64)p * 32u), __ldg(stile + (u64)p * 8u), xr, (p << 6) - k0, t);
        }
    }

    if (active) store_tile(y, acc, tile_r, g, t, N, B, gridDim.y > 1);
}


// f16 activations: staged with cp.async (no registers, no per-element waits) into two
// shared buffers, so the copy of the next activation tile overlaps compute on the current one.
// Needs K % 8 == 0 for 16-byte copies, which K % 64 == 0 guarantees.
template <int WARPS, int UNROLL, int TILE_K>
__device__ __forceinline__ void gemm_mxfp4_mma_f16x(const u8* __restrict__ q, const u8* __restrict__ sc,
                                                    const u16* __restrict__ x, float* __restrict__ y,
                                                    unsigned N, unsigned K, unsigned B) {
    constexpr unsigned XS_STRIDE = TILE_K + 8;          // 2 * (TILE_K + 8) bytes: 16-byte aligned rows
    __shared__ __align__(16) u16 xs[2][8 * XS_STRIDE];

    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned tiles_n = N >> 4, tiles_k = K >> 6;
    const unsigned tile_r = blockIdx.x * WARPS + warp;
    const bool active = tile_r < tiles_n;
    const unsigned p_begin = (unsigned)(((u64)tiles_k * blockIdx.y) / gridDim.y);
    const unsigned p_end = (unsigned)(((u64)tiles_k * (blockIdx.y + 1)) / gridDim.y);

    const uint4* qtile = reinterpret_cast<const uint4*>(q) + (u64)tile_r * tiles_k * 32u + lane;
    const unsigned* stile = reinterpret_cast<const unsigned*>(sc) + (u64)tile_r * tiles_k * 8u + g;
    const unsigned xrow = (g < B ? g : 0u) * XS_STRIDE;

    auto stage = [&](unsigned buf, unsigned pk0) {
        const unsigned pk1 = min(p_end, pk0 + TILE_K / 64);
        const unsigned k0 = pk0 << 6, chunks = (pk1 - pk0) << 3;   // 16-byte chunks per batch row
        for (unsigned i = threadIdx.x; i < B * chunks; i += blockDim.x) {
            const unsigned b = i / chunks, c = i - b * chunks;
            cp_async16(&xs[buf][b * XS_STRIDE + c * 8u], x + (u64)b * K + k0 + c * 8u);
        }
        cp_async_commit();
    };

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    if (p_begin < p_end) stage(0, p_begin);
    unsigned buf = 0;
    for (unsigned pk0 = p_begin; pk0 < p_end; pk0 += TILE_K / 64, buf ^= 1u) {
        const unsigned pk1 = min(p_end, pk0 + TILE_K / 64);
        const unsigned k0 = pk0 << 6;
        if (pk1 < p_end) stage(buf ^ 1u, pk1);        // prefetch the next tile into the other buffer
        else cp_async_commit();                         // keep the group count uniform
        cp_async_wait<1>();                             // everything except the newest group has landed
        __syncthreads();
        if (active) {
            const u16* xr = xs[buf] + xrow;
            unsigned p = pk0;
            for (; p + UNROLL <= pk1; p += UNROLL) {
                uint4 w[UNROLL];
                unsigned s[UNROLL];
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    w[u] = load_w(qtile + (u64)(p + u) * 32u);
                    s[u] = __ldg(stile + (u64)(p + u) * 8u);
                }
#pragma unroll
                for (int u = 0; u < UNROLL; ++u) tile_pair(acc, w[u], s[u], xr, ((p + u) << 6) - k0, t);
            }
            for (; p < pk1; ++p) tile_pair(acc, load_w(qtile + (u64)p * 32u), __ldg(stile + (u64)p * 8u), xr, (p << 6) - k0, t);
        }
        __syncthreads();                                // buffer `buf` is free for the tile after next
    }
    cp_async_wait<0>();
    if (active) store_tile(y, acc, tile_r, g, t, N, B, gridDim.y > 1);
}

#define MMA_ENTRY(W, U, T)                                                                                 \
    extern "C" __global__ void __launch_bounds__(32 * W) astral_gemm_mxfp4_mma_f32_w##W##u##U##t##T(       \
        const u8* __restrict__ q, const u8* __restrict__ sc, const float* __restrict__ x,                  \
        float* __restrict__ y, unsigned N, unsigned K, unsigned B) { PDL_WAIT();                                       \
        gemm_mxfp4_mma<W, U, T>(q, sc, x, y, N, K, B);                                              \
    }                                                                                                      \
    extern "C" __global__ void __launch_bounds__(32 * W) astral_gemm_mxfp4_mma_f16x_w##W##u##U##t##T(      \
        const u8* __restrict__ q, const u8* __restrict__ sc, const u16* __restrict__ x,                    \
        float* __restrict__ y, unsigned N, unsigned K, unsigned B) { PDL_WAIT();                                       \
        gemm_mxfp4_mma_f16x<W, U, T>(q, sc, x, y, N, K, B);                                               \
    }
MMA_ENTRY(4, 8, 1024)
MMA_ENTRY(4, 16, 1024)
MMA_ENTRY(8, 4, 1024)
MMA_ENTRY(8, 8, 1024)
MMA_ENTRY(2, 8, 1024)
MMA_ENTRY(2, 16, 512)
