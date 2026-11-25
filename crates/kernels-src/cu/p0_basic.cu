
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
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//
// Astral P0 kernels (family F80, portable CUDA C++):
//   astral_read_bw_f32x4   - read-only DRAM bandwidth probe (decode is read-bound)
//   astral_copy_bw_f32x4   - read+write bandwidth probe
//   astral_rmsnorm_f32     - RMSNorm, optional weight (Inkling uses weight-less norms too)
//
// Parameter lists here are mirrored exactly by the typed launchers in
// crates/kernels/src/ops.rs. Change both together.

typedef unsigned long long u64;

extern "C" __global__ void astral_read_bw_f32x4(const float4* __restrict__ src, u64 n4, float* __restrict__ sink) { PDL_WAIT();
    const u64 stride = (u64)gridDim.x * blockDim.x;
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    // Four independent loads in flight per thread for memory-level parallelism.
    for (; i + 3 * stride < n4; i += 4 * stride) {
        const float4 a = src[i];
        const float4 b = src[i + stride];
        const float4 c = src[i + 2 * stride];
        const float4 d = src[i + 3 * stride];
        acc += (a.x + a.y + a.z + a.w) + (b.x + b.y + b.z + b.w) + (c.x + c.y + c.z + c.w) + (d.x + d.y + d.z + d.w);
    }
    for (; i < n4; i += stride) {
        const float4 a = src[i];
        acc += a.x + a.y + a.z + a.w;
    }
    // Data-dependent store keeps the loads alive without writing in practice.
    if (acc == -1.0f) {
        sink[0] = acc;
    }
}

extern "C" __global__ void astral_copy_bw_f32x4(const float4* __restrict__ src, float4* __restrict__ dst, u64 n4) { PDL_WAIT();
    const u64 stride = (u64)gridDim.x * blockDim.x;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        dst[i] = src[i];
    }
}

// One block per row. blockDim.x must be a multiple of 32 and <= 1024.
// weight may be null (weight-less RMSNorm).
extern "C" __global__ void astral_rmsnorm_f32(const float* __restrict__ x,
                                              const float* __restrict__ weight,
                                              float* __restrict__ y,
                                              unsigned rows,
                                              unsigned cols,
                                              float eps) { PDL_WAIT();
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (u64)row * cols;
    float* yr = y + (u64)row * cols;

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = xr[i];
        ss += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);

    __shared__ float partial[32];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) partial[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        const unsigned nwarps = (blockDim.x + 31u) >> 5;
        ss = lane < nwarps ? partial[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) partial[0] = ss;
    }
    __syncthreads();

    const float inv = rsqrtf(partial[0] / (float)cols + eps);
    for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
        const float w = weight ? weight[i] : 1.0f;
        yr[i] = xr[i] * inv * w;
    }
}
