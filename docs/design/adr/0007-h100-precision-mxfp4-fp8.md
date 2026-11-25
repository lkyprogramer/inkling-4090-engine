# ADR-0007: On the H100, MXFP4 weights first, FP8 second

**Status:** Accepted (owner decision) · **Date:** 2026-11-24

## Context
The dev GPU is now an H100 PCIe 80 GB (sm_90a). Its native low-precision tensor-core formats are FP8 (E4M3/E5M2), BF16/FP16 and INT8. It has **no FP4 tensor cores**: MXFP4 and NVFP4 are only native on Blackwell (sm_100/103/120/121). The owner asked to run only what the GPU executes natively (no CPU emulation) and to prefer MXFP4, then FP8.

## Decision
1. **MXFP4 weights are the primary H100 format.** Weights stay 4-bit in GPU memory (4.25 bit/weight: E2M1 values plus one E8M0 scale per 32) and are unpacked **inside the GPU kernel**, in registers or shared memory. This is still entirely GPU execution; it is also how vLLM serves 4-bit checkpoints on the H100 (Inkling's experts stay at 18.2 GiB in vLLM).
   - Step 1, **W4A16**: unpack E2M1 × E8M0 to BF16 and compute in BF16. Exact. Decode is memory-bound, so 4-bit weights are where the speed comes from.
   - Step 2, **W4A8**: unpack to FP8 E4M3 and use FP8 WGMMA. Every E2M1 value (0, ±0.5 … ±6) is exactly representable in E4M3; the per-32 E8M0 scale is applied per k-chunk on the FP32 accumulators. This doubles tensor-core throughput for prefill and large batches.
2. **FP8 is the second format**: native FP8 checkpoints (e.g. `Qwen/Qwen3.8-27B-FP8`) run directly on FP8 WGMMA.
3. **NVFP4 shares the kernel family**: same E2M1 values, but a 16-element block and an E4M3 scale instead of 32 and E8M0. Inkling's checkpoint stores its experts as NVFP4 (layers 0–19) and MXFP8 (20–23), so it rides on the same kernels with a different scale decode (spec 03 §4 already parameterises block size and scale type).
4. **Astral repacks 4-bit weights at load** into a kernel-friendly layout: values contiguous and 16-byte aligned, scales in a separate array. GGUF stores MXFP4 as interleaved 17-byte blocks, which cannot be read with 128-bit loads.
5. **CPU code is only a test oracle.** `astral-quant` decoders check GPU kernels bit-for-bit; no model runs on the CPU.

## Consequences
+ The kernel work lands on formats the owner cares about, and on the GPU we can test.
+ The weight-unpacking FP4 kernels are also the Blackwell fallback path (spec 03 §4: a Marlin-style fallback must exist before native FP4 is enabled).
− MXFP4 checkpoints are needed for a like-for-like comparison: `RedHatAI/Qwen3.8-27B-MXFP4` exists (named by vLLM's recipe; size not yet checked). Inkling has no MXFP4 release; converting its NVFP4 experts to MXFP4 would be lossy (coarser block, power-of-two scale), so Inkling runs as NVFP4 on the same kernels.
