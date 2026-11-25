# ADR-0001: Rust owns the runtime; precompiled CUDA artifacts own the kernels

**Status:** Proposed · **Date:** 2026-11-23

## Context
The research [R] found every competitive Rust engine (Atlas, mistral.rs, Dynamo's Rust modules, Grout) still depends on CUDA C++/cuBLAS/FlashInfer kernels. NVIDIA's Rust kernel tracks (cuda-oxide, cuTile Rust, announced Sept 2026) are labelled experimental. Triton/CuTe-DSL AOT produces cubins loadable from Rust but has no known Rust production precedent.

## Decision
Rust for everything above kernel launch. Kernels are CUDA C++/CUTLASS compiled to per-arch cubins at build time and loaded through a manifest. Triton/CuTe AOT only for non-critical glue, build-time only. Experimental Rust-kernel tracks only behind feature flags.
A100 quantised inference uses GGML-format dequant-matmul kernels (user decision 2026-11-23: Unsloth GGUF Q4 for speed, Q8 reference) rather than the NVFP4/MXFP8 checkpoint, which A100 cannot execute natively.

## Consequences
+ End users need only the NVIDIA driver; no Python or toolkit at runtime.
− Build needs CUDA 13.x toolchain; kernel authors need CUDA skills.
− GGML-derived kernels need attribution and legal review (risk 10).
