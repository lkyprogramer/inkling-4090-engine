# Astral — Master Specification

Version 0.1 (draft) · 2026-11-23 · Tags: see [README](../README.md)

## 1. Purpose and positioning
**Goal (product owner's decision):** a from-scratch, Rust-hosted engine, with precompiled CUDA kernels, that is **faster than vLLM and SGLang** and replaces them for users on NVIDIA hardware. The research report advised a narrower wedge [R]; this spec keeps the goal and treats the wedge as the **sequencing strategy**: win measurably where the incumbents are weakest first, then widen. "Faster" is a claim that must be proven by the benchmark harness (spec 08), not asserted. The beachheads, per the research [R]:
1. Correct, fast **sm_120/sm_121** (RTX PRO 6000, GB10) execution, where incumbents have documented FP4 bugs and mis-dispatch.
2. **Typed KV memory** for hybrid/local-attention/looped models, where incumbents pad to a single page size.
3. **Fast cold start** and a single binary from workstation to datacentre.
4. **Low CPU overhead** and predictable tail latency (to be *measured*, not assumed: vLLM V1 already recovered much overhead [R]).

Commercial rule [R]: charging is only defensible with a reproducible ≥20% gain over vLLM/SGLang on a buyer's workload. Phase gates below therefore start with a benchmark harness.

## 2. Goals / non-goals
**Goals (v1):** serve Inkling 2 Mini on A100 from GGUF with an OpenAI-compatible API, continuous batching, prefix caching, chunked prefill, and speculation; a capability-driven hardware layer so other architectures are additive.
**Non-goals (v1):** training; non-NVIDIA hardware; Python in the runtime or install path; multi-node (Enterprise edition); native NVFP4/MXFP8 kernels running in production (specified, unvalidated); LoRA serving; models other than Inkling 2 Mini beyond a dense-Llama smoke model used for testing, and **Qwen3.8-27B as a second test model** (decided 2026-11-23) so that vLLM and SGLang, which cannot run Inkling on an A100, can be compared on the same GPU (see STATUS.md).

## 3. Requirements
IDs are stable; each phase gate in §6 and spec 08 references them. MUST = required for the phase named; SHOULD = target.

### Hardware & dispatch (spec 01)
- **R-HW-1 (MUST, P0)** Detect arch via CUDA driver API (capability, SM count, opt-in smem, clocks, bus width, integrated flag); NVML is optional and `NOT_SUPPORTED` is tolerated.
- **R-HW-2 (MUST, P0)** Dispatch on `Capabilities` flags, never on `major >= N`.
- **R-HW-3 (MUST, P0)** Memory accounting has a UMA mode keyed on the integrated flag.
- **R-HW-4 (SHOULD, P1)** Startup microbenchmark (≤300 ms) measures bandwidth and detects under-clocking; cached by (device, driver, clock state).
- **R-HW-5 (MUST, P3)** Per-arch cubins for sm_80/90a/100a/103a/120a/121a in one fatbin; CI disassembles to assert expected instructions.

### Runtime (spec 02)
- **R-RT-1 (MUST, P0)** All `unsafe` GPU code lives in the `gpu` crate behind typed handles.
- **R-RT-2 (MUST, P0)** Scheduler runs on a dedicated OS thread with no `.await`; Tokio only at the edges.
- **R-RT-3 (MUST, P0)** A mock GPU backend lets the whole engine run in CPU-only CI.
- **R-RT-4 (MUST, P1)** CUDA graph capture per batch-size bucket over a fixed-address arena.

### Kernels & quantisation (spec 03)
- **R-KQ-1 (MUST, P1)** Fused dequant-matmul kernels for every GGML type present in the target GGUF file — for `UD-Q4_K_XL`: **NVFP4 (id 40, with per-expert F32 scale) and Q8_0**, plus F32/BF16 ops — bit-exact (integer paths) vs CPU dequant reference. K-quant/IQ kernels only if a chosen file needs them.
- **R-KQ-2 (MUST, P1)** MoE grouped path for 128 experts, top-8.
- **R-KQ-3 (MUST, P1)** Custom attention kernels for Inkling's no-RoPE relative-position bias, sigmoid output gate, Q/K/R L2 clip and global log-scale (GQA 24/8, head_dim 128; SWA and global variants) with paged KV. FlashAttention/FlashInfer are not drop-in (spec 03).
- **R-KQ-6 (MUST, P1)** Fused kernels for attention-residual mix, loop controllers (F32) and 4-site short causal conv with per-slot state (spec 06 §3).
- **R-KQ-4 (SHOULD, P2)** Decode path reaches ≥70% of A100 memory bandwidth on weight reads at batch 1 `[I]` (target, not sourced).
- **R-KQ-5 (MUST, P3)** NVFP4/MXFP8 block-scaled kernels for sm_100/120/121, validated on real silicon before any claim.

### KV & memory (spec 04)
- **R-KV-1 (MUST, P1)** Typed multi-pool KV arena under one byte-budget arbiter, sized from a per-slot table (Inkling: 48 slots = 8 global + 40 SWA, plus per-slot conv state).
- **R-KV-2 (MUST, P1)** Analytic memory budget from the model graph plus short measured warm-up; no OOM at admitted concurrency.
- **R-KV-3 (SHOULD, P2)** Prefix cache across all pool types with cost-aware eviction.
- **R-KV-4 (SHOULD, P2)** FP8 KV option on sm_89+/90+ only; on sm_80 FP16/BF16 KV (INT8 KV optional).

### Scheduler (spec 05)
- **R-SC-1 (MUST, P1)** Continuous batching with chunked prefill and stall-free decode.
- **R-SC-2 (MUST, P1)** Overlap scheduling (step n+1 prepared while step n runs).
- **R-SC-3 (SHOULD, P2)** Load-adaptive speculation depth using the DSpark drafter.
- **R-SC-4 (Enterprise edition)** Priority classes and per-tenant quotas.

### Model (spec 06)
- **R-MD-1 (MUST, P1)** Loop-aware execution of Inkling 2 Mini (24 weight layers, 48 applications).
- **R-MD-2 (MUST, P1)** Perplexity and top-token agreement within tolerance of llama.cpp on the same GGUF.
- **R-MD-3 (SHOULD, P2)** Vision via `mmproj`.

### API & ops (spec 07)
- **R-API-1 (MUST, P1)** **OpenAI-compatible endpoint, drop-in for the official OpenAI SDKs** (change only `base_url`): Chat Completions + Completions, SSE streaming, `/v1/models`, OpenAI-format errors and `usage`. Full contract in spec 07 §1; a failing conformance test blocks the P1 gate.
- **R-API-2 (SHOULD, P2)** Responses API, tool-call and reasoning parsers matching the model's chat template.
- **R-API-3 (MUST, P1)** Prometheus metrics with `vllm:`-compatible names; OTel GenAI metrics.

### Validation (spec 08)
- **R-VA-1 (MUST, P0)** Reproducible benchmark harness with pinned baselines before any performance claim.
- **R-VA-2 (MUST, P0)** Every non-A100 feature is labelled *unvalidated* in code, docs, and release notes until run on real silicon.

## 4. Architecture overview
```
   HTTP/gRPC (axum/tonic, Tokio multi-thread)  ── tokenise (blocking pool)
                     │  bounded lock-free channels
                     ▼
   Engine thread (pinned, no .await):  sched → batch plan → runtime step
        │                                         │
        ▼                                         ▼
   kvcache (typed pools, prefix tree)      model (layer specs, loop driver)
        │                                         │
        └────────────► mem (VMM arena) ◄──────────┘
                                │
                kernels (manifest + typed launch)  ◄── kernels-src (CUDA/CUTLASS → cubin, build time)
                                │
                  gpu (only `unsafe`): Context/Stream/Event/Graph  ── hw (ArchProfile, Capabilities)
   detok/stream thread overlaps step n+1.
```
Crate list and interfaces: spec 02. Hardware layer: spec 01.

## 5. Rust vs CUDA split [R]
Rust: API, scheduler, batching, block manager, prefix cache, tokenisation (HF `tokenizers`), grammar masks (llguidance), loader, quant metadata/packing, control plane, observability, CUDA wrapper, graphs, allocator. CUDA C++/CUTLASS: GEMM/dequant-matmul, attention, MoE grouped GEMM, fused norm/RoPE/quant, custom all-reduce. Triton/CuTe-DSL AOT only for glue/experimental kernels, behind a build-time feature; **end users never need Python**. cuTile Rust / cuda-oxide: optional, non-critical kernels only (NVIDIA labels them not production-ready [R]).

## 6. Phase gates
| Phase | Scope | Gate (must pass to exit) |
|---|---|---|
| P0 | workspace, `hw`, `gpu`, `mem`, mock backend, harness | R-HW-1..3, R-RT-1..3, R-VA-1; harness reproduces a pinned llama.cpp baseline on the A100 |
| P1 | Inkling 2 Mini on A100 (GGUF) end-to-end | R-KQ-1..3, R-KV-1..2, R-SC-1..2, R-MD-1..2, R-API-1,3; UD-Q8_K_XL perplexity within tolerance of ~7.65 [G]; top-token agreement vs llama.cpp |
| P2 | A100 speed | R-KQ-4, R-KV-3..4, R-SC-3, R-MD-3, R-API-2; speed vs llama.cpp on identical file |
| P3 | other architectures | R-HW-5, R-KQ-5; **requires rented/borrowed hardware per arch**; no claim without measurement |
| — | Enterprise edition (separate, commercially licensed) | R-SC-4, multi-node, disaggregation, RBAC |

## 7. Open questions (blockers listed in the spec that owns them)
**Resolved 2026-11-23** from llama.cpp source and the HF repo (spec 06): loop passes use **per-pass KV** (48 slots); positions are a learned **relative-position bias, no RoPE**; GGUF loop keys and tensor names; chat template, tool and reasoning formats; native checkpoint quantises only experts.
Also resolved from the real `UD-Q4_K_XL` header and llama.cpp source: per-tensor types (NVFP4 experts + Q8_0, no K-quants), SITU formula, final softcap, routing details.
**Still open (spec 06 §8):** types in the other `UD-*` files; NVFP4 scale decode details; first-token behaviour after the (empty) generation prompt; tokenizer parity. Spec 01: datasheet-confirmed chip specs. **A100 SKU resolved: A100-PCIE-40GB** on the dev server (Ubuntu 24.04, 30 vCPU AMD EPYC 7H12, 221 GiB RAM, driver 580.105.08, CUDA toolkit 12.8, no Rust installed yet).

## 8. Licensing and attribution
**AGPL-3.0-only, copyright (c) 2026 Synthetic Minds LTD.** Commercial licences are offered for use without AGPL obligations. Multi-node serving, custom deployments and support are offered separately as the Enterprise edition. A Contributor Licence Agreement is required before accepting outside code, so dual licensing remains possible. GGML block-format-derived kernels are written against the format specification and the MIT-licensed llama.cpp reference; each such kernel keeps the upstream MIT notice. Every file carries the SPDX header `AGPL-3.0-only` and the copyright line.
