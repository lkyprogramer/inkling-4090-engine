# 09 — Roadmap and technical risks

Phase gates are defined in [SPEC §6](../spec.md); tests in [08](08-testing-and-validation.md).

## 1. Roadmap

| Phase | Scope | Exit gate | Status |
|---|---|---|---|
| P0 | Workspace, hardware detection, GPU boundary, memory arena, CPU mock backend, benchmark harness | R-HW-1..3, R-RT-1..3, R-VA-1 | Done |
| P1 | Inkling 2 Mini end to end from GGUF: loader, loop-aware graph, MoE, sliding-window and global attention, OpenAI-compatible API | Layer-by-layer parity with the llama.cpp reference; conformance suite | Done, except continuous batching |
| P2 | Speed: kernel tuning, CUDA graphs, DSpark speculative decoding, prefix cache, vision, Responses API | Reproducible benchmark against vLLM and SGLang | Decode and speculative decoding done; prefix cache, vision, batching open |
| P3 | Further architectures on real hardware: RTX PRO 6000, GB10, B200, GB300; native NVFP4/MXFP8 with a fallback path; FP8 KV | Per-architecture `VALIDATED` records | H100 validated; others open |

Multi-GPU serving on a single node belongs to the open-source developer edition and is planned after continuous batching. Multi-node serving belongs to the Enterprise edition.

## 2. Editions

- **Developer edition (this repository, Apache-2.0):** the engine, all kernels, the OpenAI-compatible server, speculative decoding, metrics; one node with one or more GPUs. Free for any use, commercial included.
- **Enterprise edition (separate, commercially licensed):** everything in the developer edition, plus multi-node and disaggregated serving, custom deployments and support.

## 3. Technical risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Unvalidated architectures ship broken** | High | `VALIDATED` status labels per architecture; static disassembly gates; real-hardware runs before a feature is on by default; no claims for unvalidated architectures |
| 2 | Inkling's architecture is documented mainly by its reference implementations; novel attention (relative bias, no RoPE, gating, clip), loop controllers, attention-residual mix and short convolutions all need custom kernels | High | Parity against llama.cpp on the same file, layer by layer (`astral verify-model`); cross-checks against the native checkpoint |
| 3 | Model churn; no generic model fallback in a pure-Rust runtime | Medium | One model family at a time; declarative layer and cache specifications |
| 4 | GGUF / DSpark format changes | Medium | Version-gate on metadata; speculative decoding stays optional |
| 5 | Toolchain risk on sm_12x (block-scaled FP4 instruction emission, target naming) | Medium | Disassembly gate; explicit per-architecture cubins |
| 6 | Licence and attribution of GGML-format-derived kernels | Medium | Written from the format specification; MIT attribution kept |
| 7 | Hardware access for P3 validation | Medium | Fully scripted validation runs, so a rented sweep takes hours |
