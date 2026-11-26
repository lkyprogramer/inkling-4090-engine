# Changelog

All notable changes to Astral Inference Server are documented here.

## [Unreleased] — inkling-4090-engine fork

Changes in this fork on top of upstream `49a0a2b`.

### Added

- `ArchProfile::Sm89` for the RTX 4090 (F80 kernel family, FP8 tensor cores reported, no FP4 path); validation level Logic.
- `astral dump-logits <model> <llama kld base> <out>`: token-by-token log-probs in llama.cpp's `--kl-divergence-base` format.
- `astral cmp-prefill` also prints the near-top max difference, where the overall largest difference sits, and KL(step ‖ prefill).
- `bench/llamacpp_compare/`: KLD and top-1 comparison, tokenizer parity, greedy references and teacher-forced verdicts, streaming serving benchmark and paired statistics.
- Docs: [RTX 4090](docs/hardware/rtx4090.md) and [Inkling 2 Mini on RTX 4090](docs/benchmarks/inkling2_mini_rtx4090.md).
- GitHub issue and pull request templates; host-only CI (format, tests, script syntax).

### Fixed

- `DeviceBuffer::zeroed`, `DeviceBuffer::copy_from_slice` and `VmmRegion::write` wait for the legacy stream before returning, so kernels on non-blocking streams no longer race with the fill or the pageable upload. Seen as 3–4 failing MXFP4 self-test cases on a PCIe Gen3 RTX 4090.
- `quant::tests::mxfp4_block` builds the expected subnormal 2^-128 from bits; `2f32.powi(-128)` is 0.0 in debug builds.

### Performance (RTX 4090, Inkling 2 Mini UD-Q4_K_XL, single stream)

- Decode: 143.1 tok/s without a draft, 173.9 tok/s with DSpark; 1.92× llama.cpp b12751 on the same GGUF.

## [Unreleased] — upstream

### Changed

- Relicensed from AGPL-3.0-only to Apache-2.0. Releases up to and including 2.5 remain available under AGPL-3.0-only.

### Added

- Inkling 2 Mini loads straight from its native safetensors checkpoint (`astral serve <checkpoint dir>`), converted on load to the same tensors as the GGUF path.
- OpenAI-compatible server (`astral serve`): `/v1/completions`, `/v1/chat/completions` (streaming, stop strings, reasoning output, tool calls), `/v1/models`, `/tokenize`, `/health`.
- Inkling 2 Mini chat template and tokenizer, matching reference token ids.
- Speculative decoding with the Inkling 2 Mini DSpark draft: GPU-side acceptance, short-conv rollback, one CUDA graph per round, adaptive draft length.
- Int8 tensor-core GEMV for 1–8 tokens; grouped MoE GEMV for small batches.
- Fused kernels: depth-mix + RMSNorm + quantisation, attention merge + output gate, router GEMV + top-k, batched loop controllers.
- Programmatic dependent launch on compute capability 9.0+.
- Qwen3.8-27B: native 4-bit decode, wgmma prefill on Hopper.
- Benchmark client and prompt sets (`bench/serving/`).

### Performance (H100 PCIe, Inkling 2 Mini, single stream)

- Decode: 172.0 tok/s (1.32× vLLM) without a draft; 233.3 tok/s (1.11× vLLM) with the DSpark draft.

## [0.0.1]

### Added

- Workspace, GPU wrapper, memory arena, kernel build pipeline, hardware probing, self-tests.
