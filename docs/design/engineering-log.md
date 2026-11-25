# Astral — status

Dev server: **H100 PCIe 80 GB** (sm_90a, 114 SMs, 50 MB L2, PCIe 5.0 x16, 350 W), driver 580.105.08, CUDA 12.8, Ubuntu 24.04, Rust 1.100, since 2026-11-24. P0 was first verified on an A100-PCIE-40GB (sm_80), now decommissioned; its results stay in `bench/results/`.

## P0 progress (2026-11-23)

| P0 item | Requirement | State | Evidence |
|---|---|---|---|
| Workspace, 18 crates | — | done (8 implemented, 10 placeholders) | `cargo build --release`, clippy clean |
| Hardware detection `astral hw` | R-HW-1, R-HW-2 | done | JSON report; arch table unit tests (no `major >= N`) |
| UMA memory accounting | R-HW-3 | done (logic) | `memory_budget()`: discrete = free × 0.90; unified = min(free, `MemAvailable`) − 8 GiB; unit-tested with fake `/proc/meminfo`; **on-GB10 check is L7** |
| Bandwidth microbench + cache | R-HW-4 | done | 1,490 GB/s read; cached per (GPU UUID, driver, max clocks) in `~/.cache/astral/hw-bench-v1.json`; time-based warm-up (see below) |
| Single `unsafe` boundary | R-RT-1 | done | `unsafe` only in `gpu` + typed launchers in `kernels/src/ops.rs` |
| Engine thread | R-RT-2 | done (skeleton) | `astral-runtime`: dedicated thread, optional core pinning, no async; bounded queue reports "full"; dropped receiver cancels and frees; tested with a deterministic mock model (5 tests). Tokio edges arrive with the API in P1 |
| Mock backend | R-RT-3 | done (first op) | `Backend` trait; `CpuBackend` runs in CPU-only tests; CPU vs CUDA on identical code passes |
| CUDA graph capture/replay | R-RT-4 | done | 48-launch forward bit-identical to eager |
| VMM arena + slab pools | R-KV-1 (base) | done | write/read/reuse on GPU; 20k-step randomised overlap/budget test on CPU |
| Benchmark harness | R-VA-1 | v0 done | `bench/baseline_llamacpp.sh`, results in `bench/results/` |
| Golden reference data | spec 08 L3/L4 | v0 done | `bench/golden_llamacpp.sh`: perplexity, full log-prob dump for KLD, per-op tensor stats |
| Static arch gate (SASS check) | R-HW-5 | done for all six targets | `ci/check_fatbin.sh` + `ci/sass_expectations.txt`: cubins for sm_80, 90a, 100a, 103a, 120a, 121a present and expected instructions found, built with CUDA 13.5 (`/usr/local/cuda-13.5`). CUDA 12.8 (default, used for A100 runs) rejects `compute_103a`/`121a` |

`astral selftest`: 17/17 cases pass (RMSNorm ×14, CPU-vs-CUDA backend, VMM arena, CUDA graph). `cargo test`: 29 unit tests pass. clippy (all targets): clean.

## Measurements

### Bandwidth benchmark warm-up
Idle, the A100 parks its SM clock at **210 MHz** (`nvidia-smi`, clock-event reason "idle"). A benchmark that starts immediately under-reads by ~30% (1,074 GB/s observed). The bench now warms for 300 ms of wall time before measuring; from idle it reads **1,491 GB/s** (95.9% of the 1,555 GB/s derived from clocks). Every Astral benchmark must warm up the same way, and so must the baselines we compare against.

### CUDA graphs vs eager (48 tiny RMSNorm launches = one "forward" over 48 slots)
| | GPU time / forward | Host enqueue time / forward |
|---|---|---|
| Eager launches | 454 µs | 406 µs |
| CUDA graph replay | 323 µs | 2.4 µs |

Even inside a graph, 48 small kernels cost ~6.7 µs each. With 48 slot applications per Inkling token and several ops per slot, **launch count, not just bandwidth, will bound batch-1 decode** unless ops are fused (spec 03 §3 fused ops; megakernel track).

### llama.cpp baseline: Inkling 2 Mini `UD-Q4_K_XL` on the A100
llama.cpp `b12655-20-gc212d25ad`, model SHA-256 `e6427665…cd3` (matches Hugging Face), 5 reps, default flags (`-ngl 99`, flash-attn auto). Raw files: `bench/results/20261123T220809Z-llamacpp/`.

| Test | Result |
|---|---|
| Prefill pp512 | 1,251 tok/s (±15) |
| Prefill pp2048 | 1,033 tok/s (±2) |
| **Decode tg128, batch 1** | **45.7 tok/s** |

Concurrent decode (`llama-batched-bench`, 512-token prompts, 128 generated per sequence):

| Parallel sequences | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Aggregate decode tok/s | 45.0 | 81.9 | 115.9 | 169.2 | 286.3 | 431.9 |
| Per-sequence tok/s | 45.0 | 40.9 | 29.0 | 21.1 | 17.9 | 13.5 |
| Prefill tok/s | 1,128 | 1,267 | 1,277 | 1,290 | 1,289 | 1,290 |

**Reading this:** batch-1 decode at 45.7 tok/s is ~13% of the ~360 tok/s weight-bandwidth ceiling computed in spec 03 §3 (4.1 GB read per token at the measured 1,483 GB/s). The ceiling ignores attention/KV reads and launch overhead, so the real attainable figure is lower, but the gap is large: this is where Astral's fused, Inkling-specific kernels have the most room. These are llama.cpp numbers only; vLLM and SGLang have not been run yet (they need the native NVFP4 checkpoint, which the A100 cannot run natively — to be checked).

llama.cpp log warning observed: `special_eos_id is not in special_eog_ids` (tokenizer config); not yet investigated.

### Golden reference data (llama.cpp, `UD-Q4_K_XL`)
Raw files: `bench/results/20261123T224727Z-golden/` (the 816 MB log-prob dump stays on the server at `~/astral-work/results/20261123T224727Z-golden/`; its SHA-256 is in the repo).

| Item | Result |
|---|---|
| Corpus | llama.cpp `README.md` + `docs/**/*.md` at commit `c212d25a`, 604,168 bytes, SHA-256 `4cd0c1d9…6390` (no download needed; rebuildable) |
| Perplexity (ctx 2048, 16 chunks) | **3.2393 ± 0.0529**, no NaN (upstream CUDA NaN issue #32710 not hit) |
| Log-prob base for KLD (ctx 512, 8 chunks) | 816 MB, for Astral-vs-llama.cpp KLD and top-token agreement in P1 |
| Per-op tensor stats, prompt "The capital of France is" | 12,178 tensors with shapes, sample values and sums |

The perplexity value is on our corpus, so it is **not** comparable to Unsloth's 7.65 (unknown corpus); it is the like-for-like target for Astral on this file.

Confirmed from the dump: the per-slot short-conv state (`cache_r_l0`) is **19,968 F32 values** = 3 × 6,656 channels, as estimated in spec 04 §4 → ≈ 3.8 MB per sequence across 48 slots.

## P0 exit gate (SPEC §6)
| Gate item | State |
|---|---|
| R-HW-1..3, R-RT-1..3, R-VA-1 | met on the A100 (R-HW-3 logic only: no unified-memory hardware here) |
| Harness reproduces a pinned llama.cpp baseline | met (baseline + golden runs, versions and hashes recorded) |

## Server-level baselines (2026-11-23)
Same load generator for every engine: `vllm bench serve` over the OpenAI `/v1/completions` API, random prompts, 512 input / 128 output tokens, `ignore_eos`, 8×concurrency prompts (min 16). Script: `bench/baseline_servers.sh`; raw files in `bench/results/*-servers/`.

### Which engines can run Inkling 2 Mini on an A100 at all
| Engine | Image / build | Result |
|---|---|---|
| llama.cpp `llama-server` | b12655-20-gc212d25ad, `UD-Q4_K_XL` GGUF | **runs** |
| vLLM release | `vllm/vllm-openai:latest` = vLLM 0.34.0, transformers 5.22.2 | **fails**: transformers does not recognise model type `inkling2_loop` |
| vLLM nightly | `vllm/vllm-openai:nightly` = 0.34.0rc1.dev1163+geff4de9be, transformers 5.23.0 | **fails**: "quantization method inkling2 is not supported for the current GPU. Minimum capability: 90. Current capability: 80" |
| SGLang | `lmsysorg/sglang:latest` | **fails**: `KeyError: 'inkling2_loop'` (transformers does not recognise the architecture) |

So on the A100 the only incumbent that serves Inkling 2 Mini today is llama.cpp. vLLM nightly deliberately requires Hopper or newer for Inkling's native NVFP4/MXFP8 checkpoint, and the checkpoint has no BF16 variant. That is direct evidence for the spec's thesis that hardware coverage is a real gap. It also means **"beat vLLM/SGLang" cannot be measured on this GPU for this model**; that comparison needs an H100 or newer (P3 rented hardware), or a different model both can run on an A100.

### llama.cpp `llama-server` over HTTP
| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Output tok/s (aggregate) | 34.9 | 64.3 | 74.6 | 90.4 | 94.7 | 106.6 |
| Median TTFT (ms) | 641 | 389 | 1,573 | 1,917 | 3,992 | 4,456 |
| Median time per output token (ms) | 23.9 | 28.3 | 42.2 | 67.0 | 138.7 | 266.1 |
| p99 inter-token latency (ms) | 47 | 56 | 84 | 298 | 2,159 | 2,617 |

Compared with llama.cpp's in-process `llama-batched-bench` on the same file (45 → 432 tok/s from 1 to 32 sequences), the server reaches only 35 → 107 tok/s, with multi-second p99 stalls from 16 concurrent requests up. The gap is the serving layer: scheduling, prefill interleaving, HTTP and sampling. That is exactly what Astral's engine thread, chunked prefill and stall-free batching target. **These are the numbers Astral's server must beat on this A100** (same client, same workload).

## Qwen3.8-27B on the A100 (2026-11-24) — all three engines run
Same client and workload as above. Checkpoints: vLLM/SGLang `RedHatAI/Qwen3.8-27B-INT4` (W4A16, group 128); llama.cpp `unsloth/Qwen3.8-27B-GGUF` `UD-Q4_K_XL`. vLLM `vllm/vllm-openai:latest` (0.34.0) with `--language-model-only`; SGLang `lmsysorg/sglang:latest`; llama.cpp b12655-20-gc212d25ad with 32 slots × 2,048 context. Raw files: `bench/results/20261124T1*-servers-qwen38/`.

Output tokens/s (aggregate):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| llama.cpp server | 36.5 | 67.8 | 73.4 | 26.8 / 41.3 (re-run) | 81.7 | 110.0 |
| vLLM | 57.4 | 103.4 | 175.5 | 269.5 | 365.2 | **434.5** |
| SGLang | **58.6** | **105.1** | **196.3** | **277.9** | 279.1 | 283.6 |

Median time per output token (ms):

| Concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| llama.cpp server | 22.2 | 25.0 | 27.9 | 31.9 / 75.2 | 90.8 | 217.2 |
| vLLM | 16.0 | 16.8 | 17.2 | 21.6 | 32.5 | 60.9 |
| SGLang | 15.6 | 16.6 | 16.9 | 18.2 | 21.6 | 20.4 |

Median TTFT at 32 concurrent: llama.cpp 5.2 s, vLLM 1.7 s, SGLang 12.8 s.

**Reading this:**
- vLLM and SGLang run the hybrid (Gated DeltaNet + attention) model on sm_80 even though neither recipe lists the A100. These are the numbers "faster than vLLM/SGLang" is measured against on this class of GPU.
- SGLang is fastest up to 8 concurrent, then plateaus near 280 tok/s with low per-token latency but long queueing (TTFT 12.8 s at 32). That pattern suggests it caps the running batch, likely because each request reserves large recurrent-state slots (cookbook: ~78–154 MB per slot, 5 slots per request by default). This is an inference from the numbers, not checked in its logs. It is exactly the hybrid-model memory problem Astral's typed pools target (spec 04, ADR-0003).
- vLLM scales further (434 tok/s at 32) but per-token latency triples from 8 to 32.
- llama.cpp's server trails badly and is erratic at 8 concurrent (two runs: 26.8 and 41.3 tok/s, both far below its 4- and 16-concurrent figures). The cause is not investigated.
- Fairness caveat: both checkpoints are 4-bit weights with 16-bit activations, but not bit-identical. The GGUF mixes quant types per layer; the Red Hat build leaves 304 modules in BF16 (not yet itemised).

**A100 decommissioned after this run** (owner can keep only one GPU node). Work moved to the H100 PCIe node. Nothing remains only on the A100: models and images are re-downloadable, and the 816 MB KLD reference is regenerable from `bench/golden_llamacpp.sh`.

## H100 PCIe baselines (2026-11-24)
Same client and workload as on the A100. Raw files: `bench/results/h100/`. Astral on the H100: `astral hw --bench` read **1,962 GB/s** (96% of the 2,039 GB/s derived from clocks); self-test passes with sm_90a cubins.

Output tokens/s (aggregate):

| Model / engine | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Inkling, llama.cpp server (GGUF UD-Q4_K_XL) | 44.9 | 83.7 | 98.3 | 123.2 | 120.9 | 135.4 |
| **Inkling, vLLM nightly** (native NVFP4/MXFP8, `0.34.0rc1.dev1235`) | **108.3** | **167.2** | **271.0** | **395.8** | **919.3** | **1,308.3** |
| Inkling, SGLang | fails: architecture not recognised | | | | | |
| Qwen 4-bit, llama.cpp server | 52.3 | 100.4 | 119.2 | 57.0 | 139.7 | 165.1 |
| Qwen 4-bit, vLLM 0.34.0 (`--max-num-seqs 256`) | 59.3 | 111.1 | 209.6 | 365.8 | 570.3 | **823.6** |
| Qwen 4-bit, SGLang | **75.3** | **136.8** | **211.7** | 317.2 | 569.9 | 622.4 |

- **The Inkling target for Astral on the H100 is vLLM nightly:** 108 tok/s single-stream (7.3 ms/token) and 1,308 tok/s at 32 concurrent. vLLM keeps Inkling's experts 4-bit (18.2 GiB loaded), so on Hopper it unpacks FP4 in-kernel, as Astral will.
- vLLM's per-layer backend table lists **48 attention caches for Inkling** (24 layers + 2 extra passes × 12 core layers): independent confirmation of the per-pass KV design (spec 04).
- vLLM 0.34.0 refuses to start Qwen on the H100 without `--max-num-seqs`: it defaults to 1,024 sequences but only 1,023 Mamba state blocks fit.
- llama.cpp in-process on Inkling (H100): decode 56.1 tok/s at batch 1; `llama-batched-bench` 59 → 560 tok/s from 1 to 32 sequences; prefill 1,732 tok/s (pp512).
- llama.cpp's server shows the same dip at 8 concurrent on Qwen as on the A100 (57.0 tok/s); still not investigated.

### Qwen3.8-27B on the H100 by weight format (2026-11-24)
Same client and workload. Output tokens/s (aggregate); raw files `bench/results/h100/*-servers-qwen38-{mxfp4,fp8}/`.

| Engine / checkpoint | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| **vLLM, MXFP4** (`RedHatAI/Qwen3.8-27B-MXFP4`) | **74.1** | **131.5** | **255.8** | **452.5** | **713.7** | **1,001.3** |
| vLLM, FP8 (`Qwen/Qwen3.8-27B-FP8`) | 55.5 | 102.6 | 200.1 | 363.5 | 603.4 | 869.6 |
| vLLM, INT4 W4A16 (`RedHatAI/Qwen3.8-27B-INT4`) | 59.3 | 111.1 | 209.6 | 365.8 | 570.3 | 823.6 |
| SGLang, FP8 | 54.5 | 103.9 | 170.7 | 314.2 | 603.5 | 749.4 |
| SGLang, INT4 W4A16 | 75.3 | 136.8 | 211.7 | 317.2 | 569.9 | 622.4 |
| SGLang, MXFP4 | fails: `NotImplementedError: No compressed-tensors compatible scheme was found` | | | | | |

MXFP4 is vLLM's fastest Qwen format on the H100 (+34% over FP8 at batch 1, +15% at 32). Median time per output token at batch 1: 12.7 ms (MXFP4) vs 17.3 ms (FP8). These are the MXFP4 and FP8 targets for Astral.

## P1 progress (2026-11-24)
| Item | State | Evidence |
|---|---|---|
| H100 precision decision | done | ADR-0007: MXFP4 weights first (unpacked in-kernel; the H100 has no FP4 tensor cores), FP8 second, NVFP4 on the same kernel family |
| `astral-quant` | done | GGML type table (ids, block layouts for 20 types); bit-exact ggml decoders for F16, BF16, Q8_0, NVFP4, MXFP4 (CPU test oracle only); MXFP4 repack to kernel layout; exhaustive f16 test |
| `astral-loader` (GGUF) | done | bounds-checked parser (9 hostile-file cases rejected); Inkling and Qwen GGUFs parse in 0.13 s; **0 bytes of the data sections unaccounted for**, which validates the layout table for every type those files use (nvfp4, q8_0, q3/4/5/6_K, iq4_xs, iq4_nl) |
| Loader vs llama.cpp | **pass** | `astral verify-embd`: found the 5 embedding rows llama.cpp printed for "The capital of France is" (token ids 371, 151451, 149879, 151194, 149884); sum −144.010567 vs −144.010513 (4×10⁻⁷ relative) |
| MXFP4 decode GEMV (sm_80/sm_90a) | v1 correct, half of bandwidth at batch 1 | 15 shape × batch cases pass (worst error 1.0×10⁻⁷ of Σ\|w·x\|, f32 activations). See the kernel log below |

Unit tests: 44 pass. GPU self-test: 72/72. clippy clean. The all-six-architecture SASS gate needs CUDA 13.5, which is not yet installed on the H100 (it was on the A100).

### MXFP4 GEMV kernel log (H100 PCIe, 16384 × 8192 MXFP4 = 71.3 MB, larger than L2)
Measured with `astral kbench mxfp4-gemv` (300 ms warm-up, 50 calls) and Nsight Compute 2025.1.1.

| Version | Change | b=1 | b=2 | b=3 | b=4 | Bottleneck (ncu) |
|---|---|---|---|---|---|---|
| v0 | correctness baseline | 299 GB/s | | | 147 GB/s | compute 63%, DRAM 9.7%: runtime-indexed accumulators spill to local memory, branchy E2M1 decode, per-element shared reads |
| v1 | one kernel per batch size; bit-trick E2M1→f16x2 unpack with the 2^14 folded into the scale; padded 128-bit shared reads; 2 blocks in flight | **1,003 GB/s** | 810 | 644 | 460 | issue-bound at b=1 (77% issue slots busy, DRAM 46%); FP32 FMA count grows with b |
| v2 | tensor cores: E2M1→f16 in registers, `mma.sync.m16n8k16`, fp32 accumulate, exact per-block fp32 scaling; f16 activations (W4A16); weights pre-arranged per lane at load (`repack_mxfp4_mma`) | 550 | 507 | | 407 (b8: 289) | each 16-row block re-read all activations: ~268 MB of L2 traffic at b=8 |
| v2.1 | 64 rows per block, stage only the B real rows, 4 loads in flight per lane, split-K across blocks | 1,092 | 1,011 | | 846 (b8: 629) | latency-bound: 14% achieved occupancy, DRAM 52% |
| v2.2 | split-K target swept (2/4/6/8/12 blocks per SM), best at 4 | 1,354 | 1,274 | | 1,136 (b8: 926) | ~60% of stalls waiting on global loads, ~4 active warps per scheduler |
| v2.3 | kernel templated on warps / loads in flight / activation tile; variants × blocks-per-SM swept; default 4 warps, 8 loads in flight, 1,024-column tile, 8 blocks/SM; f16-activation entry point | 1,445 | 1,378 | | 1,314 (b8: 1,175) | batch 8 ~20% below batch 1 per byte: per-thread serial staging loads behind a barrier (ncu: +20% long-scoreboard stalls, +18% instructions) |
| **v2.4** (f16 activations) | activations staged with `cp.async` into two shared buffers, so the next tile's copy overlaps compute; default 8 warps, 8 loads in flight, 1,024-column tile | **1,436** | **1,420** | | **1,388** (b8: **1,338**) | batch 8 now within 7% of batch 1 per byte |

Columns are batch sizes 1, 2, 3 and 4 (b8 noted in brackets). **v2.4 (f16 activations) reaches 73% of the H100's measured 1,962 GB/s at batch 1 and 68% at batch 8**; v1 was at 51% at batch 1 and 23% by batch 4. Medians of 3 runs; spread about ±1%. The f32-activation entry keeps v2.3 (synchronous staging; 1,446 / 1,164 GB/s at b1 / b8) as the exact-input reference path.

Measured and rejected while closing the batch-8 gap: 4 or 8 staging loads in flight per thread (slower: registers cost occupancy), and removing the split-K atomics (no change, so atomics were not the cost).

v2.4 f16 path, all shapes (GB/s, b1 / b2 / b4 / b8): 16384×8192 1,436 / 1,420 / 1,388 / 1,338; Qwen 5120×17408 1,311 / 1,288 / 1,248 / 1,223; Qwen 17408×5120 1,316 / 1,234 / 1,224 / 1,172; Inkling 2304×2304 335 / 343 / 339 / 317. Correctness: 36 shape × batch cases (f32 and f16 activation entry points) match an f64 reference on f16-rounded activations to within 7×10⁻⁸ of Σ|w·x|. The SASS gate asserts both entry points compile to `HMMA.16816.F32` on sm_80 and sm_90a.

Other shapes, v2.3 (b1 / b8): Qwen FFN 5120×17408 1,275–1,304 / 1,038–1,049 GB/s; 17408×5120 1,316 / 1,080; Inkling 2304×2304 341–371 / 317–347. Matrices this small (2.8 MB) are dominated by launch latency and are best fused with neighbouring ops in the real forward pass. Run-to-run spread on the same shape is about ±1.5%.

Tuning knobs (for other GPUs): `ASTRAL_MMA_VARIANT` (e.g. `w4u8t1024`), `ASTRAL_MMA_BLOCKS_PER_SM`, and `ASTRAL_NVCC_DEFINES` for build-time experiments. Performance is a sawtooth in blocks-per-SM because the K split quantises into waves.

**Known limits and next steps:** split-K uses fp32 atomics, so results vary in the last bits between runs (a deterministic mode needs a two-pass reduction); grouped expert dispatch for MoE (Inkling's 128 experts) is not written yet. The fp32-activation v1 stays as the exact variant.

## Open items carried out of P0
- **vLLM/SGLang comparison: decided 2026-11-23: add Qwen3.8-27B as a second test model** (work deferred; owner will pick it up later). Planning notes, unverified until started:
  - Purpose: a model vLLM, SGLang and llama.cpp can all serve on the A100, so "beat vLLM/SGLang" becomes measurable on this GPU. Inkling 2 Mini stays the flagship.
  - Size: BF16 weights (~54 GB at 27B params) do not fit the 40 GB A100, so the comparison needs a 4-bit checkpoint (AWQ/GPTQ, run via Marlin-class kernels in vLLM/SGLang) and the matching 4-bit GGUF for llama.cpp. Which published checkpoints exist, and whether each engine supports this architecture on sm_80, must be checked first.
  - Architecture: hybrid (per the research notes, full attention in 16 of 64 layers, the rest linear attention), so it exercises Astral's typed KV pools differently from Inkling. Needs its own model spec (`docs/spec/06b-model-qwen3.8-27b.md`).
  - First steps when resumed: confirm checkpoints and engine support, download (needs sizes stated), run `bench/baseline_servers.sh` with a Qwen config.
- **HTTP-level Pareto curves for Astral** arrive with the API server in P1 (`bench/baseline_servers.sh` gains an `astral` engine).
- Disk on the dev server: container images use ~129 GB (vLLM latest and nightly, SGLang); 278 GB free. They can be removed if no further incumbent runs are planned on the A100.

## P1: Inkling 2 Mini runs end to end (2026-11-24, H100 PCIe)

- `crates/model` (`astral-model`): the full Inkling 2 Mini forward pass (48 slots, 3 loop passes, attention residuals, loop controllers, short convs, rel-bias attention with sinks, MoE with SITU), mirroring llama.cpp `inkling2` / `inkling2_loop`.
- CLI: `astral verify-model <gguf> <eval_callback.log>` (per-slot l_out and layer-0 intermediates vs llama.cpp) and `astral generate <gguf> <n> <ids...>`.
- Bug fixed: the relative bias and sink are part of the attention mask and are **not** multiplied by 1/sqrt(hd). The logit is `scale*q.k + bias - sink`.
- Correctness: layer-0 Q/K/V/R/sinks match llama.cpp to the printed precision. All 48 slots agree to about 1% relative; the drift is accumulated rounding, because llama.cpp quantises activations to 8 bits and Astral keeps f32. The verify threshold (absolute 2e-2) is too strict for late layers and should become relative. **The 32-token greedy output is identical to llama.cpp's** ("The capital of France is" → " Paris.\n\nA. Noun clause…").
- Speed at batch 1, decode: 55.5 tok/s eager and **65.6 tok/s with a CUDA graph** (`ASTRAL_NO_GRAPH=1` turns it off). For comparison, llama.cpp server gets 44.9 and vLLM 108.3.
- Roofline estimate: about 3.8 GB of weights read per token (looped layers re-read 3x; LM head 0.49 GB), so roughly 1.9 ms per token at 2 TB/s. The current 15.2 ms per token is about 250 GB/s. The next work is GEMV efficiency (small K = 2304, one warp per row), fusing the elementwise ops, and an MoE kernel that handles all 8 experts in one launch.
- Profiler: ncu is at /opt/nvidia/nsight-compute/2025.1.1/ncu; nsys is at .../host/target-linux-x64/nsys.

## P1 decode optimisation (2026-11-24, H100 PCIe, batch 1, Inkling 2 Mini UD-Q4_K_XL)

| Step | Decode tok/s | ms/token |
|---|---|---|
| First correct version (eager) | 55.5 | 18.0 |
| + CUDA graph replay | 65.6 | 15.2 |
| + int8 activations, dp4a GEMV (Q8_0, NVFP4), 16-byte weight loads | 72.9 | 13.7 |
| + single-pass mix, block-per-row F32/BF16 GEMV, wider RMS norm | 94.5 | 10.6 |
| + Q/gate/R/K/V in one launch | 101.0 | 9.9 |
| + mix depth 8 (no register spills), rank-based top-k, stacked gain/forget | 108.2 | 9.2 |
| + exact UE4M3 to f16 scale trick (`x << 7` = value * 2^-8) | 109.3 | 9.1 |
| + fused kernels: attention prep, loop controller (2 launches), RMS norm + quantise, gate + quantise, SITU + quantise, conv + residual, MoE combine + conv + residual; no buffer copies | 135.6 | 7.4 |
| + split-K flash-decode attention (64-position chunks) | **136.6** (131.8 at 2k tokens, 130.7 at 4k) | 7.32 |

- Correctness: every generated token matches llama.cpp greedy over the 32-token golden continuation, except one near-tie (top-2 margin 0.05 logits, position 11) that int8 activations flip. llama.cpp makes the same int8 approximation. Layer-0 intermediates match llama.cpp better than the f32 path did (attention output max diff 1e-4). Selftest: 80/80.
- Comparison with vLLM (bench/results/h100, ISL 512 / OSL 128, concurrency 1): vLLM's 108.3 tok/s is end-to-end and includes TTFT 254 ms. **vLLM's decode TPOT is 7.30 ms; Astral's is 7.32 ms, so decode is level.** Astral has no batched prefill yet (tokens go one at a time, about 3.7 s for 512 tokens), so end to end it would lose clearly today.
- Profile at 7.4 ms: NVFP4 experts about 1.7 ms (about 900 GB/s; integer ALU 47% busy, L1 stalls), Q8_0 about 1.4 ms, then many small kernels.
- Still to do to beat vLLM: (1) batched prefill (GEMM over prompt tokens: tensor-core NVFP4/Q8_0 MMA, causal attention over the prompt); (2) further decode gains (NVFP4 decode ALU, fusing router+top-k and the remaining tiny ops, overlap via PDL); (3) continuous batching for concurrency (vLLM 1,308 tok/s at c32); (4) tokenizer, chat template and the OpenAI-compatible server, then the same `bench/baseline_servers.sh` run.
- Debug: `ASTRAL_TAP_LAYER=<n> astral verify-model ...` taps any layer. `ASTRAL_ATTN_SIMPLE=1` uses the reference attention kernel. `ASTRAL_NO_GRAPH=1` runs eagerly. `astral score <gguf> <ids...>` gives teacher-forced agreement and top-2 margins.

## P1 batched prefill (2026-11-24, H100 PCIe, Inkling 2 Mini UD-Q4_K_XL)

`Session::prefill(tokens)` runs the prompt in chunks of up to 512 tokens (`PREFILL_TMAX`). Decode is the one-token case of the same code: every decode kernel was generalised over T tokens (token-major buffers, a bank of `[rows][tmax][dim]`).
- **GEMMs:** tensor-core int8 `mma.sync` (`astral_gemm_{q8_0,nvfp4}_mma`).
  - Q8_0 uses m16n8k32 (one MMA per 32-value block).
  - NVFP4 uses m16n8k16, with E2M1 decoded to int8 in registers via the byte-permute table.
  - Both rescale in f32 by the weight scale times the activation scale per block.
  - Each warp does 16 rows × 64 tokens.
  - Dense layers use contiguous token items. For MoE, the host groups the (token, expert) pairs by expert after one sync per layer, so each expert reads its weights once per 64 of its tokens.
  - The dp4a grouped kernel (`ASTRAL_PREFILL_DP4A=1`) is kept as a reference.
- **Short convs:** no recurrence (4 taps), so every (token, channel) runs in parallel (`conv_par`, `conv_res_par`). `conv_state` stores the last three inputs afterwards.
- **Loop controller:** 8 tokens per block (`ctrl_t`: rms2 + ctrl_down_t + ctrl_combine_t).
- **Attention:** causal multi-query through the same split-K kernel (grid z = query token).
- **Decode graph:** captured at `Session::new`, so capture costs neither TTFT nor decode.

| Prefill tokens | Time | tok/s |
|---|---|---|
| 128 | 130 ms | 984 |
| 512 | **277 ms** (dp4a grouped GEMM: 562 ms; token by token: about 3.7 s) | 1,849 |
| 1,024 | 683 ms | 1,498 |
| 2,048 | 1,987 ms | 1,031 |

- **Correctness:** `astral verify-model` now also runs the golden prompt as one prefill and checks every slot's `l_out` against llama.cpp. Result: PASS, with the same next token as decode and final logits within 0.13 of llama.cpp (max |d|). Selftest 80/80.
- **Against vLLM** (ISL 512 / OSL 128, c1): vLLM TTFT is 254 ms (measured through HTTP); Astral prefill is 277 ms (in-process, no server yet). Decode after a prefill runs at 127 tok/s (7.87 ms) against vLLM's 7.30 ms. End to end that is about 1.28 s per request against vLLM's 1.18 s (100 vs 108 tok/s): close, not yet ahead.
- **Open issue: decode is about 6% slower once any kernel has been launched eagerly in the context** (prefill, even 1 token, or a warm-up). The graph replays the same kernels with identical durations (nsys), but each node boundary costs about 0.5 µs more: 7.75 ms per step against 7.30 ms. Re-capturing the graph, eager vs lazy module loading, and a separate stream for eager work do not change it. Worth an NVIDIA-side look (driver/graph node launch path), or keeping prefill inside graphs too.
- **Next for prefill:**
  1. GEMM efficiency: shared-memory staging with cp.async and ldmatrix, larger tiles, and grouping on the GPU instead of a host sync per layer. The MMA kernels currently reach roughly 30–60 TOPS of about 1,500 int8.
  2. Fixed overhead at short prompts (128 tokens = 130 ms).
  3. Super-linear growth past 512 tokens: per-query K/V re-reads in attention, and weights re-read for every 512-token chunk (raise `PREFILL_TMAX`, or chunk only attention).

## P1 prefill GEMM and attention optimisation (2026-11-24, H100 PCIe)

| Step | 128 tok | 512 tok | 2,048 tok |
|---|---|---|---|
| Tensor-core int8 MMA, direct global loads (previous) | 130 ms | 277 ms | 1,987 ms |
| + 4-stage cp.async shared-memory pipeline, 64x64 tiles (`*_mma2`) | 73 | 168.5 | 1,216 |
| + controller as f32 SGEMM (split-K down, gain/forget GEMM) | 67 | 143.8 | 1,115 |
| + flash-attention prefill on tensor cores (`astral_ink_attn_prefill`) | 65.6 | 126.5 | 937 |
| + 2,048-token chunks (flash needs no split workspace) | 65.5 | 126.9 | 385 |
| + weights dequantised to f16 in registers, f16 activations, f16 MMA (`*_mma3`) | 57.6 | 108.4 | 326.7 |
| + MoE grouping on the GPU (no host sync per layer) | 54.9 | 102.7 | 322 |
| + split-K for small grids | **51.7** | **103.6** | **323** |

1,024 tokens: 175.5 ms; 4,096: 674 ms (6,076 tok/s, linear).

- **Exact f16 dequant tricks:**
  - An E2M1 nibble n as f16 bits `((n & 8) << 12) | ((n & 7) << 9)` equals value × 2^-14 (subnormals included).
  - A canonical UE4M3 byte as f16 bits `x << 7` equals value × 2^-8.
  - So the NVFP4 weight is exact in f16. Q8_0 int8 uses the 1024-bias trick, with one f16 rounding of int8 × scale.
  - Activations in f16 are more accurate than the int8 they replace. Decode still uses int8 dp4a.
- **Correctness:** verify-model PASS, with the same next token from prefill and decode and final logits within 0.15 of llama.cpp. Selftest 80/80. Non-determinism note: GPU MoE grouping (atomic order within an expert) and split-K atomics change low-order bits run to run (about 1e-5 relative in layer sums).
- **Against vLLM** (ISL 512): vLLM TTFT 254 ms (end-to-end through its server). **Astral prefill: 103.6 ms** (in-process). Decode after prefill is still about 127 tok/s (the open eager-launch slowdown) against vLLM's 7.30 ms TPOT. Estimated per request (512/128): about 104 + 127 × 7.87 = 1.10 s against vLLM's 1.18 s.
- **Where 128-token prefill goes now:** 52 of its 55 ms is GPU-busy, mostly NVFP4 experts (about 25 ms). Every expert's weights are read for only about 8 tokens each, about 25 GB in total, whose bandwidth floor is about 12.6 ms. Dequant ALU work is the other limit.
- **Formats on the H100:** it has no FP4 tensor cores (Blackwell-only), so NVFP4/MXFP4 are decoded in software to int8 (decode) or f16 (prefill) and run on the int8/f16 tensor cores. The 4-bit storage still halves memory traffic. The official Inkling 2 Mini checkpoint is itself NVFP4 + MXFP8 (`hf_quant_config.json`, modelopt NVFP4), so it has no higher-precision source to make an MXFP4 version from.

## Qwen3.8-27B MXFP4 (2026-11-24, H100 PCIe)

`crates/model/src/qwen35.rs`, `cu/qwen.cu`, `crates/kernels/src/qwen.rs`, `crates/loader/src/safetensors.rs`.
Run with `astral qwen-generate <checkpoint dir> <n> <token ids...>`.

- **Architecture** (`qwen3_5` text): 64 layers.
  - 48 Gated DeltaNet layers: causal conv + SiLU, L2-normed q/k, a gated delta-rule recurrence with the f32 state in registers (one warp per 32 state columns), and a gated RMSNorm.
  - 16 gated full-attention layers: 24/4 heads of 256, q/k norms, NeoX RoPE on 64 dims with θ 1e7, sigmoid output gate.
  - Dense SwiGLU MLP. RMSNorm weights are zero-centred (`1 + w`) except the gated norm's.
- **Weights:** compressed-tensors `mxfp4-pack-quantized` (E8M0 per 32, value 2i in the low nibble) is converted at load time to the NVFP4 sub-block layout. Each block becomes two sub-blocks with an exact power-of-two UE4M3 scale, plus a per-matrix 2^offset in `mat_scale` (`astral_quant::mxfp4_ct_to_nvfp4`, unit test bit-exact against the NVFP4 decoder). 25 blocks out of the whole model fell below a matrix's 18-binade window and were re-encoded. Matrices with the same input are stacked at load (qkv+z, q+k+v, gate+up, a+b) so each group is one launch. Embedding and LM head stay BF16. Load: 18.8 GB in about 35 s.
- **Correctness:** the 32-token greedy continuation of "The capital of France is" is identical to vLLM's (same checkpoint, `vllm/vllm-openai:latest`), for both decode and prefill.
- **Decode (c1):** 77.3–78.0 tok/s (12.8 ms) against vLLM's TPOT of 12.72 ms (78.6 tok/s). Effectively level.
  - New shared-memory NVFP4 GEMV (`astral_gemv_nvfp4_s{1,2,4}`): activations staged in smem once per block, weights streamed with `ld.global.nc.L1::no_allocate`, one row per warp. Qwen went from 68.9 to 77.5 tok/s; Inkling is unchanged (about 127).
  - Gate+up reach 1.71 TB/s. Remaining per-step costs: the down projection at about 1.3 TB/s, the BF16 LM head (1.3 ms, at bandwidth), and RMS norms (0.8 ms, single-block).
- **Prefill:** 512 tokens in 293 ms against vLLM's 110 ms TTFT. **Not competitive yet.**
  - The GEMMs (v3, f16 dequant in registers) run at about 115 TFLOPS, about 15% of peak, and are issue-bound on dequant and fragment-load instructions.
  - The DeltaNet recurrence is sequential over tokens (about 60 ms per 512).
  - A big-tile v4 (`ASTRAL_MMA4=1`, 128 × 128, ldmatrix.x4, swizzled B) is still ALU-bound and slower (344 ms), so it is off by default.
  - Next: Marlin-style offline weight permutation into MMA-fragment order with LOP3 dequant, and a chunked delta-rule (WY/UT form) for prefill.

### Qwen3.8 prefill optimisation (2026-11-24)

| Step | 512 tok | 2,048 tok |
|---|---|---|
| Start (v3 GEMM, sequential delta rule, split-K attention, 512-token chunks) | 293 ms | 996 ms |
| Recurrence: each state column over 4 lanes | 287 | |
| GEMM v5 on a load-time fragment layout (`astral_quant::nvfp4_frag_layout`): 128 × 128 block, 16 rows × 128 tokens per warp, 2–3 bit ops + one `hmul2` per f16 weight pair, ldmatrix.x4 B | 199 | |
| Split-K for small v5 grids; 16-token staging in the recurrence | 188.7 | 996 |
| Chunked gated delta rule (`astral_qw_gdn_chunk`, WY form, 64-token chunks, register-tiled, 203 KB dynamic shared memory) | 176.6 | 965 |
| Flash-attention prefill for head_dim 256 (`astral_qw_attn_prefill`, tensor cores, no workspace) + 2,048-token chunks | **167.5** | **668** |

128 tokens: 48.6 ms; 1,024: 332 ms (linear).

- **Exactness:** e2m1 × 2^-14 × UE4M3 is exact in f16: every product is an integer multiple of 2^-24 and far below the f16 maximum. So the v5 dequant is one `hmul2` per pair, with 2^14 applied once to the f32 result. The chunked delta rule is an exact rearrangement of the recurrence (decay ratios as exp(G_t − G_s), never divided). Its tokens match the sequential kernel (197-token prompt), the flash kernel's tokens match split-K, and the 32-token output still equals vLLM's.
- **New GPU capability:** `Function::set_max_dynamic_shared` (cuFuncSetAttribute) for kernels above 48 KB of shared memory.
- **Memory:** the fragment layout keeps a second copy of every dense NVFP4 matrix (Qwen loads 32.5 GB instead of 18.8). `ASTRAL_NO_FRAG=1` drops it and falls back to the v3 GEMM.
- **Against vLLM** (512 tokens): Astral 167.5 ms against vLLM's 110 ms TTFT. Still 1.5× behind; it was 2.7×.
  - GEMMs are about 140 ms: every shape runs at 200–220 TFLOPS (about 28% of f16 peak), with the tensor pipe 47% active, 3.8 warps per scheduler, and stalls spread across fixed-latency, shared-memory, math-throttle and barrier.
  - The chunked delta rule takes about 24 ms (62 µs per 64-token chunk per layer).
  - Next: Hopper `wgmma` (async warpgroup MMA from shared memory) for the prefill GEMM, and tensor cores (tf32) inside the chunked delta rule.
- Inkling is unaffected: verify PASS, 512-token prefill 102 ms, decode 126.8 tok/s.

## Decode faster than vLLM on both models; wgmma prefill GEMM (2026-11-24, H100 PCIe)

| Decode, c1 (after a real prefill) | Astral | vLLM | |
|---|---|---|---|
| Inkling 2 Mini UD-Q4_K_XL | **142.4 tok/s (7.02 ms)** | 137 tok/s (7.30 ms TPOT) | +4% |
| Qwen3.8-27B MXFP4 | **84.8–85.0 tok/s (11.8 ms)** | 78.6 tok/s (12.72 ms) | +8% |

**What changed for decode:**
1. **Decode attention:** split-K with 4 keys per warp iteration as independent chains (interleaved loads, dots and shuffle reductions, one online-softmax update per group). Inkling attention per step went from 205→834 µs over positions 8→68 to flat. This applies to both models.
2. **The "after prefill" slowdown is solved by never launching eagerly.** Measured on this driver: any burst of about 100 eager operations (kernels or memsets; one or two are fine) leaves every later CUDA-graph replay about 6% slower for the rest of the process. The per-kernel durations are identical; the gaps between graph nodes grow.
   - Ruled out: carveout, lazy module loading, device max connections, clocks (1,755 MHz in both), a separate stream, re-capture, and the order of graph vs eager.
   - Fix: prefill is captured and replayed as a one-off graph, and `reset()` is a pre-captured graph. `ASTRAL_PREFILL_EAGER=1` restores eager prefill.
   - Graph capture adds about 3 ms to a 512-token prefill.
3. **Native MXFP4 decode matvec for Qwen** (`astral_gemv_mx4_s1`): the checkpoint's own E8M0 scales (1 byte per 32 values) instead of the converted UE4M3 (1 per 16). That is about 5.6% fewer bytes and one scale per 16-byte load, exact to the checkpoint (no re-encoded blocks on the decode path). 81.1 → 84.1 tok/s.
4. **Qwen residual add fused into the next RMS norm + quantise** (`astral_qw_add_rms_q8`): 84.1 → 85.0.

**Prefill GEMM on Hopper `wgmma`** (`astral_gemm_nvfp4_wgmma{,2}`, sm_90a; v5 elsewhere):
- A stays the exact f16 dequant in registers, which is also wgmma's register-A layout.
- B (f16 activations) is read by the tensor cores straight from the v5 swizzled tile, which is exactly the 128B-swizzle K-major layout, through a descriptor.
- One `m64n128k16` per warpgroup per 16-column step.
- Correct on the first run.

| Qwen prefill | before (v5) | wgmma |
|---|---|---|
| 512 tok | 169.6 ms | **135.7 ms** |
| 2,048 tok | 670.7 ms | **531.6 ms** |

- **GEMMs:** about 88 ms of the 512-token prefill (gate+up about 296 TFLOPS). The pipelined variant (3 stages, 1 group in flight, `wgmma2`, default) measures the same as the simple one.
- **Rest of the 512-token prefill:** the chunked delta rule (24 ms, f32 SIMT, shared-memory bound at about 10% of f32 peak), SwiGLU (5 ms) and the a/b projection (4 ms, now a split-K f32 SGEMM).
- **Two-phase delta rule** (`ASTRAL_GDN_TWOPASS=1`: parallel prep + sequential scan): correct but slower, because the state-dependent matmuls dominate and need 120 KB of shared memory per block. Making them fast needs tf32 tensor cores, which would trade away the exact-f32 state.
- **Against vLLM** (512 tokens): Astral 135.7 ms against vLLM's 110 ms TTFT. Still about 23% behind on Qwen prefill. Inkling prefill: 104 ms against vLLM's 254 ms.

## Decode: programmatic dependent launch (2026-11-25, H100 PCIe)

| Decode, c1 after a 512-token prefill | before | now | vLLM |
|---|---|---|---|
| Inkling 2 Mini | 142.4 tok/s | **149.1 tok/s (6.71 ms)** | 137 (+9%) |
| Qwen3.8-27B MXFP4 | 85.0 | **86.7 (11.5 ms)** | 78.6 (+10%) |
| Qwen3.8-27B, `ASTRAL_LM_HEAD_Q8=1` | | **91.1** | (+16%) |

- **PDL:** on compute capability ≥ 9 every kernel launches with `cuLaunchKernelEx` + programmatic stream serialization, and every kernel starts with `griddepcontrol.wait`. Inserted into all 77 entry points; a no-op without PDL, so results are unchanged. A successor is staged while its predecessor drains. Captured graphs keep the programmatic edges. `ASTRAL_PDL=0` disables it.
  - Kernels in the small-kernel files (`inkling.cu`, `qwen.cu`, `p0_basic.cu`) also call `griddepcontrol.launch_dependents` right after the wait, so the successor is scheduled immediately. The GEMV/GEMM files do not (waiting successors would hold SM slots during multi-wave launches).
  - PDL is switched off while prefill graphs are captured (`Context::set_pdl`): it slowed Inkling prefill (104 → 110 ms). `ASTRAL_PREFILL_PDL=1` keeps it on.
- **MXFP4 decode GEMV for short, wide matrices** (N ≤ 8192, K ≥ 4096): each row's K split over 2 warps (`astral_gemv_mx4_w2`): +0.3 tok/s. Splitting over 4 warps everywhere is much slower.
- **`ASTRAL_LM_HEAD_Q8=1` (opt-in):** Qwen's BF16 LM head (2.54 GB, 1.3 ms per token) is quantised to Q8_0 at load: 86.7 → 91.1 tok/s. Greedy output is identical for 192 of 256 tokens, then diverges at a near-tie. Not bit-exact to the checkpoint, hence opt-in.
- **Prefill is unchanged:** Inkling 105.6 ms, Qwen 133.9 ms at 512 tokens.
- **Checks:** selftest 80/80, Inkling verify PASS, Qwen 32-token output equal to vLLM's.

## Inkling speculative decoding with the official DSpark draft (2026-11-25, H100 PCIe)

`astral spec-generate <gguf> <native checkpoint dir> <k> <n> <ids...|@file>` runs greedy generation with the draft that ships with the checkpoint (`mtp.safetensors` + `mtp_config`), verifying `k` drafted tokens per round. It then checks the result against plain decoding.

**What the draft is** (`crates/model/src/dspark.rs`; kernels in `cu/dspark.cu`, launchers in `crates/kernels/src/dspark.rs`):
- A 3-layer dense transformer (hidden 2304, 24 heads / 8 KV heads, 2 sliding-window layers of 1024).
- Each target token leaves four features: the residual stream entering layer visits 5, 23 and 41, and the final-norm output. Every draft layer mixes them with `softmax(fuse_weight)`, normalises them and projects them into its own K/V cache ("context").
- A draft step runs one 8-token block bidirectionally over that context. The block is the anchor (the last accepted token, by its normalised target embedding) plus 7 `mask_embed` rows.
- A recurrent Markov head (`rnn_hidden`, rank 256) then picks the drafts left to right. Its `hidden_out` is folded into the target LM head at load (`fold = lm_head @ hidden_out`, Q8_0, computed on the GPU).
- Matches vLLM's `Inkling2DSparkModel` (greedy). Weights here are Q8_0 (BF16 in vLLM).

**One round = one CUDA graph:**
1. Verify the anchor and `k` drafts in a T = k+1 target pass.
2. Accept the matching prefix plus the target's next token (`astral_dsp_accept`, on the GPU).
3. Roll back the short convs: verify passes save each conv site's inputs instead of advancing the state, and `astral_dsp_conv_commit` applies the accepted prefix. The KV cache needs no rollback.
4. Store the context K/V for the verified rows.
5. Draft the next block.

The host only reads the emitted tokens after each round.

**Small-batch target path (T = 2..8, verify).** The prefill kernels are built for large T. At T = 4 they made a round cost 37 ms; now 11.6 ms:
- the five attention projections (and O, LM head, draft matmuls) use `astral_gemv_q8_0_multi_t{2..8}`, the fused decode GEMV with T tokens per weight read;
- MoE runs on a GPU-built grouped dp4a GEMV (`astral_ink_moe_group_q`), which reads each used expert once;
- attention uses the decode split-K kernel at full capacity, so the graph works at any position;
- the loop controllers use the decode kernels;
- MoE work lists are bounded by the number of picks; empty grouped items exit early.

Draft head: the fold GEMV, logits and argmax are fused into one multi-block pass (`astral_dsp_fold_part` + `astral_dsp_argmax_fin`). Verify uses the same multi-block argmax.

**Correctness:**
- Forced along vLLM's own greedy output (`ASTRAL_SPEC_FORCE=@ids`), Astral's draft gets the same tokens per round as vLLM's, so the port matches:

  | prompt | Astral | vLLM |
  |---|---|---|
  | 0 | 2.92 | 2.92 |
  | 1 | 2.12 | 2.13 |
  | 2 | 2.10 | 2.15 |
  | 3 | 3.02 | 3.10 |

- With int8 activations in verify (`ASTRAL_MMA_INT8=1` on the old path), speculative output equals plain greedy for 128 of 128 tokens.
- Default path, teacher-forced: plain decode run along the speculative tokens disagrees only at near-ties, with logit gaps ≤ 0.42. Batched-vs-decode kernels differ by up to about 1 logit (`astral cmp-prefill`).

**vLLM refuses 3 speculative tokens** for this model: `Inkling2 DSpark requires num_speculative_tokens = 7 (the checkpoint's block size - 1)`. The comparison is therefore against vLLM's official setting, 7 (`--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'`, nightly image, greedy).

**Single-stream decode, tok/s.** Four chat prompts, 512 new tokens, `ignore_eos`. Astral is forced along vLLM's output, so both engines see identical acceptance:

| prompt | vLLM DSpark-7 | Astral DSpark k=3 | Astral DSpark k=7 | Astral plain |
|---|---|---|---|---|
| hash map explainer | 224.8 | 214.7 | 190.6 | 148 |
| short story | 164.2 | 170.5 | 137.0 | 148 |
| French Revolution | 168.7 | 169.4 | 138.7 | 148 |
| Rust CSV parser | 239.5 | 226.2 | 195.5 | 148 |
| **mean** | **199.3** | **195.2** | 165.5 | 148 |

vLLM without speculation: 137.

Free-running (Astral's own greedy text, so acceptance differs per prompt), k=3:

| prompt | 0 | 1 | 2 | 3 | mean |
|---|---|---|---|---|---|
| tok/s | 201.7 | 169.6 | 161.5 | 258.3 | **197.8** |
| tokens/round | 2.33 | 1.96 | 1.87 | 2.99 | |
| speculative token ≠ plain argmax (teacher-forced) | 9 / 500 | 4 / 500 | 10 / 500 | 5 / 500 | |

Every mismatch is a near-tie: the largest logit gap is 0.44.

**Where it stands.**
- Astral with the official draft at k=3 is level with vLLM's official DSpark-7: 2% behind on the mean, ahead on the low-acceptance prompts, behind on the high-acceptance ones.
- k=7 is slower here: the verify pass's MoE grows with T. At T=8, MoE is about 6 ms of the ~15 ms round, already at ~1.8 TB/s per used expert.
- Next levers: cheaper MoE at T ≤ 8, the ~100 small per-visit kernels, and an adaptive k.
- Not done yet: the OpenAI server path for speculative decoding, sampling (temperature > 0, probabilistic acceptance) and batching.

## Faster than vLLM with and without the draft (2026-11-25, H100 PCIe)

Single stream, greedy, `ignore_eos`, the four chat prompts above, 512 new tokens. vLLM is the nightly image on the same machine and the same token ids.

| Inkling 2 Mini | vLLM nightly | Astral | Astral vs vLLM |
|---|---|---|---|
| Plain decode (no draft) | 129.9 tok/s | **172.6** (173.9 / 171.3 / 171.4 / 173.8) | **+33%** |
| DSpark draft, forced along vLLM's own text (same acceptance) | 199.3 (official k=7) | **227.0** (k=3: 248.3 / 196.6 / 197.6 / 265.3) | **+14%** |
| DSpark draft, free-running (Astral's own greedy text) | 199.3 | **227.9** (k=3: 206.1 / 213.6 / 195.8 / 295.9) | **+14%** |

**SGLang cannot be compared on this model.**
- Neither the 0.5.23 release nor the current dev image (`0.0.0.dev1+ge96e96757`) loads Inkling 2 Mini. Both fail with `model type inkling2_loop ... not recognize[d]`.
- SGLang registers `inkling2` (`Inkling2ForConditionalGeneration`) but has no `loop_config` / `num_loop_passes` handling in its code. Its `--inkling2-loop-*` flags exist only in its docs.
- Relabelling the checkpoint would run a different, unlooped model.

**What changed** (each is used by decode and by the T ≤ 8 verify/draft passes; each has an env switch back to the old path):
- **Fused projections:** the attention-sink projection and, on global layers, the K/V conv-tap projections are Q8_0 items of the one fused Q/gate/R/K/V launch (`ASTRAL_SIDE_BF16=1` to split).
- **Merge + gate:** `astral_ink_attn_merge_q8` applies the output gate and quantises for the O projection inside the split-K attention merge.
- **Mix + norm:** `astral_ink_mix_rms_q8` fuses the depth mix with the RMSNorm + quantisation after it, at both mix sites of non-loop visits (`ASTRAL_NO_FUSE_MIX=1`).
- **Router:** `astral_ink_router_fused` does router GEMV + top-k in one kernel; the last block per token selects, via a self-resetting counter. Used for T ≤ 8 (`ASTRAL_NO_FUSE_ROUTER=1`).
- **Batched loop controllers:** the Q, K and V controllers of a looped visit run as two launches plus one batched input norm (`ctrl_down3`, `ctrl_combine3`, `rms_q8_3`), instead of 3×(down, combine, norm). The three chains now run in parallel. This was the largest single gain: decode 156 → 171 tok/s (`ASTRAL_NO_BATCH_CTRL=1`).
- **Tensor-core GEMV:** `astral_gemv_q8_0_multi_mma` is the fused Q8_0 GEMV on int8 tensor cores (`mma.sync m16n8k32`, T ≤ 8 tokens as the n=8 side, one Q8_0 scale block per k-step, K split over 4 warps). It is the default for verify passes, the draft, and decode (171.4 → 174.5 tok/s). `ASTRAL_SMALL_DP4A=1` / `ASTRAL_DECODE_DP4A=1` revert.
- **Conv history:** verify passes write it from the parallel conv kernels themselves; the 192 copy nodes per round are gone.
- **Draft:** base logits only for the k drafted rows; block attention on 16 warps.

**Checks:**
- selftest 80/80.
- `verify-model` PASS against the llama.cpp reference dump.
- Teacher-forced speculative checks at k = 1, 2, 3, 7: the speculative token differs from plain greedy only at near-ties (≤ 0.42 logit).
- 512-token prefill: 97.8–100.5 ms on short repeated prompts (was 105.6). It is about 115 ms on the long chat prompt, whose routing differs. vLLM TTFT for 512 tokens is 254 ms.

## OpenAI-compatible server, against vLLM's Python and Rust frontends (2026-11-25, H100 PCIe)

`astral serve <gguf> --checkpoint <native dir> [--no-draft] [--k 3] [--port 8000] [--ctx 8192]` (crate `astral-api`).

- **Endpoints:** `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions` (SSE streaming, `stream_options.include_usage`, stop strings, `ignore_eos`) and `/tokenize`.
- **Engine:** std threads with no async runtime. One engine thread owns the GPU session and runs requests FIFO from a bounded queue; a disconnected client cancels its request.
- **Chat:** Inkling's chat template ported to Rust. Its token ids equal vLLM's `/tokenize` on all four prompts. Thinking goes to `reasoning_content`, the answer to `content`, and tool calls to `tool_calls`.
- **Limits:** decoding is greedy; `temperature` / `top_p` are accepted and ignored. Image input is rejected. One request runs at a time.

**Attention at large contexts.** Graph-captured passes used to launch one attention split per 64 positions of the whole capacity: 128 splits at context 8,192. At T > 1 that cost the verify pass about 0.5 ms per round, and it also changed the summation order, and with it the free-running text. Verify passes now use at most 32 splits, each looping over chunks (`ASTRAL_ATT_SPLITS`). Decode keeps one chunk per split, which measured about 3% faster per call.

Results: forced k=3 at context 8,192 went from 237.7 to 247.0 tok/s (context 600: 246–249). Plain decode at context 8,192 is 171.4–174.3 tok/s.

**Single stream over HTTP.** Same client for every server (streaming `/v1/completions` on the same token ids, 512 tokens, greedy, `ignore_eos`); context 8,192 everywhere.

| | vLLM nightly, Python frontend | vLLM nightly, Rust frontend (`VLLM_USE_RUST_FRONTEND=1`) | Astral server |
|---|---|---|---|
| No draft | 130.2 tok/s | 130.2 tok/s | **172.0 tok/s (1.32×)** |
| DSpark draft | 199.8 (k=7: 225.2 / 164.8 / 169.2 / 239.9) | 199.8 (225.5 / 164.8 / 169.2 / 239.9) | **224.2 (1.12×, k=3: 208.8 / 202.7 / 191.5 / 293.8)** |
| TTFT, 33–42-token prompts | 42–54 ms | 38–52 ms | 70–78 ms |

- **vLLM's Rust frontend** replaces only its HTTP layer: generation speed is identical to the Python frontend, and time to first token improves by a few ms.
- **Chat needs an extra flag:** with the Rust frontend, chat requests fail unless the tool parser matches the reasoning parser (`--tool-call-parser inkling --enable-auto-tool-choice`).
- **Astral's first token is ~25–35 ms slower than vLLM's.** Small prompts go through the same graph-captured prefill as long ones; not optimised yet.

Correctness: selftest 80/80; `verify-model` PASS. At context 8,192 the free-running speculative output differs from plain greedy (teacher-forced) only at near-ties (≤ 0.34 logit) on all four prompts. The server's chat, streaming and stop-string checks pass with and without the draft.

**SGLang** still cannot run Inkling 2 Mini (no `inkling2_loop` support in 0.5.23 or the dev image), so there is no SGLang number.

## Adaptive draft length; 16-prompt comparison with vLLM and SGLang (2026-11-25, H100 PCIe)

**Adaptive draft length** (`Policy` in `crates/model/src/dspark.rs`; the default).
- `--k` / the `k` argument is now the most drafts per round (default 6 for `serve`), with at least 2 (`ASTRAL_SPEC_KMIN`). There is one round graph per verified length.
- After a fully accepted round the next one verifies one more draft; after a reject before the last draft, one fewer. Predictable text comes in runs.
- A round verifying k drafts only drafts k + 1 tokens for the next one, the most the rule can pick.
- `ASTRAL_SPEC_FIXED=1` verifies `k` every round. `ASTRAL_SPEC_POLICY=model` picks the length from per-position acceptance rates and measured round times; it was measured worse.

**SGLang now runs Inkling 2 Mini** with its dedicated image `lmsysorg/sglang:inkling-2-mini` (`0.0.0.dev1+inkling2`, 2026-11-20). The release and dev images still lack the loop model. Its DSpark uses gamma 7. Startup is about 11 minutes, most of it capturing 58 prefill CUDA graphs.

**16 prompts** (code, prose, explanations, math, lists; `prompts16.json`). Each engine runs on its own greedy text through its server: streaming `/v1/completions` on the chat-template token ids, 512 tokens, `ignore_eos`, one request at a time, context 8,192. Mean decode tok/s:

| | No draft | DSpark draft |
|---|---|---|
| vLLM nightly (Python frontend) | 130.2 | 218.0 (k=7) |
| vLLM nightly (Rust frontend) | 130.4 | 218.2 (k=7) |
| SGLang `inkling-2-mini` | 135.6 | 219.0 (gamma 7) |
| **Astral** (final build, defaults) | **172.2** | **242.0** (adaptive, kmax 6, kmin 2; fixed k=3: 234.6) |
| **Astral vs vLLM Python / vLLM Rust / SGLang** | **1.32× / 1.32× / 1.27×** | **1.11× / 1.11× / 1.10×** |

Final build checks: selftest 80/80, `verify-model` PASS. With the default draft (adaptive, kmax 6, kmin 2, context 8,192) the teacher-forced check on 4 prompts finds the speculative token differing from plain greedy only at near-ties (≤ 0.74 logit). The server's functional checks pass with and without the draft: template tokens equal vLLM's, chat with reasoning split, streaming, stop strings.

Free-running acceptance swings by up to ±15% per prompt, because a near-tie early on changes the whole continuation. Hence 16 prompts; 4-prompt means are not reliable at the 10% level.

### Second prompt set (B) and final correctness check (2026-11-25)

Sixteen different prompts (`promptsB.json`: Go server, RSA, LRU cache, haiku, Dijkstra, …), same method and the final build's defaults. Mean decode tok/s:

| | No draft | DSpark draft |
|---|---|---|
| vLLM nightly | 130.3 | 201.2 (k=7) |
| SGLang `inkling-2-mini` | 135.7 | 192.8 (gamma 7) |
| **Astral** | **171.8** | **224.6** (adaptive) |
| **Astral vs vLLM / SGLang** | **1.32× / 1.27×** | **1.12× / 1.16×** |

- **Per prompt, with the draft,** Astral is faster than vLLM on 14 of 16 prompts (behind on 2 by ≤ 2.3%) and faster than SGLang on 15 of 16 (one tie at 232.6 vs 233.0).
- **Over all 32 prompts (A + B):**
  - With the draft: Astral 233.3, vLLM 209.6, SGLang 205.9, so **1.11× / 1.13×**.
  - Without: Astral 172.0, vLLM 130.3, SGLang 135.7, so **1.32× / 1.27×**.
- **Correctness of the final default draft** (adaptive, kmax 6, kmin 2, context 8,192), teacher-forced on all 16 set-A prompts: the speculative token differs from plain greedy at only 1–22 of 500 positions per prompt, every one a near-tie (largest logit gap 0.95). Batched-vs-one-token kernels differ by up to about 1 logit.
- **Rejected variants:** an NVFP4 draft LM head (`ASTRAL_DRAFT_HEAD_FP4=1`) measured no faster and accepted slightly less. Policy variants (`ASTRAL_SPEC_GROW`, `ASTRAL_SPEC_SHRINK`, kmin 3) were within ±1% of the default on fixed texts.

### vLLM Rust frontend on set B; set-B correctness (2026-11-25)

- **vLLM Rust frontend, set B:** 130.4 tok/s without a draft and 201.2 with DSpark-7, identical to the Python frontend. Over 32 prompts it averages 130.4 / 209.7, so Astral is 1.32× / 1.11× against it.
- **Teacher-forced correctness of the default draft on all 16 set-B prompts:** the speculative token differs from plain greedy at only 1–14 of 500 positions per prompt, all near-ties (largest logit gap 0.58).
- **Charts:** `bench/charts/astral_vs_vllm_{python,rust}.png` and their `_table.png` result tables, generated by `bench/charts/make_charts.py`. Each compares Astral, SGLang and one vLLM frontend, without and with MTP, over 32 prompts.
