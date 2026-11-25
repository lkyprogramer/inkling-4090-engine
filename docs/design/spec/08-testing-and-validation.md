# 08 — Testing and validation (A100-only reality)

Requirements: R-VA-1..2. Crate: `testing`.

## 1. The constraint
Only one A100 (sm_80) is available. Consequences:
1. **Only kernel family F80 and the GGUF path can be validated locally.** F90/F100/F120 kernels, native NVFP4/MXFP8, FP8 KV, NVLink/NCCL multi-GPU and unified-memory (GB10) behaviour cannot.
2. Therefore the spec uses a **validation ladder**: each feature has a status in code, docs and release notes (R-VA-2):
   - `VALIDATED(arch, date, hw)` — run on real silicon with the test below.
   - `STATIC` — compiled and disassembly-checked only.
   - `LOGIC` — exercised only on the mock backend.
   - `SPECIFIED` — design only.
3. **No performance or correctness claim** is published for a non-`VALIDATED` arch. Published numbers must be reproducible by third parties.

## 2. Test layers
| Layer | What | Runs on | Catches |
|---|---|---|---|
| L0 unit/property | `core`, `mem`, `kvcache`, `sched` with proptest-style invariants (no leak, no double-free, budget never exceeded, prefix hash stability) | CPU | logic bugs |
| L1 mock engine | whole engine on `MockBackend` with tiny synthetic model | CPU CI | scheduler/API/streaming regressions |
| L2 kernel reference | each kernel vs `quant::dequant_ref()` / CPU reference on random + real tensors | A100 (F80) | numerics |
| L3 model golden | per-layer activations vs a reference (llama.cpp dumps for the same GGUF; PyTorch/Transformers reference for native weights where runnable on A100 in BF16 as a *separate* check `[I]`) | A100 | wiring, loop, MoE routing |
| L4 end-to-end quality | perplexity, KLD, top-token agreement vs llama.cpp on same file | A100 | accuracy parity |
| L5 serving/soak | OpenAI conformance, concurrency soak, memory-leak soak, cancel/disconnect storms | A100 | ops bugs |
| L6 static arch gate | cross-compile all arches; `cuobjdump -sass` asserts characteristic instructions; fatbin contains every declared arch | CPU+nvcc (no GPU) | silent arch drops (e.g. `-arch=sm_121a` issue [R]) |
| L7 remote-silicon | scripted run of L2–L4 on rented/borrowed H100, B200, RTX PRO 6000, GB10 | rented | everything F80 can't see |

## 3. Benchmark harness (R-VA-1, P0 deliverable)
Principles drawn from the InferenceX critique [R]: it uses random data, disables prefix caching, depends on SemiAnalysis's cost model, and gave NVIDIA early config access. Astral's harness:
- **Reports Pareto curves** (throughput per GPU vs tokens/s/user), not single points.
- **Prefix caching on and off** both reported; realistic traffic profiles (chat multi-turn, agentic tool loops, long-prompt/short-output, short-prompt/long-output) plus a random baseline for comparability.
- **Pinned baselines:** exact commit/version, flags and container digest recorded; baselines run on the *same A100*. For GGUF like-for-like use **llama.cpp (mainline build ≥ b12611, same `Mainline_Support` file)** [G]; vLLM/SGLang run their own supported formats and are reported separately and clearly labelled not-like-for-like.
- **Fixed-seed, fixed-corpus, N≥5 runs, report median and spread;** warm-up excluded and reported; GPU clocks/power state logged (low-power-state GPUs seen in community reports [R]).
- Records hardware, driver, CUDA, git SHAs; emits a machine-readable result file and a human report. Every number in docs links to a result file.
- Metrics: TTFT (both definitions, spec 07), ITL percentiles, output tok/s, total tok/s, max concurrency at SLO, GPU memory, cold start (process start → first token), cost/M tokens from measured throughput and a stated $/hr.
- Cost method [R]: `$/M tokens = ($/hr ÷ (tok/s × 3600)) × 10^6`; show utilisation sensitivity (÷utilisation).

## 4. Correctness thresholds (set at P1 start; placeholders here)
- **Kernels (L2):** integer paths bit-exact; float paths ≤ stated ULP. 
- **Model (L3/L4):** thresholds are set empirically from a **llama.cpp-vs-llama.cpp noise floor** (different batch sizes/threads on the same file) so Astral is not held to a tighter standard than the reference holds itself. Reference points from Unsloth [G]: UD-Q8_K_XL perplexity 7.651 (root) / 7.653 (mainline); Q4_K_XL KLD vs Q8 = 0.0107.
- **Determinism:** greedy first-128-token identity vs llama.cpp on prompts where llama.cpp is self-consistent.

## 5. Requirement → test map (phase gates, SPEC §6)
| Requirement | Test | Layer | Phase |
|---|---|---|---|
| R-HW-1,2 | `astral hw` output snapshot on A100; table-driven tests of dispatch with synthetic `Capabilities` for all 6 chips (ensures no `major>=N` branch) | L0 (+A100) | P0 |
| R-HW-3 | UMA-mode accounting unit tests with fake `/proc/meminfo`; **on-GB10 check is L7** | L0 / L7 | P0 / P3 |
| R-HW-4 | microbench variance test on A100 | L5 | P1 |
| R-HW-5 | static arch gate | L6 | P3 |
| R-RT-1 | `cargo geiger`-style check: `unsafe` only in `gpu` (and allow-listed FFI) | L0 | P0 |
| R-RT-2 | engine-thread lint (no `.await`/blocking channel misuse) + latency soak with CPU contention | L1/L5 | P0 |
| R-RT-3 | full engine run on mock in CI | L1 | P0 |
| R-RT-4 | graph replay equals eager outputs for all buckets; memory-stable across 10k steps | L3/L5 | P1 |
| R-KQ-1 | every GGML type found in headers: kernel vs reference | L2 | P1 |
| R-KQ-2 | MoE routing/dispatch vs reference on real expert tensors | L2/L3 | P1 |
| R-KQ-3 | attention (SWA/global, relative-position bias, gate, clip, log-scale τ) vs a CPU reference transcribed from llama.cpp `inkling2.cpp`, incl. window boundary and long-context cases | L2 | P1 |
| R-KQ-6 | attn-res `mix`, loop controllers, 4-site sconv (incl. state carry across chunk/decode boundaries and speculative rollback) vs CPU reference and per-layer llama.cpp dumps | L2/L3 | P1 |
| R-KQ-4 | bandwidth-utilisation microbench for decode GEMV and end-to-end decode | L5/bench | P2 |
| R-KQ-5 | native FP4 on sm_100/120/121 **plus Marlin-style fallback** | L7 | P3 |
| R-KV-1,2 | no-OOM at max admitted concurrency over long soak; budget vs measured within tolerance | L0/L5 | P1 |
| R-KV-3 | prefix hit-rate and correctness (cached vs uncached output identical) | L1/L3 | P2 |
| R-KV-4 | KV-INT8 accuracy delta vs BF16 KV; FP8 KV only L7 | L4 / L7 | P2 / P3 |
| R-SC-1,2 | stall-free invariants (decode never starved); overlap does not reorder outputs | L1/L5 | P1 |
| R-SC-3 | speculative output identical to non-speculative under greedy; acceptance-rate vs batch-size curve | L3/bench | P2 |
| R-SC-4 | quota/priority behaviour tests (Enterprise edition) | L1 | — |
| R-MD-1,2 | model golden + L4 quality | L3/L4 | P1 |
| R-MD-3 | vision golden vs llama.cpp mmproj path | L3 | P2 |
| R-API-1 | **OpenAI SDK conformance** (spec 07 §1): official Python and JS SDKs sync/async/streaming/error paths, schema validation against OpenAI's OpenAPI spec, recorded-fixture comparison, cancel-on-disconnect, public compatibility matrix | L5 | P1 |
| R-API-2 | parser conformance against recorded template outputs | L5 | P2 |
| R-API-3 | metrics name/type snapshot vs vLLM docs; OTel schema validation | L5 | P1 |
| R-VA-1,2 | harness self-test reproduces pinned llama.cpp numbers ±noise; status labels linted in docs/release notes | L5 | P0 |

## 6. Remote-silicon plan (L7, P3 gates)
Priority order: (1) RTX PRO 6000 (sm_120), (2) GB10 (sm_121; also the only unified-memory case), (3) H100 (sm_90a; widely available rental), (4) B200 (sm_100a), (5) GB300 (sm_103a; hardest to access). Cost sketch (reported rental prices [R], estimates): H100 ≈ $2.82/h, B200 ≈ $5.93/h (neocloud, Oct 2026 via secondary source) — a full validation sweep per arch is hours, not weeks, **if** L6 and L2–L4 are fully scripted beforehand. Budget a line item; do not plan P3 on "when we get access".

Rules: a kernel family does not merge as default until it has a `VALIDATED` record; scripts are idempotent and emit the same result files as local runs; every L7 run archives logs, SASS and result JSON.

## 7. Fuzz/robustness
Malformed GGUF headers (bounds/overflow on tensor offsets — loader is a parsing attack surface), pathological prompts (huge, empty, invalid UTF-8), cancel storms, OOM injection via the mock allocator.

## 8. Known gaps
No determinism/eval harness details beyond perplexity/KLD (lm-eval integration not researched [R]); Nsight/CUPTI profiling workflow not researched; no independent baseline for any Rust engine exists to compare against [R].
