# Benchmarks: Inkling 2 Mini on RTX 4090

Single-stream decode on one RTX 4090 (PCIe Gen3 x16, driver 610.57.04), 2026-11-26.

## Method

- Model: unsloth `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf` for both engines. Draft:
  Astral uses the native `mtp.safetensors`; llama.cpp uses `dspark-Inkling-2-Mini-Q8_0.gguf`.
- Prompts: the 32 prompts of `bench/serving/prompts_set_a.json` and `prompts_set_b.json`,
  rendered with the chat template by Astral's `/tokenize` and sent as the same token ids to
  both servers (the tokenizer check shows llama.cpp produces the same ids).
- Requests: streaming `/v1/completions`, 512 tokens, greedy, `ignore_eos`, one at a time,
  context 8,192. One unrecorded warm-up, then 5 repetitions of all 32 prompts per configuration.
- Decode tok/s is `(tokens − 1) / (last chunk − first chunk)`; per prompt the median over the
  5 repetitions, then the mean over prompts. Paired speedups are equal-weight geometric means
  with a 95% case bootstrap ([`paired_stats.py`](../../bench/llamacpp_compare/paired_stats.py)).
- llama.cpp b12751 (`532d674`), `llama-server -ngl 99 -fa on -c 8192 -np 1`; DSpark with
  `--spec-type draft-dspark --spec-draft-n-max 3` or `6`.
- Memory clock 10,251 MHz for almost all samples; GPU temperature at most 74 °C.

## Results

| Configuration | Decode tok/s | Range over prompts | TTFT p50 | ITL p50 / p95 |
|---|---|---|---|---|
| **Astral, no draft** | **143.1** | 143.0–143.3 | 90 ms | 7.01 / 7.05 ms |
| **Astral, DSpark** (adaptive, k ≤ 6) | **173.9** | 141.0–270.7 | 99 ms | 0.21 / 14.4 ms |
| llama.cpp, no draft | 74.7 | 74.6–74.9 | 197 ms | 13.2 / 13.4 ms |
| llama.cpp, DSpark k = 3 | 90.5 | 75.8–120.9 | 176 ms | 0.07 / 26.3 ms |
| llama.cpp, DSpark k = 6 | 86.3 | 65.1–135.8 | 180 ms | 0.05 / 33.1 ms |

| Paired comparison | Speedup (95% CI) |
|---|---|
| Astral vs llama.cpp, no draft | 1.92× (1.914–1.916) |
| Astral DSpark vs llama.cpp DSpark k = 3 | 1.91× (1.87–1.95) |
| Astral DSpark vs Astral no draft | 1.20× (1.14–1.27) |
| llama.cpp DSpark k = 3 vs llama.cpp no draft | 1.20× (1.15–1.26) |

### Update, 2026-11-27: decode projections on dp4a

Same method and prompts; Astral at fork `9593eb4` plus the sm_89 dp4a default for the fused
attention-input projections of decode steps. llama.cpp was not re-run (same binary and file).

| Configuration | Decode tok/s | Range over prompts | TTFT p50 | ITL p50 / p95 |
|---|---|---|---|---|
| **Astral, no draft** | **156.6** (was 143.1) | 156.5–156.6 | 90 ms | 6.41 / 6.45 ms |
| **Astral, DSpark** (adaptive, k ≤ 6) | **172.4** (was 173.9) | 142.2–248.9 | 95 ms | 0.10 / 14.5 ms |

Astral no draft vs llama.cpp no draft: 2.10×. Astral DSpark vs Astral no draft: 1.09× (95% CI
1.03–1.15), down from 1.20×: draft and verify passes are unchanged, but plain decode got faster
and the greedy text changed at near-ties, which moves the acceptance per prompt.

With a draft, several tokens arrive in one chunk, so the ITL percentiles mix near-zero gaps
with whole verification rounds. The DSpark gain here (1.20×) is below the 1.36× reported for
H100 upstream; acceptance is low on the reasoning-heavy openings of these prompts (14–49% of
drafts accepted in a 16-prompt `spec-generate` run). That gap is not analysed yet.
