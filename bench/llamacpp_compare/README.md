# llama.cpp comparison tools

Scripts used to validate the RTX 4090 port against llama.cpp on the same GGUF file, and to
benchmark single-stream serving. They need Python 3 with `numpy` (and `tokenizers` for
`tokenizer_parity.py`); nothing here is needed to build or run Astral.

Reference used for the published results: llama.cpp `532d67426bb874f06a8acc6959af81b34206f7f9`
(b12751), built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`, and the unsloth
`Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf` (SHA256 `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`).

| Script | What it does |
|---|---|
| [`kld_compare.py`](kld_compare.py) | KL(P‖Q), top-1 agreement (overall and where the reference top-2 margin ≥ 0.5) and perplexity, from two `--kl-divergence-base` files |
| [`near_top_diff.py`](near_top_diff.py) | Per-position max log-prob difference among tokens with log p > −16 on both sides |
| [`tokenizer_parity.py`](tokenizer_parity.py) | `llama-tokenize` against Hugging Face `tokenizer.json` on mixed-script text |
| [`chat_prompt_ids.py`](chat_prompt_ids.py) | Chat-template token ids for the prompt sets, from Astral's `/tokenize` |
| [`greedy_reference.py`](greedy_reference.py) | Greedy continuations, raw top-1 and top-2 margins from `llama-server` |
| [`teacher_forced_verdicts.py`](teacher_forced_verdicts.py) | V5/V6/V7 verdicts from `astral score` and `astral cmp-prefill`; non-zero exit on failure or missing data |
| [`logit_compare.py`](logit_compare.py) | One calculator for raw-logit pairs: reference-side window, probability-weighted and max log-prob differences, full-vocabulary KL, argmax gaps. `judge` (num-2) applies absolute limits from `calibrate`; `judge-paired` (num-3) compares a candidate with a reference control on the same positions; both check completeness and set exit codes |
| [`final_logits.cpp`](final_logits.cpp) | llama.cpp side of `astral dump-final-logits`: raw logits after chosen prefix lengths, one token per decode (`single`) or one batch (`batch`) |
| [`stream_bench.py`](stream_bench.py) | Streaming `/v1/completions` benchmark: TTFT, inter-chunk latency, decode tok/s |
| [`stream_report.py`](stream_report.py) | Per-configuration summary and paired CSVs |
| [`paired_stats.py`](paired_stats.py) | Case-weighted paired bootstrap of per-prompt latencies |

## Workflow

```bash
M=Inkling-2-Mini-UD-Q4_K_XL.gguf; LL=/path/to/llama.cpp; B=$LL/build/bin

# 1. Reference data: perplexity, KLD base (c512 x 8 chunks), per-op log for verify-model
bench/golden_llamacpp.sh $M $LL out/            # writes out/<stamp>-golden/
G=$(ls -d out/*-golden | tail -1)
astral verify-model $M $G/eval_callback.log

# 2. The reference's own noise: the same run with single-token ubatches
$B/llama-perplexity -m $M -ngl 99 -f $G/corpus.txt -c 512 --chunks 8 -ub 1 --kl-divergence-base ub1.bin
python3 kld_compare.py $G/kld_base_c512x8.bin ub1.bin > n0.json

# 3. Astral's decode path on the same chunks
astral dump-logits $M $G/kld_base_c512x8.bin astral.bin
# serving path instead: prefill each chunk's first half, decode the scored positions
ASTRAL_DUMP_PREFILL=1 astral dump-logits $M $G/kld_base_c512x8.bin astral_serve.bin
python3 kld_compare.py $G/kld_base_c512x8.bin astral.bin > astral.json

# 4. Greedy / teacher-forced checks on chat prompts (Astral server running for /tokenize)
python3 chat_prompt_ids.py http://127.0.0.1:8000 ids.json ../serving/prompts_set_a.json ../serving/prompts_set_b.json
python3 greedy_reference.py http://127.0.0.1:8080 ids.json seq/ 500      # llama-server
for f in seq/*.ids; do i=$(basename $f .ids)
  astral score $M $(cat $f) > seq/score_$i.txt
  astral cmp-prefill $M $(cat $f) > seq/cmp_$i.txt
done
python3 teacher_forced_verdicts.py seq/

# 5. Serving benchmark (one server at a time, same ids for every engine)
python3 stream_bench.py http://127.0.0.1:8000 ids.json astral-ar 5 runs/astral-ar.jsonl '{"speculative": false}'
python3 stream_report.py runs/
python3 paired_stats.py --input runs/pair_llama-ar__astral-ar.csv --output runs/ar.json
```

## Same-units comparison of batched and token-by-token logits

`astral dump-final-logits` and `final_logits` write raw f32 logits for the same prefixes, so
both engines go through one calculator. A manifest line is `name<TAB>L1,L2,...<TAB>ids...`.

```bash
g++ -O2 -std=c++17 -I$LL/include -I$LL/ggml/include final_logits.cpp -L$B -lllama -Wl,-rpath,$B -o final_logits
./final_logits $M manifest.tsv ll single; ./final_logits $M manifest.tsv ll batch
astral dump-final-logits $M manifest.tsv as      # kinds step, prefill, decode (prefill of the first length, then steps)
# DSpark verify rows (k drafts, k + 1 rows per pass), teacher-forced along each sequence
ASTRAL_SPEC_FIXED=1 astral dump-verify-logits $M <native checkpoint> 3 manifest.tsv as   # kind verify3
# Reference shape control first; freeze the rule before looking at the candidate.
python3 logit_compare.py compare manifest.tsv ll:single ll:batch ref_shape.jsonl
python3 logit_compare.py calibrate ref_shape.jsonl rule.json --factor 2 --margin 0.5
python3 logit_compare.py compare manifest.tsv as:step as:prefill astral_shape.jsonl
python3 logit_compare.py judge manifest.tsv astral_shape.jsonl rule.json    # exit 0 / 1 / 2
# num-3: ratios against the reference control on the same positions
python3 logit_compare.py judge-paired manifest.tsv astral_shape.jsonl ref_shape.jsonl rule_num3.json
python3 -m unittest test_logit_compare                                      # offline tests
```

## Caveats

- `kld_compare.py` and `near_top_diff.py` read the u16-encoded log-softmax of the llama.cpp file
  format on both sides. `astral cmp-prefill` prints raw-logit differences over a step-side
  window and a float64 full-vocabulary KL. The two sets of numbers are **not in the same
  units**; use `logit_compare.py` for anything that is judged.
- `greedy_reference.py` runs with `ignore_eos`; compare against `ref_top1` (raw argmax), not
  the sampled token.
- Inkling 2's chat template has no generation prompt: the model emits `<|message_model|>`
  itself, so the ids from `/tokenize` end at `<|end_message|>`.
