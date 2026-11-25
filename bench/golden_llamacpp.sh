#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Synthetic Minds LTD
#
# Golden reference data from llama.cpp (spec 08 L3/L4), on the same GGUF Astral will run:
#   - perplexity on a fixed local corpus (no download: built from text in the llama.cpp
#     checkout, so its hash is reproducible from the recorded commit)
#   - full log-prob dump (`--kl-divergence-base`) for later KLD / top-token comparison
#   - per-op tensor statistics for a fixed prompt (`llama-eval-callback`)
#
# Note: the corpus differs from whatever Unsloth used for its reported 7.65, so the
# perplexity value is only comparable between engines on this corpus.
#
# Usage: bench/golden_llamacpp.sh <model.gguf> <llama.cpp dir> [out root]
set -euo pipefail

MODEL=${1:?model.gguf}
LLAMA=${2:?llama.cpp checkout}
OUT_ROOT=${3:-$HOME/astral-work/results}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$OUT_ROOT/$STAMP-golden"
mkdir -p "$OUT"
BIN="$LLAMA/build/bin"

echo "== building tools"
cmake --build "$LLAMA/build" -j "$(nproc)" --target llama-perplexity llama-eval-callback >/dev/null

echo "== corpus (deterministic, from llama.cpp docs)"
CORPUS="$OUT/corpus.txt"
( cd "$LLAMA" && git ls-files 'README.md' 'docs/*.md' 'docs/**/*.md' | LC_ALL=C sort | xargs cat ) > "$CORPUS"
{
  echo "{"
  echo "  \"timestamp_utc\": \"$STAMP\","
  echo "  \"llama_cpp_commit\": \"$(git -C "$LLAMA" rev-parse HEAD)\","
  echo "  \"model_sha256\": \"$(sha256sum "$MODEL" | cut -d' ' -f1)\","
  echo "  \"corpus_sha256\": \"$(sha256sum "$CORPUS" | cut -d' ' -f1)\","
  echo "  \"corpus_bytes\": $(stat -c %s "$CORPUS")"
  echo "}"
} > "$OUT/env.json"

echo "== perplexity (ctx 2048, 16 chunks)"
"$BIN/llama-perplexity" -m "$MODEL" -ngl 99 -f "$CORPUS" -c 2048 --chunks 16 > "$OUT/perplexity.log" 2>&1
grep -E "Final estimate|nan" "$OUT/perplexity.log" | tail -3 || true

echo "== log-prob base for KLD (ctx 512, 8 chunks)"
"$BIN/llama-perplexity" -m "$MODEL" -ngl 99 -f "$CORPUS" -c 512 --chunks 8 \
  --kl-divergence-base "$OUT/kld_base_c512x8.bin" > "$OUT/kld_base.log" 2>&1
ls -l "$OUT/kld_base_c512x8.bin"

echo "== per-op tensor statistics, fixed prompt"
"$BIN/llama-eval-callback" -m "$MODEL" -ngl 99 -n 1 --temp 0 --seed 1 \
  -p "The capital of France is" > "$OUT/eval_callback.log" 2>&1 || true
grep -c "common_debug_cb_eval" "$OUT/eval_callback.log" || true

echo "$OUT"
