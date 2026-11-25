#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Synthetic Minds LTD
#
# Astral benchmark harness, baseline v0 (spec 08 §3): llama.cpp on the same GGUF file
# Astral will run. Records exact versions, hardware state and file hashes next to every
# number, so results are reproducible and comparable across runs.
#
# Usage: bench/baseline_llamacpp.sh <model.gguf> <llama.cpp dir> <astral binary> [out root]
set -euo pipefail

MODEL=${1:?model.gguf}
LLAMA=${2:?llama.cpp checkout}
ASTRAL=${3:?astral binary}
OUT_ROOT=${4:-$HOME/astral-work/results}
REPS=${REPS:-5}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$OUT_ROOT/$STAMP-llamacpp"
mkdir -p "$OUT"
BIN="$LLAMA/build/bin"

echo "== building llama-batched-bench"
cmake --build "$LLAMA/build" -j "$(nproc)" --target llama-batched-bench >/dev/null

echo "== recording environment"
{
  echo "{"
  echo "  \"timestamp_utc\": \"$STAMP\","
  echo "  \"host\": \"$(hostname)\","
  echo "  \"os\": \"$(. /etc/os-release; echo "$PRETTY_NAME")\","
  echo "  \"kernel\": \"$(uname -r)\","
  echo "  \"llama_cpp_commit\": \"$(git -C "$LLAMA" rev-parse HEAD)\","
  echo "  \"llama_cpp_describe\": \"$(git -C "$LLAMA" describe --tags --always)\","
  echo "  \"model_path\": \"$MODEL\","
  echo "  \"model_bytes\": $(stat -c %s "$MODEL"),"
  echo "  \"model_sha256\": \"$(sha256sum "$MODEL" | cut -d' ' -f1)\","
  echo "  \"reps\": $REPS"
  echo "}"
} > "$OUT/env.json"
"$ASTRAL" hw --bench-fresh > "$OUT/astral_hw.json"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,pstate --format=csv > "$OUT/nvidia_smi_before.csv"

echo "== llama-bench: prefill (pp) and decode (tg), batch 1"
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -p 512,2048 -n 128 -r "$REPS" -o json > "$OUT/llama_bench.json" 2> "$OUT/llama_bench.log"

echo "== llama-batched-bench: concurrent decode (npl = parallel sequences)"
"$BIN/llama-batched-bench" -m "$MODEL" -ngl 99 -c 65536 -b 2048 -ub 512 \
  -npp 512 -ntg 128 -npl 1,2,4,8,16,32 --output-format jsonl \
  > "$OUT/batched_bench.jsonl" 2> "$OUT/batched_bench.log"

nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,pstate --format=csv > "$OUT/nvidia_smi_after.csv"
echo "$OUT"
