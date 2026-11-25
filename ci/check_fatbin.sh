#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Synthetic Minds LTD
#
# Static architecture gate (spec 01 §7, spec 08 L6, R-HW-5). Needs nvcc + cuobjdump, no GPU.
#
#  1. Builds astral-kernels-src for every arch in ASTRAL_CUDA_ARCHS.
#  2. Asserts every fatbin contains an ELF cubin for *each* requested arch
#     (catches silent arch drops such as `-arch=sm_121a` emitting plain sm_121).
#  3. Disassembles each cubin and checks ci/sass_expectations.txt: lines of
#        <kernel> <arch|*> <regex>
#     meaning "this kernel's SASS on this arch must match regex".
#
# Usage: ASTRAL_CUDA_ARCHS=80,90a,100a,120a ci/check_fatbin.sh
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARCHS=${ASTRAL_CUDA_ARCHS:-80}
TARGET="$ROOT/target/fatbin-check"
EXPECT="$ROOT/ci/sass_expectations.txt"
REPORT="$TARGET/report"
mkdir -p "$REPORT"

echo "== building kernels for: $ARCHS"
# Start clean so only this build's fatbins are checked. `cargo clean -p` also makes the build
# script rerun; deleting the fatbins alone leaves cargo thinking they are up to date.
cargo clean -q --release -p astral-kernels-src --target-dir "$TARGET" 2>/dev/null || true
if ! ASTRAL_CUDA_ARCHS="$ARCHS" cargo build -q --release -p astral-kernels-src --target-dir "$TARGET" 2> "$REPORT/build.log"; then
  echo "FAIL: nvcc could not build for [$ARCHS] with $(nvcc --version | grep release)"
  grep -iE "unsupported|not defined|error" "$REPORT/build.log" | head -5
  exit 1
fi
mapfile -t FATBINS < <(find "$TARGET" -path '*astral-kernels-src*' -name '*.fatbin' | sort)
[ ${#FATBINS[@]} -gt 0 ] || { echo "FAIL: no fatbins built (is nvcc on PATH?)"; exit 1; }

fail=0
IFS=',' read -r -a ARCH_LIST <<< "$ARCHS"
for fb in "${FATBINS[@]}"; do
  name=$(basename "$fb" .fatbin)
  elfs=$(cuobjdump --list-elf "$fb")
  for a in "${ARCH_LIST[@]}"; do
    a=$(echo "$a" | tr -d ' ')
    if grep -Eq "\.sm_${a}\.cubin$" <<< "$elfs"; then
      echo "ok    $name has sm_$a"
    else
      echo "FAIL  $name missing sm_$a cubin"; fail=1
    fi
  done
  for a in "${ARCH_LIST[@]}"; do
    a=$(echo "$a" | tr -d ' ')
    cuobjdump -sass -arch "sm_$a" "$fb" > "$REPORT/$name.sm_$a.sass" 2>/dev/null || true
  done
done

if [ -f "$EXPECT" ]; then
  while read -r kernel arch regex; do
    [[ -z "${kernel:-}" || "$kernel" == \#* ]] && continue
    for a in "${ARCH_LIST[@]}"; do
      a=$(echo "$a" | tr -d ' ')
      [[ "$arch" != "*" && "$arch" != "$a" ]] && continue
      for sass in "$REPORT"/*.sm_"$a".sass; do
        # Extract the kernel's function body from the dump.
        body=$(awk -v k="$kernel" '/Function :/ {on = ($3 == k)} on' "$sass")
        [ -z "$body" ] && continue
        if grep -Eq "$regex" <<< "$body"; then
          echo "ok    $kernel sm_$a matches /$regex/"
        else
          echo "FAIL  $kernel sm_$a does not match /$regex/"; fail=1
        fi
      done
    done
  done < "$EXPECT"
fi

echo "SASS dumps: $REPORT"
[ $fail -eq 0 ] && echo "PASS" || { echo "FAILED"; exit 1; }
