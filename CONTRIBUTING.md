# Contributing to Astral Inference Server

> **This fork (inkling-4090-engine).** Issues and pull requests about the RTX 4090 port, the
> comparison tools or anything else added here go to
> [lkyprogramer/inkling-4090-engine](https://github.com/lkyprogramer/inkling-4090-engine). Fixes that
> apply to upstream as well (such as the legacy-stream ordering fix) are best also offered to
> [upstream](https://github.com/syntheticminds-lab/Astral-Inference-Server). The rules below apply to both.

Thank you for your interest in contributing to Astral Inference Server. Bug reports, benchmarks on new hardware, documentation fixes and code are all welcome.

## Reporting issues

Please open a GitHub issue with:

- the output of `astral version` and `astral hw`;
- the exact command, model files and request that reproduce the problem;
- for wrong output, the expected and the actual tokens or text;
- for performance, the benchmark command and its full output.

## Development setup

```bash
ASTRAL_CUDA_ARCHS=80,90a cargo build --release
cargo test                         # CPU tests (no GPU needed)
./target/release/astral selftest   # GPU kernels against their CPU references
```

## Pull requests

1. **One change per pull request.** Explain why the change is made, and include measurements when it affects speed.
2. **Correctness first.** A kernel change must keep `astral selftest` passing and `astral verify-model` matching the reference. A change to speculative decoding must keep `astral spec-generate` reporting near-ties only.
3. **Benchmarks** use [`bench/serving/benchmark_serving.py`](bench/serving/benchmark_serving.py) on both prompt sets, and report the GPU and driver.
4. **Code style:** `cargo fmt` and `cargo clippy` clean. `unsafe` stays inside `crates/gpu` and the kernel launchers, each block with a `SAFETY:` comment. Every launcher checks buffer sizes before launching.
5. **File headers:** every source file starts with the SPDX identifier and copyright line used throughout the repository.

## Licensing of contributions

Astral is licensed under `Apache-2.0`, and contributions are accepted under the same licence (section 5 of the License). Sign off every commit (`git commit -s`) to certify the [Developer Certificate of Origin](https://developercertificate.org/).
