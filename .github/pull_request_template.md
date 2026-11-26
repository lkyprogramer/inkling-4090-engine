## What and why

<!-- One change per pull request. -->

## Checks

- [ ] `cargo fmt --all --check` and `cargo test --workspace`
- [ ] `astral selftest` passes on a GPU (which one?)
- [ ] For kernel or model changes: `astral verify-model` still matches the reference
- [ ] For speculative decoding: `astral spec-generate` reports near-ties only
- [ ] For speed: before/after numbers with the command used
