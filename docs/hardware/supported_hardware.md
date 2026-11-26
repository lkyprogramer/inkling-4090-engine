# Supported Hardware

Astral runs on NVIDIA GPUs with CUDA 12 drivers, one GPU per process.

| GPU family | Target | Status |
|---|---|---|
| Hopper (H100) | `sm_90a` | **Verified.** All results and benchmarks in these docs |
| Ampere data-centre (A100) | `sm_80` | Default build target; early releases verified, current release not re-verified |
| Ada consumer (RTX 4090) | `sm_89` (`ASTRAL_CUDA_ARCHS=89`) | **Verified in this fork** for Inkling 2 Mini, see [RTX 4090](rtx4090.md) |
| Ampere workstation and consumer | `sm_86` (`ASTRAL_CUDA_ARCHS=86`) | Builds; not tested |
| Blackwell (`sm_100`, `sm_120`) | `ASTRAL_CUDA_ARCHS=100a` / `120a` | Builds; native FP4 tensor cores not used yet |

Features enable themselves by compute capability:

- **wgmma prefill GEMMs:** Hopper.
- **Programmatic dependent launch:** compute capability 9.0 and newer.
- **Tensor-core GEMVs:** `mma.sync`, sm_80 and newer.

## Memory

Inkling 2 Mini needs about 18 GB for weights plus 0.6 GB for the DSpark draft, and memory for the context. A 24 GB GPU is tight but works: on an RTX 4090 the UD-Q4_K_XL GGUF with the DSpark draft serves a context of 8,192 tokens.

## Not supported

- AMD and Apple GPUs.
- CPU-only inference: the CPU backend is a test reference only.
- More than one GPU per process. Multi-GPU serving on one node is planned for this (Developer) edition. Multi-node serving is part of the Enterprise edition.
