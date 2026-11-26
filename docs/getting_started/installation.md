# Installation

Astral is built from source. It produces one binary, `astral`, with the CUDA kernels embedded as fatbins.

## Requirements

- Linux (x86_64)
- An NVIDIA GPU and driver with CUDA 12 support (see [supported hardware](../hardware/supported_hardware.md))
- The CUDA toolkit (`nvcc` 12.8 or newer) to compile the kernels
- A recent stable Rust toolchain (`rustup` recommended)
- A C compiler (the tokenizer library builds a small C dependency)

Without `nvcc` the workspace still builds, but with no GPU kernels.

## Build

```bash
cargo build --release
```

Kernels are compiled for the targets in `ASTRAL_CUDA_ARCHS` (default `80`). Build for every GPU generation you will run on. For Hopper, include `90a` to enable the wgmma prefill kernels:

```bash
ASTRAL_CUDA_ARCHS=80,90a cargo build --release
```

For an RTX 4090 use `ASTRAL_CUDA_ARCHS=89`; see [RTX 4090](../hardware/rtx4090.md) for installing the toolkit in a user directory without touching the driver.

Each target is compiled to SASS ahead of time (`-gencode arch=compute_X,code=sm_X`), so the binary carries no PTX JIT step.

## Check the installation

```bash
./target/release/astral version        # build info and kernel targets
./target/release/astral hw             # GPU, driver and capability detection
./target/release/astral hw --bench     # plus a memory-bandwidth microbenchmark
./target/release/astral selftest       # every kernel against its CPU reference
```

`selftest` should report every case as passed. If a kernel module has no binary for your GPU, rebuild with the matching `ASTRAL_CUDA_ARCHS` entry.
