# Astral Inference Server Documentation

Astral Inference Server is an inference engine and OpenAI-compatible server for large language models, written in Rust with hand-written CUDA kernels.

## Getting started

- [Installation](getting_started/installation.md): requirements, building from source, checking your GPU
- [Quickstart](getting_started/quickstart.md): serve a model and send your first request

## Serving

- [OpenAI-compatible server](serving/openai_compatible_server.md): endpoints, request parameters, server options
- [Speculative decoding](features/speculative_decoding.md): the DSpark draft, adaptive draft length, tuning

## Reference

- [Supported models](models/supported_models.md)
- [Supported hardware](hardware/supported_hardware.md), including the [RTX 4090 port](hardware/rtx4090.md)
- [Environment variables](configuration/environment_variables.md)
- [Benchmarks: Inkling 2 Mini](benchmarks/inkling2_mini.md) (H100) and [on RTX 4090](benchmarks/inkling2_mini_rtx4090.md)
- [llama.cpp comparison tools](../bench/llamacpp_compare/README.md)

## Editions

This is the Developer edition (Apache-2.0): one node, one or more GPUs. Multi-node serving, custom deployments and support are offered in the Enterprise edition; see the [README](../README.md#editions).

## Design

- [Design specification](design/spec.md), with per-area specs in [design/spec/](design/spec/)
- [Architecture decision records](design/adr/)
- [Engineering log](design/engineering-log.md): measurements and the reasoning behind each optimisation
