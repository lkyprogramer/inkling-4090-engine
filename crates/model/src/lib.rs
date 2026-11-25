// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-model`: model implementations on the GPU.
//!
//! [`inkling`] runs Inkling 2 Mini (`inkling2_loop` GGUF) one token at a time, mirroring
//! llama.cpp's inkling2 graph so it can be verified layer by layer against llama.cpp dumps.

#![forbid(unsafe_code)]

pub mod dspark;
pub mod inkling;
pub mod qwen35;
