// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-kernels-src`: embedded fatbins built from `cu/*.cu` by `build.rs`.

#![forbid(unsafe_code)]

include!(concat!(env!("OUT_DIR"), "/fatbins.rs"));
