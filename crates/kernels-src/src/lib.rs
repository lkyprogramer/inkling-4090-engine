// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-kernels-src`: embedded fatbins built from `cu/*.cu` by `build.rs`.

#![forbid(unsafe_code)]

include!(concat!(env!("OUT_DIR"), "/fatbins.rs"));
