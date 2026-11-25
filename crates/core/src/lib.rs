// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-core`: types shared by every Astral crate. No GPU dependency.

#![forbid(unsafe_code)]

pub mod arch;
pub mod error;

pub use error::{AstralError, Result};
