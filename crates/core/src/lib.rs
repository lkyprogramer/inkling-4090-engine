// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-core`: types shared by every Astral crate. No GPU dependency.

#![forbid(unsafe_code)]

pub mod arch;
pub mod error;

pub use error::{AstralError, Result};
