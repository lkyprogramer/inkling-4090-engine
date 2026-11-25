// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Synthetic Minds LTD
use std::fmt;

/// The single error type used across Astral crates.
#[derive(Debug)]
pub enum AstralError {
    /// A CUDA driver call returned a non-success code.
    Cuda {
        call: &'static str,
        code: i32,
        name: String,
    },
    /// A required shared library (e.g. `libcuda`) could not be loaded.
    LibraryNotFound {
        lib: String,
        detail: String,
    },
    /// The hardware or build does not support the requested operation.
    Unsupported(String),
    /// Invalid argument or state.
    Invalid(String),
    Io(std::io::Error),
}

impl fmt::Display for AstralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AstralError::Cuda { call, code, name } => write!(f, "{call} failed: {name} ({code})"),
            AstralError::LibraryNotFound { lib, detail } => {
                write!(f, "could not load {lib}: {detail}")
            }
            AstralError::Unsupported(m) => write!(f, "unsupported: {m}"),
            AstralError::Invalid(m) => write!(f, "invalid: {m}"),
            AstralError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for AstralError {}

impl From<std::io::Error> for AstralError {
    fn from(e: std::io::Error) -> Self {
        AstralError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, AstralError>;
