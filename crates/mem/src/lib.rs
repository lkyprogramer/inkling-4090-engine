// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-mem`: GPU memory management (spec 04 §3, §5; ADR-0003).
//!
//! - [`BudgetArbiter`] owns one byte budget and hands out granule-aligned chunks to
//!   named pools, honouring per-pool floors. Pure bookkeeping: testable on CPU.
//! - [`SlabPool`] carves fixed-size blocks out of arbiter chunks (one pool per cache
//!   kind: global KV, SWA KV, conv state, ...), so pools never pad to a shared page size.
//! - [`GpuArena`] backs the arbiter's address space with a [`astral_gpu::VirtualRange`],
//!   giving every block a fixed device address for CUDA graphs.

#![forbid(unsafe_code)]

mod arbiter;
mod arena;
mod slab;

pub use arbiter::{BudgetArbiter, PoolId};
pub use arena::GpuArena;
pub use slab::{BlockId, SlabPool};
