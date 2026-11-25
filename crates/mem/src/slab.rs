// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
use crate::{BudgetArbiter, PoolId};
use astral_core::{AstralError, Result};

/// A block inside a [`SlabPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId {
    chunk: u32,
    index: u32,
}

#[derive(Debug, Clone)]
struct Chunk {
    /// Arena offset, or `None` once trimmed back to the arbiter (slot reusable).
    offset: Option<usize>,
    live: Vec<bool>,
    live_count: u32,
}

/// Fixed-size blocks carved from granule-aligned arbiter chunks.
/// A block never straddles two chunks.
#[derive(Debug, Clone)]
pub struct SlabPool {
    pool: PoolId,
    block_bytes: usize,
    chunk_bytes: usize,
    blocks_per_chunk: u32,
    chunks: Vec<Chunk>,
    free: Vec<BlockId>,
    live: usize,
}

impl SlabPool {
    /// `block_bytes` may be any size; chunks are the smallest granule multiple that
    /// holds at least one block (and at least one granule).
    pub fn new(pool: PoolId, block_bytes: usize, granule: usize) -> Self {
        assert!(block_bytes > 0 && granule > 0);
        let chunk_bytes = block_bytes.max(granule).div_ceil(granule) * granule;
        let blocks_per_chunk = (chunk_bytes / block_bytes) as u32;
        Self { pool, block_bytes, chunk_bytes, blocks_per_chunk, chunks: Vec::new(), free: Vec::new(), live: 0 }
    }

    pub fn pool(&self) -> PoolId {
        self.pool
    }

    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk_bytes
    }

    pub fn blocks_per_chunk(&self) -> u32 {
        self.blocks_per_chunk
    }

    /// Live (allocated) blocks.
    pub fn live(&self) -> usize {
        self.live
    }

    /// Blocks currently backed by chunks (live + free).
    pub fn capacity(&self) -> usize {
        self.chunks.iter().filter(|c| c.offset.is_some()).count() * self.blocks_per_chunk as usize
    }

    fn grow(&mut self, arb: &mut BudgetArbiter) -> bool {
        let Some(offset) = arb.grant(self.pool, self.chunk_bytes) else { return false };
        let n = self.blocks_per_chunk;
        let fresh = Chunk { offset: Some(offset), live: vec![false; n as usize], live_count: 0 };
        let slot = match self.chunks.iter().position(|c| c.offset.is_none()) {
            Some(i) => {
                self.chunks[i] = fresh;
                i
            }
            None => {
                self.chunks.push(fresh);
                self.chunks.len() - 1
            }
        } as u32;
        // Push in reverse so blocks are handed out in ascending address order.
        self.free.extend((0..n).rev().map(|index| BlockId { chunk: slot, index }));
        true
    }

    /// Allocates one block, taking a new chunk from the arbiter if needed.
    pub fn alloc(&mut self, arb: &mut BudgetArbiter) -> Option<BlockId> {
        if self.free.is_empty() && !self.grow(arb) {
            return None;
        }
        let id = self.free.pop()?;
        let c = &mut self.chunks[id.chunk as usize];
        c.live[id.index as usize] = true;
        c.live_count += 1;
        self.live += 1;
        Some(id)
    }

    /// Frees a block. Double frees and stale ids are errors, not undefined behaviour.
    pub fn free(&mut self, id: BlockId) -> Result<()> {
        let c = self.chunks.get_mut(id.chunk as usize).filter(|c| c.offset.is_some()).ok_or_else(|| AstralError::Invalid(format!("free of stale block {id:?}")))?;
        let slot = c.live.get_mut(id.index as usize).ok_or_else(|| AstralError::Invalid(format!("bad block {id:?}")))?;
        if !*slot {
            return Err(AstralError::Invalid(format!("double free of block {id:?}")));
        }
        *slot = false;
        c.live_count -= 1;
        self.live -= 1;
        self.free.push(id);
        Ok(())
    }

    /// Arena offset of a live block.
    pub fn offset(&self, id: BlockId) -> Option<usize> {
        let c = self.chunks.get(id.chunk as usize)?;
        let base = c.offset?;
        c.live.get(id.index as usize).copied().filter(|&l| l)?;
        Some(base + id.index as usize * self.block_bytes)
    }

    /// Returns fully-free chunks to the arbiter. Returns bytes released.
    pub fn trim(&mut self, arb: &mut BudgetArbiter) -> usize {
        let mut released = 0;
        for (i, c) in self.chunks.iter_mut().enumerate() {
            if let (Some(off), 0) = (c.offset, c.live_count) {
                arb.release(self.pool, off, self.chunk_bytes);
                c.offset = None;
                released += self.chunk_bytes;
                self.free.retain(|b| b.chunk as usize != i);
            }
        }
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    #[test]
    fn non_granule_block_sizes_do_not_straddle_chunks() {
        let mut arb = BudgetArbiter::new(64 * MIB, 2 * MIB);
        let p = arb.register("state", 0).unwrap();
        // 1.5 MiB blocks: one per 2 MiB chunk.
        let mut s = SlabPool::new(p, 3 * MIB / 2, 2 * MIB);
        assert_eq!(s.blocks_per_chunk(), 1);
        let a = s.alloc(&mut arb).unwrap();
        let b = s.alloc(&mut arb).unwrap();
        assert_eq!(s.offset(b).unwrap() - s.offset(a).unwrap(), 2 * MIB);
    }

    #[test]
    fn double_free_is_rejected() {
        let mut arb = BudgetArbiter::new(8 * MIB, 2 * MIB);
        let p = arb.register("p", 0).unwrap();
        let mut s = SlabPool::new(p, 64 << 10, 2 * MIB);
        let id = s.alloc(&mut arb).unwrap();
        s.free(id).unwrap();
        assert!(s.free(id).is_err());
    }

    #[test]
    fn trim_returns_empty_chunks() {
        let mut arb = BudgetArbiter::new(8 * MIB, 2 * MIB);
        let p = arb.register("p", 0).unwrap();
        let mut s = SlabPool::new(p, MIB, 2 * MIB);
        let ids: Vec<_> = (0..4).map(|_| s.alloc(&mut arb).unwrap()).collect();
        assert_eq!(arb.used(), 4 * MIB);
        for id in ids {
            s.free(id).unwrap();
        }
        assert_eq!(s.trim(&mut arb), 4 * MIB);
        assert_eq!(arb.used(), 0);
        assert!(s.alloc(&mut arb).is_some(), "pool regrows after trim");
    }

    /// Randomised invariant test: two pools, random alloc/free/trim. Live blocks never
    /// overlap, stay below the high-water mark, and the budget is never exceeded.
    #[test]
    fn randomised_no_overlap_no_overcommit() {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut arb = BudgetArbiter::new(96 * MIB, 2 * MIB);
        let pa = arb.register("global_kv", 16 * MIB).unwrap();
        let pb = arb.register("swa_kv", 8 * MIB).unwrap();
        let mut pools = [SlabPool::new(pa, 96 << 10, 2 * MIB), SlabPool::new(pb, 640 << 10, 2 * MIB)];
        let mut live: Vec<Vec<BlockId>> = vec![Vec::new(), Vec::new()];
        for step in 0..20_000 {
            let k = (rnd() % 2) as usize;
            match rnd() % 10 {
                0..=5 => {
                    if let Some(id) = pools[k].alloc(&mut arb) {
                        live[k].push(id);
                    }
                }
                6..=8 if !live[k].is_empty() => {
                    let i = (rnd() as usize) % live[k].len();
                    pools[k].free(live[k].swap_remove(i)).unwrap();
                }
                _ => {
                    pools[k].trim(&mut arb);
                }
            }
            assert!(arb.used() <= arb.total());
            if step % 500 == 0 {
                let mut all = Vec::new();
                for (k, ids) in live.iter().enumerate() {
                    for &id in ids {
                        let off = pools[k].offset(id).expect("live block has offset");
                        assert!(off + pools[k].block_bytes() <= arb.high_water());
                        all.push((off, off + pools[k].block_bytes()));
                    }
                }
                all.sort();
                for w in all.windows(2) {
                    assert!(w[0].1 <= w[1].0, "overlap {:?} {:?}", w[0], w[1]);
                }
            }
        }
        // Floors held: each pool can always get its floor back after the other trims.
        for (k, ids) in live.iter_mut().enumerate() {
            for id in ids.drain(..) {
                pools[k].free(id).unwrap();
            }
            pools[k].trim(&mut arb);
        }
        assert_eq!(arb.used(), 0);
    }
}
