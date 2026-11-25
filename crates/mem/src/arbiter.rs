// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
use astral_core::{AstralError, Result};
use std::collections::BTreeMap;

/// Handle to a pool registered with a [`BudgetArbiter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolId(pub(crate) usize);

#[derive(Debug, Clone)]
struct Account {
    name: String,
    floor: usize,
    used: usize,
}

/// One byte budget shared by all pools, in units of `granule` bytes.
///
/// Offsets are relative to the arena base. Chunks are taken from an exact-size free
/// list first, then by bumping the high-water mark. The high-water mark never exceeds
/// the budget, so physical memory mapped for it never exceeds the budget either.
#[derive(Debug, Clone)]
pub struct BudgetArbiter {
    total: usize,
    granule: usize,
    used: usize,
    high_water: usize,
    free: BTreeMap<usize, Vec<usize>>,
    accounts: Vec<Account>,
}

fn round_up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

impl BudgetArbiter {
    /// `total` is rounded down to a whole number of granules.
    pub fn new(total: usize, granule: usize) -> Self {
        assert!(granule > 0, "granule must be > 0");
        Self { total: total / granule * granule, granule, used: 0, high_water: 0, free: BTreeMap::new(), accounts: Vec::new() }
    }

    /// Registers a pool. `floor` bytes are guaranteed to remain available to it.
    pub fn register(&mut self, name: &str, floor: usize) -> Result<PoolId> {
        let floor = round_up(floor, self.granule);
        let floors: usize = self.accounts.iter().map(|a| a.floor).sum();
        if floors + floor > self.total {
            return Err(AstralError::Invalid(format!("pool `{name}` floor {floor} B makes total floors exceed budget {} B", self.total)));
        }
        self.accounts.push(Account { name: name.to_string(), floor, used: 0 });
        Ok(PoolId(self.accounts.len() - 1))
    }

    /// Bytes `pool` may still take without eating into other pools' unused floors.
    pub fn available_to(&self, pool: PoolId) -> usize {
        let reserved_for_others: usize = self.accounts.iter().enumerate().filter(|(i, _)| *i != pool.0).map(|(_, a)| a.floor.saturating_sub(a.used)).sum();
        self.total.saturating_sub(self.used).saturating_sub(reserved_for_others)
    }

    /// Grants a chunk of `bytes` (rounded up to the granule). Returns its offset, or
    /// `None` if the budget, other pools' floors, or fragmentation prevents it.
    pub fn grant(&mut self, pool: PoolId, bytes: usize) -> Option<usize> {
        let size = round_up(bytes.max(1), self.granule);
        if size > self.available_to(pool) {
            return None;
        }
        let offset = match self.free.get_mut(&size).and_then(|v| v.pop()) {
            Some(off) => off,
            None => {
                if self.high_water + size > self.total {
                    return None;
                }
                let off = self.high_water;
                self.high_water += size;
                off
            }
        };
        if self.free.get(&size).is_some_and(|v| v.is_empty()) {
            self.free.remove(&size);
        }
        self.used += size;
        self.accounts[pool.0].used += size;
        Some(offset)
    }

    /// Returns a chunk previously granted to `pool` with the same `bytes`.
    pub fn release(&mut self, pool: PoolId, offset: usize, bytes: usize) {
        let size = round_up(bytes.max(1), self.granule);
        let acct = &mut self.accounts[pool.0];
        assert!(acct.used >= size, "pool `{}` releasing more than it holds", acct.name);
        acct.used -= size;
        self.used -= size;
        self.free.entry(size).or_default().push(offset);
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn granule(&self) -> usize {
        self.granule
    }

    /// Highest offset ever granted; the arena must map at least this much.
    pub fn high_water(&self) -> usize {
        self.high_water
    }

    pub fn pool_used(&self, pool: PoolId) -> usize {
        self.accounts[pool.0].used
    }

    pub fn pool_name(&self, pool: PoolId) -> &str {
        &self.accounts[pool.0].name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: usize = 1 << 20;

    #[test]
    fn floors_are_protected() {
        let mut a = BudgetArbiter::new(10 * MIB, 2 * MIB);
        let kv = a.register("global_kv", 0).unwrap();
        let swa = a.register("swa_kv", 4 * MIB).unwrap();
        // global_kv can take 10 - 4 (swa floor) = 6 MiB.
        assert_eq!(a.available_to(kv), 6 * MIB);
        assert!(a.grant(kv, 6 * MIB).is_some());
        assert!(a.grant(kv, 1).is_none(), "must not eat swa's floor");
        assert!(a.grant(swa, 4 * MIB).is_some());
        assert_eq!(a.used(), 10 * MIB);
    }

    #[test]
    fn floors_cannot_exceed_budget() {
        let mut a = BudgetArbiter::new(4 * MIB, 2 * MIB);
        a.register("a", 4 * MIB).unwrap();
        assert!(a.register("b", 1).is_err());
    }

    #[test]
    fn released_chunks_are_reused_without_raising_high_water() {
        let mut a = BudgetArbiter::new(8 * MIB, 2 * MIB);
        let p = a.register("p", 0).unwrap();
        let x = a.grant(p, 2 * MIB).unwrap();
        let _y = a.grant(p, 2 * MIB).unwrap();
        assert_eq!(a.high_water(), 4 * MIB);
        a.release(p, x, 2 * MIB);
        assert_eq!(a.grant(p, 2 * MIB), Some(x));
        assert_eq!(a.high_water(), 4 * MIB);
    }

    #[test]
    fn sizes_round_to_granule() {
        let mut a = BudgetArbiter::new(8 * MIB, 2 * MIB);
        let p = a.register("p", 0).unwrap();
        a.grant(p, 1).unwrap();
        assert_eq!(a.pool_used(p), 2 * MIB);
        assert_eq!(BudgetArbiter::new(7 * MIB, 2 * MIB).total(), 6 * MIB);
    }
}
