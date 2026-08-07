use std::collections::VecDeque;
use std::hash::Hash;

use egg::{
    Analysis, CostFunction, Extractor, Id, Language, LpCostFunction, LpExtractor, RecExpr, Runner,
};
use hashbrown::{HashMap, HashSet};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

/// Minimal union-find for dense index types.
///
/// Storage is one index per element. `find` applies path halving, and `union`
/// deliberately makes the left representative the new root; callers that
/// only care about the partition need not reproduce another union-find's
/// representative choices.
#[derive(Debug)]
pub(crate) struct DenseUnionFind<I> {
    parents: Vec<I>,
}

impl<I> DenseUnionFind<I>
where
    I: Copy + Eq + From<usize>,
    usize: From<I>,
{
    pub(crate) fn new(len: usize) -> Self {
        Self {
            parents: (0..len).map(I::from).collect(),
        }
    }

    pub(crate) fn find(&mut self, mut id: I) -> I {
        loop {
            let parent = self.parents[usize::from(id)];
            if parent == id {
                return id;
            }
            let grandparent = self.parents[usize::from(parent)];
            self.parents[usize::from(id)] = grandparent;
            id = grandparent;
        }
    }

    pub(crate) fn union(&mut self, left: I, right: I) -> bool {
        let left = self.find(left);
        let right = self.find(right);
        if left == right {
            false
        } else {
            self.parents[usize::from(right)] = left;
            true
        }
    }
}

/// A data structure to maintain a queue of unique elements.
///
/// Notably, insert/pop operations have O(1) expected amortized runtime complexity.
///
/// Thanks @Bastacyclop for the implementation!
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(crate) struct UniqueQueue<T: Eq + Hash + Clone> {
    set: HashSet<T>, // hashbrown::
    queue: VecDeque<T>,
}

impl<U: Eq + Hash + Clone + Default> FromIterator<U> for UniqueQueue<U> {
    fn from_iter<T: IntoIterator<Item = U>>(iter: T) -> Self {
        let mut queue = Self::default();
        for t in iter {
            queue.insert(t);
        }
        queue
    }
}

impl<T: Eq + Hash + Clone> UniqueQueue<T> {
    pub fn insert(&mut self, t: T) {
        if self.set.insert(t.clone()) {
            self.queue.push_back(t);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for t in iter {
            self.insert(t);
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        let res = self.queue.pop_front();
        if let Some(t) = &res {
            self.set.remove(t);
        }
        res
    }
}

#[must_use]
pub fn combined_rng<const N: usize>(values: [u64; N]) -> ChaCha12Rng {
    const { assert!(N >= 1 && N <= 4, "must provide 1 to 4 u64 values") };

    let mut seed = [0u8; 32];
    for (i, v) in values.iter().enumerate() {
        seed[i * 8..(i + 1) * 8].copy_from_slice(&v.to_le_bytes());
    }
    ChaCha12Rng::from_seed(seed)
}

/// hash consed storage for expressions,
/// cheap replacement for garbage collected expressions
#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) struct ExprHashCons<L: Hash + Eq + Language> {
    node_store: Vec<L>,
    memo: HashMap<L, usize>,
}

impl<L: Language> ExprHashCons<L> {
    pub fn new() -> Self {
        ExprHashCons {
            node_store: Vec::new(),
            memo: HashMap::default(),
        }
    }

    pub(crate) fn add(&mut self, node: L) -> usize {
        if let Some(id) = self.memo.get(&node) {
            return *id;
        }
        let new_id = self.node_store.len();
        self.node_store.push(node.clone());
        self.memo.insert(node, new_id);
        new_id
    }

    pub(crate) fn extract(&self, id: usize) -> RecExpr<L> {
        let mut used = HashSet::new();
        used.insert(id);
        for (i, node) in self.node_store.iter().enumerate().rev() {
            if used.contains(&i) {
                used.extend(node.children().iter().map(|c_id| usize::from(*c_id)));
            }
        }

        let mut fresh = RecExpr::default();
        let mut map = HashMap::<Id, Id>::default();
        for (i, node) in self.node_store.iter().enumerate() {
            if used.contains(&i) {
                let fresh_node = node.clone().map_children(|c| map[&c]);
                let fresh_id = fresh.add(fresh_node);
                map.insert(Id::from(i), fresh_id);
            }
        }

        fresh
    }
}

pub fn cheapest<CF, L, N, I>(runner: &Runner<L, N, I>, cf: CF) -> usize
where
    CF: CostFunction<L, Cost = usize>,
    L: Language,
    N: Analysis<L>,
{
    Extractor::new(&runner.egraph, cf).find_best_cost(runner.roots[0])
}

pub fn cheapest_ilp<CF, L, N, I>(runner: &Runner<L, N, I>, cf: CF) -> RecExpr<L>
where
    CF: LpCostFunction<L, N>,
    L: Language,
    N: Analysis<L>,
{
    let root = runner.egraph.find(runner.roots[0]);
    LpExtractor::new(&runner.egraph, cf).solve(root)
}

pub fn stack_children<L: Language>(children: &[RecExpr<L>], root: L) -> RecExpr<L> {
    let mut i = 0;
    root.map_children(|_c| {
        let new_id = Id::from(i);
        i += 1;
        new_id
    })
    .join_recexprs(|c_id| children[usize::from(c_id)].clone())
}

#[must_use]
pub fn id0() -> Id {
    Id::from(0)
}

/// Bytes in live (allocated-but-not-yet-freed) heap allocations, as reported by
/// jemalloc's `stats.allocated`. Unlike RSS this drops immediately when memory
/// is freed — no `malloc_trim`-style purge is needed to get a clean reading — so
/// it measures the current term's actual footprint rather than leftover
/// allocator pages. Requires the jemalloc global allocator to be installed in
/// the calling binary (`#[global_allocator]`).
///
/// This is a point-in-time reading; jemalloc keeps no high-water mark for
/// `allocated`, so there is no peak variant.
///
/// # Panics
///
/// Panics if the jemalloc ctl epoch cannot be advanced or `stats.allocated`
/// cannot be read (both indicate jemalloc is not the active allocator).
#[must_use]
pub fn live_heap_bytes() -> u64 {
    // Stats are cached per epoch; advance it so the read reflects current state.
    tikv_jemalloc_ctl::epoch::advance().expect("failed to advance jemalloc epoch");
    tikv_jemalloc_ctl::stats::allocated::read().expect("failed to read jemalloc allocated stat")
        as u64
}

#[cfg(test)]
#[must_use]
pub fn sym(name: &str) -> crate::langs::math::Math {
    crate::langs::math::Math::Symbol(name.into())
}

#[cfg(test)]
mod tests {
    use super::DenseUnionFind;

    #[test]
    fn dense_union_find_tracks_partitions_and_noops() {
        let mut sets = DenseUnionFind::<usize>::new(5);

        assert!(sets.union(0, 1));
        assert!(sets.union(1, 2));
        assert!(!sets.union(0, 2));
        assert_eq!(sets.find(0), sets.find(2));
        assert_ne!(sets.find(0), sets.find(3));
    }
}
