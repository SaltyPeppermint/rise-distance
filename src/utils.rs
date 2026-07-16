use std::collections::VecDeque;
use std::hash::Hash;

use egg::{
    Analysis, CostFunction, Extractor, Id, Language, LpCostFunction, LpExtractor, RecExpr, Runner,
};
use hashbrown::{HashMap, HashSet};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

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

/// Current process RSS in bytes via the `memory-stats` crate. `None` if the
/// platform reader is unavailable (the memory limit is then not enforced).
#[must_use]
pub fn process_rss_bytes() -> Option<u64> {
    memory_stats::memory_stats().map(|s| s.physical_mem as u64)
}

/// Peak resident set size of this process in bytes, read from
/// `/proc/self/status` (`VmHWM`). Matches what htop reports and the RSS the
/// `--max-memory` hook budgets against.
///
/// # Panics
///
/// Panics if `/proc/self/status` is unreadable or its `VmHWM` line is missing
/// or malformed.
#[must_use]
pub fn peak_rss_bytes() -> u64 {
    let status =
        std::fs::read_to_string("/proc/self/status").expect("failed to read /proc/self/status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .expect("malformed VmHWM line");
            return kb * 1024;
        }
    }
    panic!("VmHWM not found in /proc/self/status");
}
