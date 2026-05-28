use std::collections::VecDeque;
use std::hash::Hash;
use std::time::Duration;

use egg::{Analysis, EGraph, Id, Language, RecExpr, Rewrite, Runner, SimpleScheduler};
use hashbrown::{HashMap, HashSet};
use memory_stats::memory_stats;
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

    pub fn is_empty(&self) -> bool {
        let r = self.queue.is_empty();
        debug_assert_eq!(r, self.set.is_empty());
        r
    }

    /// Drain all elements from the queue, returning them in order.
    pub fn drain(&mut self) -> std::collections::vec_deque::Drain<'_, T> {
        self.set.clear();
        self.queue.drain(..)
    }
}

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

pub fn grow_egraph_until<L, N, S>(
    search_name: &str,
    egraph: EGraph<L, N>,
    rules: &[Rewrite<L, N>],
    mut satisfied: S,
) -> EGraph<L, N>
where
    S: FnMut(&mut Runner<L, N>) -> bool + 'static,
    L: Language,
    N: Analysis<L> + Default,
{
    let search_name_hook = search_name.to_owned();
    let runner = Runner::default()
        .with_scheduler(SimpleScheduler)
        .with_iter_limit(100)
        .with_node_limit(100_000_000)
        .with_time_limit(Duration::from_secs(5 * 60))
        .with_hook(move |runner| {
            let mut out_of_memory = false;
            // hook 0 <- nothing
            // iteration 0
            // hook 1 <- #0 size etc after iteration 0 + memory after iteration 0
            if let Some(it) = runner.iterations.last() {
                out_of_memory = iteration_stats(&search_name_hook, it, runner.iterations.len());
            }

            if satisfied(runner) {
                Err(String::from("Satisfied"))
            } else if out_of_memory {
                Err(String::from("Out of Memory"))
            } else {
                Ok(())
            }
        })
        .with_egraph(egraph)
        .run(rules);
    iteration_stats(
        search_name,
        runner.iterations.last().unwrap(),
        runner.iterations.len(),
    );
    runner.print_report();
    runner.egraph
}

// search name,
// iteration number,
// physical memory,
// virtual memory,
// e-graph nodes,
// e-graph classes,
// applied rules,
// total time,
// hook time,
// search time,
// apply time,
// rebuild time
fn iteration_stats(search_name: &str, it: &egg::Iteration<()>, it_number: usize) -> bool {
    let memory = memory_stats().expect("could not get current memory usage");
    let out_of_memory = memory.virtual_mem > 8_000_000_000;
    let found = match &it.stop_reason {
        Some(egg::StopReason::Other(s)) => s == "Satisfied",
        _ => false,
    };
    eprintln!(
        "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}",
        search_name,
        it_number,
        memory.physical_mem,
        memory.virtual_mem,
        it.egraph_nodes,
        it.egraph_classes,
        it.applied.iter().map(|(_, &n)| n).sum::<usize>(),
        it.total_time,
        it.hook_time,
        it.search_time,
        it.apply_time,
        it.rebuild_time,
        found
    );
    out_of_memory
}
