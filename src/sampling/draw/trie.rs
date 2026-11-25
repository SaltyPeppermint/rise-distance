use num::BigUint;
use rand_chacha::ChaCha12Rng;

use crate::utils::HashMap;

/// Handle into a [`SampledTrie`] arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TrieNodeId(usize);

impl TrieNodeId {
    const ROOT: Self = Self(0);
}

/// Mutable cursor threading the trie and RNG through one pre-order draw.
pub(crate) struct DrawCursor<'a> {
    sampled: &'a mut SampledTrie,
    node: TrieNodeId,
    pub(crate) rng: &'a mut ChaCha12Rng,
}

impl<'a> DrawCursor<'a> {
    pub(super) fn new(sampled: &'a mut SampledTrie, rng: &'a mut ChaCha12Rng) -> Self {
        Self {
            sampled,
            node: TrieNodeId::ROOT,
            rng,
        }
    }

    pub(super) fn active_rest<'r>(&self, rest_total: Option<&'r BigUint>) -> Option<&'r BigUint> {
        self.sampled
            .has_children(self.node)
            .then_some(rest_total)
            .flatten()
    }

    pub(crate) fn used(&self, key: usize) -> u64 {
        self.sampled.used(self.node, key)
    }

    pub(super) fn step(&mut self, key: usize) {
        self.node = self.sampled.step_and_increment(self.node, key);
    }
}

#[derive(Debug, Default)]
enum TrieChildren {
    #[default]
    Empty,
    One(usize, TrieNodeId),
    Many(HashMap<usize, TrieNodeId>),
}

impl TrieChildren {
    fn get(&self, key: usize) -> Option<TrieNodeId> {
        match self {
            Self::Empty => None,
            Self::One(existing_key, child) => (*existing_key == key).then_some(*child),
            Self::Many(map) => map.get(&key).copied(),
        }
    }
}

#[derive(Debug, Default)]
struct TrieNode {
    drawn: u64,
    children: TrieChildren,
}

/// Sparse trie of pre-order derivation paths already sampled for a single
/// `(root, size)` bucket.
#[derive(Debug)]
pub(super) struct SampledTrie {
    nodes: Vec<TrieNode>,
}

impl SampledTrie {
    #[must_use]
    pub(super) fn new() -> Self {
        Self {
            nodes: vec![TrieNode::default()],
        }
    }

    fn has_children(&self, node: TrieNodeId) -> bool {
        !matches!(self.nodes[node.0].children, TrieChildren::Empty)
    }

    fn used(&self, node: TrieNodeId, key: usize) -> u64 {
        self.nodes[node.0]
            .children
            .get(key)
            .map_or(0, |child| self.nodes[child.0].drawn)
    }

    fn step_and_increment(&mut self, node: TrieNodeId, key: usize) -> TrieNodeId {
        if let Some(child) = self.nodes[node.0].children.get(key) {
            self.nodes[child.0].drawn += 1;
            return child;
        }

        let child = TrieNodeId(self.nodes.len());
        self.nodes.push(TrieNode {
            drawn: 1,
            children: TrieChildren::Empty,
        });

        let children = &mut self.nodes[node.0].children;
        match children {
            TrieChildren::Empty => {
                *children = TrieChildren::One(key, child);
            }
            TrieChildren::One(existing_key, existing_child) => {
                let mut map = HashMap::default();
                map.insert(*existing_key, *existing_child);
                map.insert(key, child);
                *children = TrieChildren::Many(map);
            }
            TrieChildren::Many(map) => {
                map.insert(key, child);
            }
        }

        child
    }
}

impl Default for SampledTrie {
    fn default() -> Self {
        Self::new()
    }
}
