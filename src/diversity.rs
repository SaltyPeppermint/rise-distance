//! Diversity-aware sampling utilities.
//!
//! Provides [`DiverseSampler`], an iterator adapter that wraps any iterator
//! producing [`TreeNode`]s and filters for structural diversity using hashing
//! and feature-based deduplication.

use std::hash::{BuildHasher, Hash, Hasher};

use egg::Language;
use hashbrown::{DefaultHashBuilder, HashSet};

use super::tree::Tree;

/// Configuration for diverse sampling.
#[derive(Debug, Clone, bon::Builder)]
pub struct DiverseSamplerConfig {
    /// Maximum attempts to find a novel sample before giving up.
    #[builder(default = 100)]
    pub max_attempts_per_sample: usize,
    /// Minimum novelty ratio (0, 1]. Sample accepted if this fraction of features are new.
    #[builder(default = 0.3)]
    pub min_novelty_ratio: f64,
}

/// Iterator adapter that filters for diverse terms using structural deduplication.
pub struct DiverseSampler<L: Language, I: Iterator<Item = Tree<L>>> {
    inner: I,
    config: DiverseSamplerConfig,
    seen_hashes: HashSet<u64>,
    seen_features: HashSet<(L, usize, L)>,
}

impl<L: Language, I: Iterator<Item = Tree<L>>> DiverseSampler<L, I> {
    /// Create a new diverse sampler wrapping an existing iterator.
    pub fn new(inner: I, config: DiverseSamplerConfig) -> Self {
        Self {
            inner,
            config,
            seen_hashes: HashSet::new(),
            seen_features: HashSet::new(),
        }
    }

    /// Check if a term is novel enough to accept.
    fn is_novel(&self, term: &Tree<L>) -> (bool, u64, HashSet<(L, usize, L)>) {
        let hash = structural_hash(term);
        let features = extract_features(term);

        // Novel if we've never seen this exact structure
        if !self.seen_hashes.contains(&hash) {
            return (true, hash, features);
        }

        // Otherwise check feature novelty ratio
        let novel_count = features
            .iter()
            .filter(|f| !self.seen_features.contains(*f))
            .count();

        #[expect(clippy::cast_precision_loss)]
        let novelty_ratio = if features.is_empty() {
            0.0
        } else {
            novel_count as f64 / features.len() as f64
        };

        (
            novelty_ratio >= self.config.min_novelty_ratio,
            hash,
            features,
        )
    }

    /// Accept a term, updating seen hashes and features.
    fn accept(&mut self, hash: u64, features: HashSet<(L, usize, L)>) {
        self.seen_hashes.insert(hash);
        self.seen_features.extend(features);
    }

    /// Reset the diversity tracking state.
    pub fn reset(&mut self) {
        self.seen_hashes.clear();
        self.seen_features.clear();
    }

    pub fn seen_hashes(&self) -> &HashSet<u64> {
        &self.seen_hashes
    }

    pub fn seen_features(&self) -> &HashSet<(L, usize, L)> {
        &self.seen_features
    }
}

impl<L: Language, I: Iterator<Item = Tree<L>>> Iterator for DiverseSampler<L, I> {
    type Item = Tree<L>;

    fn next(&mut self) -> Option<Self::Item> {
        for _ in 0..self.config.max_attempts_per_sample {
            let term = self.inner.next()?;
            let (is_novel, hash, features) = self.is_novel(&term);
            if is_novel {
                self.accept(hash, features);
                return Some(term);
            }
        }
        None
    }
}

/// Compute a structural hash of a tree for diversity checking.
/// Trees with the same structure and labels will have the same hash.
#[must_use]
pub fn structural_hash<L: Language>(tree: &Tree<L>) -> u64 {
    let mut hasher = DefaultHashBuilder::default().build_hasher();
    hash_tree_rec(tree, &mut hasher);
    hasher.finish()
}

fn hash_tree_rec<L: Language, H: Hasher>(tree: &Tree<L>, hasher: &mut H) {
    tree.node().hash(hasher);
    tree.children().len().hash(hasher);
    for child in tree.children() {
        hash_tree_rec(child, hasher);
    }
}

/// Extract structural features from a tree for diversity measurement.
/// Returns bigrams of `(parent_label, child_index, child_label)`.
#[must_use]
pub fn extract_features<L: Language>(tree: &Tree<L>) -> HashSet<(L, usize, L)> {
    let mut features = HashSet::new();
    collect_features(tree, &mut features);
    features
}

fn collect_features<L: Language>(tree: &Tree<L>, features: &mut HashSet<(L, usize, L)>) {
    let parent = tree.node().clone();
    for (i, child) in tree.children().iter().enumerate() {
        features.insert((parent.clone(), i, child.node().clone()));
        collect_features(child, features);
    }
}
