use egg::{EClass, EGraph, Id, RecExpr};
use std::collections::{HashMap, VecDeque};

use crate::{MyAnalysis, MyLanguage, OriginLang, stack_children};

/// Tracks novelty information for one eqsat iteration transition.
pub struct NoveltyInfo<'a, L: MyLanguage, N: MyAnalysis<L>> {
    pub prev: &'a EGraph<L, N>,
    pub curr: &'a EGraph<L, N>,

    /// For each (current canonical class id, node index): true iff the e-node
    /// was NOT representable at the prev-canonical of its current class in `prev`.
    pub node_is_novel: HashMap<(Id, usize), bool>,

    /// True iff some extraction starting at this current class can reach a novel e-node.
    pub can_reach_novelty: HashMap<Id, bool>,
}

impl<'a, L: MyLanguage, N: MyAnalysis<L>> NoveltyInfo<'a, L, N> {
    pub fn compute(prev: &'a EGraph<L, N>, curr: &'a EGraph<L, N>) -> Self {
        let node_is_novel = compute_node_novelty(prev, curr);
        let can_reach_novelty = compute_reachability(curr, &node_is_novel);
        Self {
            prev,
            curr,
            node_is_novel,
            can_reach_novelty,
        }
    }
}

/// Map a current class id to its prev-canonical id.
/// Returns None if the class didn't exist in the previous iteration.
fn prev_canonical_of<L: MyLanguage, N: MyAnalysis<L>>(prev: &EGraph<L, N>, id: Id) -> Option<Id> {
    if usize::from(id) >= prev.total_size() {
        None
    } else {
        Some(prev.find(id))
    }
}

/// Try to translate a current e-node into its prev-canonicalized form.
/// Returns None if any child class didn't exist in prev (in which case the
/// e-node certainly didn't exist in prev either).
fn prev_canonicalize_node<L: MyLanguage, N: MyAnalysis<L>>(
    prev: &EGraph<L, N>,
    node: &L,
) -> Option<L> {
    let mut translated = node.clone();
    let mut all_children_existed = true;
    translated.for_each_mut(|child| match prev_canonical_of(prev, *child) {
        Some(prev_id) => *child = prev_id,
        None => all_children_existed = false,
    });
    if all_children_existed {
        Some(translated)
    } else {
        None
    }
}

/// Decide novelty for every current e-node.
///
/// An e-node `n` in current class `c` was representable in prev iff:
///   - prev had a class for `c` (`prev_canonical_of` returns Some), AND
///   - every child of `n` had a prev-canonical, AND
///   - the prev-canonicalized e-node existed in prev's class equal to `prev_canonical_of(c)`.
fn compute_node_novelty<L: MyLanguage, N: MyAnalysis<L>>(
    prev: &EGraph<L, N>,
    curr: &EGraph<L, N>,
) -> HashMap<(Id, usize), bool> {
    let mut out = HashMap::with_capacity(curr.total_number_of_nodes());

    for class in curr.classes() {
        let curr_id = class.id;
        let prev_class_id = prev_canonical_of(prev, curr_id);

        for (idx, node) in class.nodes.iter().enumerate() {
            let novel = match prev_class_id {
                None => true,
                Some(prev_c) => match prev_canonicalize_node(prev, node) {
                    None => true,
                    Some(prev_node) => match prev.lookup(prev_node) {
                        None => true,
                        Some(found) => prev.find(found) != prev_c,
                    },
                },
            };
            out.insert((curr_id, idx), novel);
        }
    }

    out
}

/// Reachability fixpoint via worklist. A class can reach novelty iff it has
/// a novel e-node, or some e-node in it has a child that can reach novelty.
///
/// We seed the worklist with classes containing novel e-nodes, then propagate
/// upward through parent pointers (`EClass::parents` gives parent e-nodes;
/// each parent's containing class becomes a candidate to revisit).
fn compute_reachability<L: MyLanguage, N: MyAnalysis<L>>(
    curr: &EGraph<L, N>,
    node_is_novel: &HashMap<(Id, usize), bool>,
) -> HashMap<Id, bool> {
    let mut reach: HashMap<Id, bool> = curr.classes().map(|c| (c.id, false)).collect();
    let mut worklist: VecDeque<Id> = VecDeque::new();

    // Seed with classes having a novel e-node.
    for class in curr.classes() {
        let has_novel = (0..class.nodes.len())
            .any(|idx| *node_is_novel.get(&(class.id, idx)).unwrap_or(&false));
        if has_novel {
            reach.insert(class.id, true);
            worklist.push_back(class.id);
        }
    }

    // Propagate: if `c` reaches novelty, every class containing an e-node with
    // `c` as a child also reaches novelty.
    while let Some(c) = worklist.pop_front() {
        for parent_class_raw in curr[c].parents() {
            let parent_class = curr.find(parent_class_raw);
            if !reach[&parent_class] {
                reach.insert(parent_class, true);
                worklist.push_back(parent_class);
            }
        }
    }

    reach
}

// ============================================================================
// Constructive sampler with debt-passing.
// ============================================================================

pub fn sample_novel<L: MyLanguage, N: MyAnalysis<L>, R: rand::Rng>(
    info: &NoveltyInfo<L, N>,
    root: Id,
    rng: &mut R,
) -> Option<RecExpr<OriginLang<L>>> {
    let root = info.curr.find(root);
    if !info.can_reach_novelty.get(&root).copied().unwrap_or(false) {
        return None;
    }
    Some(sample_with_debt(info, root, true, rng))
}

fn sample_with_debt<L: MyLanguage, N: MyAnalysis<L>, R: rand::Rng>(
    info: &NoveltyInfo<L, N>,
    class_id: Id,
    must_hit_novelty: bool,
    rng: &mut R,
) -> RecExpr<OriginLang<L>> {
    use rand::seq::SliceRandom;

    let class = &info.curr[class_id];

    let candidates: Vec<usize> = if must_hit_novelty {
        (0..class.nodes.len())
            .filter(|&i| node_can_carry_debt(i, info, class_id, class))
            .collect()
    } else {
        (0..class.nodes.len()).collect()
    };

    let &chosen_idx = candidates
        .choose(rng)
        .expect("can_reach_novelty implies non-empty candidate set");
    let chosen = &class.nodes[chosen_idx];
    let novel_here = info.node_is_novel[&(class_id, chosen_idx)];

    let children_ids: Vec<Id> = chosen
        .children()
        .iter()
        .map(|&c| info.curr.find(c))
        .collect();
    let n_children = children_ids.len();

    let child_must_hit: Vec<bool> = if !must_hit_novelty || novel_here {
        vec![false; n_children]
    } else {
        let eligible: Vec<usize> = (0..n_children)
            .filter(|&i| {
                info.can_reach_novelty
                    .get(&children_ids[i])
                    .copied()
                    .unwrap_or(false)
            })
            .collect();
        let &debt_child = eligible
            .choose(rng)
            .expect("non-novel chosen node must have a debt-carrying child");
        (0..n_children).map(|i| i == debt_child).collect()
    };

    let sampled_children: Vec<RecExpr<OriginLang<L>>> = children_ids
        .iter()
        .enumerate()
        .map(|(i, &child_id)| sample_with_debt(info, child_id, child_must_hit[i], rng))
        .collect();

    stack_children(&sampled_children, OriginLang::new(chosen.clone(), class_id))
}

fn node_can_carry_debt<L: MyLanguage, N: MyAnalysis<L>>(
    idx: usize,
    info: &NoveltyInfo<'_, L, N>,
    class_id: Id,
    class: &EClass<L, <N as egg::Analysis<L>>::Data>,
) -> bool {
    if info.node_is_novel[&(class_id, idx)] {
        return true;
    }
    class.nodes[idx].children().iter().any(|&ch| {
        let ch_canon = info.curr.find(ch);
        info.can_reach_novelty
            .get(&ch_canon)
            .copied()
            .unwrap_or(false)
    })
}
