//! E-Graph data structure with tree extraction support.
//!
//! This module provides the core `EGraph` and `EClass` types for representing
//! equivalence graphs with type annotations.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use super::choices::ChoiceIter;
use super::ids::{
    DataId, EClassId, ExprChildId, FunId, NatId, NumericId, TypeChildId, eclass_id_vec,
    numeric_key_map,
};
use super::nodes::{DataTyNode, ENode, FunTyNode, Label, NatNode};
use super::tree::TreeNode;

/// `EClass`: choose exactly one child (`ENode`)
/// Children are `ENode` instances directly
/// Must have at least one child
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
#[serde(bound(deserialize = "L: Label"))]
pub struct EClass<L: Label> {
    nodes: Vec<ENode<L>>,
    ty: Option<TypeChildId>,
}

impl<L: Label> EClass<L> {
    #[must_use]
    pub fn new(nodes: Vec<ENode<L>>, ty: Option<TypeChildId>) -> Self {
        Self { nodes, ty }
    }

    #[must_use]
    pub fn nodes(&self) -> &[ENode<L>] {
        &self.nodes
    }
    #[must_use]
    pub fn nodes_mut(&mut self) -> &mut Vec<ENode<L>> {
        &mut self.nodes
    }

    #[must_use]
    pub fn ty(&self) -> Option<&TypeChildId> {
        self.ty.as_ref()
    }
}

/// E-graph with type annotations for tree extraction and edit distance computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "L: Label"))]
pub struct EGraph<L: Label> {
    #[serde(with = "numeric_key_map")]
    classes: HashMap<EClassId, EClass<L>>,
    #[serde(skip)]
    root: Option<EClassId>,
    #[serde(rename = "unionFind", with = "eclass_id_vec")]
    union_find: Vec<EClassId>,
    #[serde(rename = "typeHashCon", with = "numeric_key_map")]
    fun_ty_nodes: HashMap<FunId, FunTyNode<L>>,
    #[serde(rename = "natHashCon", with = "numeric_key_map")]
    nat_nodes: HashMap<NatId, NatNode<L>>,
    #[serde(rename = "dataTypeHashCon", with = "numeric_key_map")]
    data_ty_nodes: HashMap<DataId, DataTyNode<L>>,
}

impl<L: Label> EGraph<L> {
    #[must_use]
    pub fn new(
        classes: HashMap<EClassId, EClass<L>>,
        root: EClassId,
        union_find: Vec<EClassId>,
        fun_ty_nodes: HashMap<FunId, FunTyNode<L>>,
        nat_nodes: HashMap<NatId, NatNode<L>>,
        data_ty_nodes: HashMap<DataId, DataTyNode<L>>,
    ) -> Self {
        Self {
            classes,
            root: Some(root),
            union_find,
            fun_ty_nodes,
            nat_nodes,
            data_ty_nodes,
        }
    }

    /// Set the root `EClassId` (useful after deserialization)
    /// Automatically canonicalized
    fn set_root(&mut self, root: EClassId) {
        self.root = Some(self.canonicalize(root));
    }

    /// Get the root `EClassId`.
    ///
    /// # Panics
    /// Panics if the root has not been set.
    #[must_use]
    pub fn root(&self) -> EClassId {
        self.root
            .expect("Root has not been set. This is necessary after deserializing the egraph!")
    }

    /// Parse an `EGraph` from a JSON file, extracting the root ID from the filename.
    ///
    /// Expects filename format: `..._root_<id>.json` (e.g., `ser_egraph_root_129.json`).
    ///
    /// # Panics
    /// Panics if the file cannot be read, parsed, or if the filename doesn't match the expected format.
    #[must_use]
    pub fn parse_from_file(file: &Path) -> EGraph<L> {
        let mut graph =
            serde_json::from_reader::<_, Self>(BufReader::new(File::open(file).unwrap())).unwrap();
        // Pattern: ..._root_123.json
        let stem = file.file_stem().unwrap().to_str().unwrap();
        let id_str = stem.split('_').next_back().unwrap();
        graph.set_root(EClassId::new(id_str.parse().unwrap()));
        graph
    }

    /// Canonicalize an `EClassId` through the union-find.
    #[must_use]
    pub fn canonicalize(&self, id: EClassId) -> EClassId {
        if self.union_find.is_empty() {
            return id;
        }
        let mut current = id;
        while self.union_find[current.to_index()] != current {
            current = self.union_find[current.to_index()];
        }
        current
    }

    #[must_use]
    /// Returns the corresponding `EClass`. Can take a non-canonical Id
    pub fn class(&self, id: EClassId) -> &EClass<L> {
        let canonical = self.canonicalize(id);
        &self.classes[&canonical]
    }

    #[must_use]
    pub fn fun_ty(&self, id: FunId) -> &FunTyNode<L> {
        &self.fun_ty_nodes[&id]
    }

    #[must_use]
    pub fn data_ty(&self, id: DataId) -> &DataTyNode<L> {
        &self.data_ty_nodes[&id]
    }

    #[must_use]
    pub fn nat(&self, id: NatId) -> &NatNode<L> {
        &self.nat_nodes[&id]
    }

    /// Returns an iterator over all e-class IDs in the graph.
    pub fn class_ids(&self) -> impl Iterator<Item = EClassId> + '_ {
        self.classes.keys().map(|k| self.canonicalize(*k))
    }

    pub fn nat_ids(&self) -> impl Iterator<Item = NatId> + '_ {
        self.nat_nodes.keys().copied()
    }

    pub fn fun_ids(&self) -> impl Iterator<Item = FunId> + '_ {
        self.fun_ty_nodes.keys().copied()
    }

    pub fn data_ids(&self) -> impl Iterator<Item = DataId> + '_ {
        self.data_ty_nodes.keys().copied()
    }

    #[must_use]
    pub fn union_find(&self) -> &[EClassId] {
        &self.union_find
    }

    #[must_use]
    pub fn fun_ty_nodes(&self) -> &HashMap<FunId, FunTyNode<L>> {
        &self.fun_ty_nodes
    }

    #[must_use]
    pub fn nat_nodes(&self) -> &HashMap<NatId, NatNode<L>> {
        &self.nat_nodes
    }

    #[must_use]
    pub fn data_ty_nodes(&self) -> &HashMap<DataId, DataTyNode<L>> {
        &self.data_ty_nodes
    }

    #[must_use]
    pub fn tree_from_choices(&self, id: EClassId, choices: &[usize]) -> TreeNode<L> {
        self.tree_from_choices_rec(id, 0, choices).0
    }

    fn tree_from_choices_rec(
        &self,
        id: EClassId,
        choice_idx: usize,
        choices: &[usize],
    ) -> (TreeNode<L>, usize) {
        let class = self.class(id);
        let choice = choices[choice_idx];
        let node = &class.nodes()[choice];

        let (children, curr_idx) = node.children().iter().fold(
            (Vec::new(), choice_idx),
            |(mut children, curr_idx): (Vec<_>, _), child_id| {
                let (child_tree, last_idx) = match child_id {
                    // With nats and dt we cannot make a choice so do not add anything to the choice
                    ExprChildId::Nat(nat_id) => (TreeNode::from_nat(self, *nat_id), curr_idx),
                    ExprChildId::Data(dt_id) => (TreeNode::from_data(self, *dt_id), curr_idx),
                    ExprChildId::EClass(eclass_id) => {
                        self.tree_from_choices_rec(*eclass_id, curr_idx + 1, choices)
                    }
                };
                children.push(child_tree);
                (children, last_idx)
            },
        );

        let eclass_tree = TreeNode::new_typed(
            node.label().clone(),
            children,
            TreeNode::from_eclass(self, id),
        );
        (eclass_tree, curr_idx)
    }

    #[must_use]
    pub fn count_trees(&self, max_revisits: usize) -> usize {
        super::choices::count_trees(self, max_revisits)
    }

    #[must_use]
    pub fn choice_iter(&self, max_revisits: usize) -> ChoiceIter<'_, L> {
        ChoiceIter::new(self, max_revisits)
    }
}

#[cfg(test)]
mod tests {

    use crate::ids::NumericId;
    use crate::test_utils::*;

    use super::*;

    /// Helper to build a simple graph with one class containing one node
    fn single_node_graph(label: &str) -> EGraph<String> {
        EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf(label.to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        )
    }

    #[test]
    fn choice_iter_single_leaf() {
        let graph = single_node_graph("a");
        let trees = graph
            .choice_iter(0)
            .map(|c| graph.tree_from_choices(graph.root(), &c))
            .collect::<Vec<_>>();

        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].label(), "a");
        assert!(trees[0].children().is_empty());
        assert_eq!(trees[0].ty().unwrap().label(), "0");
    }

    #[test]
    fn choice_iter_with_or_choice() {
        // Graph with one class containing two node choices
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let trees = graph
            .choice_iter(0)
            .map(|c| graph.tree_from_choices(graph.root(), &c))
            .collect::<Vec<_>>();
        assert_eq!(trees.len(), 2);

        let labels = trees.iter().map(|t| t.label().as_str()).collect::<Vec<_>>();
        assert!(labels.contains(&"a"));
        assert!(labels.contains(&"b"));
    }

    #[test]
    fn choice_iter_with_and_children() {
        // Graph: root class -> node with two child classes (each has one leaf)
        // Class 0: root, has node "a" pointing to classes 1 and 2
        // Class 1: leaf "b"
        // Class 2: leaf "c"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let trees = graph
            .choice_iter(0)
            .map(|c| graph.tree_from_choices(graph.root(), &c))
            .collect::<Vec<_>>();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].label(), "a");
        assert_eq!(trees[0].ty().unwrap().label(), "0");
        assert_eq!(trees[0].children().len(), 2);
        assert_eq!(trees[0].children()[0].label(), "b");
        assert_eq!(trees[0].children()[0].ty().unwrap().label(), "0");
        assert_eq!(trees[0].children()[1].label(), "c");
        assert_eq!(trees[0].children()[1].ty().unwrap().label(), "0");
    }

    #[test]
    fn choice_iter_with_cycle_no_revisits() {
        // Graph with a cycle: class 0 -> node -> class 0
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![
                    ENode::new("a".to_owned(), vec![eid(0)]), // points back to self
                    ENode::leaf("leaf".to_owned()),
                ],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // With 0 revisits, we can only take the leaf option
        let trees = graph
            .choice_iter(0)
            .map(|c| graph.tree_from_choices(graph.root(), &c))
            .collect::<Vec<_>>();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].label(), "leaf");
    }

    #[test]
    fn choice_iter_with_cycle_one_revisit() {
        // Graph with a cycle: class 0 -> node -> class 0
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![
                    ENode::new("rec".to_owned(), vec![eid(0)]), // points back to self
                    ENode::leaf("leaf".to_owned()),
                ],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // With 1 revisit, we can go one level deep
        let trees = graph
            .choice_iter(1)
            .map(|c| graph.tree_from_choices(graph.root(), &c))
            .collect::<Vec<_>>();

        // Should have: "leaf", "rec(leaf)", "rec(rec(leaf))"
        // Actually: at depth 0 we have 2 choices, at depth 1 we have 2 choices...
        // Let's verify we get more trees than with 0 revisits
        assert!(trees.len() > 1);

        // Check that we have the recursive structure
        let has_recursive = trees.iter().any(|t| t.label() == "rec");
        assert!(has_recursive);
    }

    #[test]
    fn tree_from_choices_single_leaf() {
        let graph = single_node_graph("a");
        let choices = vec![0];

        let tree = graph.tree_from_choices(EClassId::new(0), &choices);

        assert_eq!(tree.label(), "a");
        assert!(tree.children().is_empty());
        assert_eq!(tree.ty().unwrap().label(), "0");
    }

    #[test]
    fn tree_from_choices_or_choice_first() {
        // Graph with two OR choices
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![0];
        let tree = graph.tree_from_choices(EClassId::new(0), &choices);

        assert_eq!(tree.label(), "a");
    }

    #[test]
    fn tree_from_choices_or_choice_second() {
        // Graph with two OR choices
        let graph = EGraph::new(
            cfv(vec![EClass::new(
                vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                dummy_ty(),
            )]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![1];
        let tree = graph.tree_from_choices(EClassId::new(0), &choices);

        assert_eq!(tree.label(), "b");
    }

    #[test]
    fn tree_from_choices_with_and_children() {
        // Graph: root -> node with two child classes
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("a".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![0, 0, 0];
        let tree = graph.tree_from_choices(EClassId::new(0), &choices);

        assert_eq!(tree.label(), "a");
        assert_eq!(tree.ty().unwrap().label(), "0");
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].label(), "b");
        assert_eq!(tree.children()[0].ty().unwrap().label(), "0");
        assert_eq!(tree.children()[1].label(), "c");
        assert_eq!(tree.children()[1].ty().unwrap().label(), "0");
    }

    #[test]
    fn tree_from_choices_nested_or_choices() {
        // Graph with nested OR choices:
        // Class 0: root with two node options
        //   Node "x" -> points to Class 1
        //   Node "y" -> points to Class 1
        // Class 1: two leaf options "a" or "b"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![
                        ENode::new("x".to_owned(), vec![eid(1)]),
                        ENode::new("y".to_owned(), vec![eid(1)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Test x(a)
        let choices1 = vec![0, 0];
        let tree1 = graph.tree_from_choices(EClassId::new(0), &choices1);
        assert_eq!(tree1.label(), "x");
        assert_eq!(tree1.children()[0].label(), "a");

        // Test x(b)
        let choices2 = vec![0, 1];
        let tree2 = graph.tree_from_choices(EClassId::new(0), &choices2);
        assert_eq!(tree2.label(), "x");
        assert_eq!(tree2.children()[0].label(), "b");

        // Test y(a)
        let choices3 = vec![1, 0];
        let tree3 = graph.tree_from_choices(EClassId::new(0), &choices3);
        assert_eq!(tree3.label(), "y");
        assert_eq!(tree3.children()[0].label(), "a");

        // Test y(b)
        let choices4 = vec![1, 1];
        let tree4 = graph.tree_from_choices(EClassId::new(0), &choices4);
        assert_eq!(tree4.label(), "y");
        assert_eq!(tree4.children()[0].label(), "b");
    }

    #[test]
    fn tree_from_choices_multiple_and_children_with_or_choices() {
        // Graph with multiple AND children each having OR choices:
        // Class 0: root -> "p" with children [Class 1, Class 2]
        // Class 1: "a" or "b"
        // Class 2: "x" or "y"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("p".to_owned(), vec![eid(1), eid(2)])],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("x".to_owned()), ENode::leaf("y".to_owned())],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Test p(a, x)
        let choices1 = vec![0, 0, 0];
        let tree1 = graph.tree_from_choices(EClassId::new(0), &choices1);
        assert_eq!(tree1.label(), "p");
        assert_eq!(tree1.children()[0].label(), "a");
        assert_eq!(tree1.children()[1].label(), "x");

        // Test p(a, y)
        let choices2 = vec![0, 0, 1];
        let tree2 = graph.tree_from_choices(EClassId::new(0), &choices2);
        assert_eq!(tree2.label(), "p");
        assert_eq!(tree2.children()[0].label(), "a");
        assert_eq!(tree2.children()[1].label(), "y");

        // Test p(b, x)
        let choices3 = vec![0, 1, 0];
        let tree3 = graph.tree_from_choices(EClassId::new(0), &choices3);
        assert_eq!(tree3.label(), "p");
        assert_eq!(tree3.children()[0].label(), "b");
        assert_eq!(tree3.children()[1].label(), "x");

        // Test p(b, y)
        let choices4 = vec![0, 1, 1];
        let tree4 = graph.tree_from_choices(EClassId::new(0), &choices4);
        assert_eq!(tree4.label(), "p");
        assert_eq!(tree4.children()[0].label(), "b");
        assert_eq!(tree4.children()[1].label(), "y");
    }

    #[test]
    fn tree_from_choices_deep_nesting() {
        // Deep tree with choices at multiple levels
        // Class 0: root -> "a" with child [Class 1]
        // Class 1: "b1" or "b2", both with child [Class 2]
        // Class 2: "c1" or "c2"
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(vec![ENode::new("a".to_owned(), vec![eid(1)])], dummy_ty()),
                EClass::new(
                    vec![
                        ENode::new("b1".to_owned(), vec![eid(2)]),
                        ENode::new("b2".to_owned(), vec![eid(2)]),
                    ],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("c1".to_owned()), ENode::leaf("c2".to_owned())],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        // Test a(b1(c1))
        let choices1 = vec![0, 0, 0];
        let tree1 = graph.tree_from_choices(EClassId::new(0), &choices1);
        assert_eq!(tree1.label(), "a");
        assert_eq!(tree1.children()[0].label(), "b1");
        assert_eq!(tree1.children()[0].children()[0].label(), "c1");

        // Test a(b1(c2))
        let choices2 = vec![0, 0, 1];
        let tree2 = graph.tree_from_choices(EClassId::new(0), &choices2);
        assert_eq!(tree2.label(), "a");
        assert_eq!(tree2.children()[0].label(), "b1");
        assert_eq!(tree2.children()[0].children()[0].label(), "c2");

        // Test a(b2(c1))
        let choices3 = vec![0, 1, 0];
        let tree3 = graph.tree_from_choices(EClassId::new(0), &choices3);
        assert_eq!(tree3.label(), "a");
        assert_eq!(tree3.children()[0].label(), "b2");
        assert_eq!(tree3.children()[0].children()[0].label(), "c1");

        // Test a(b2(c2))
        let choices4 = vec![0, 1, 1];
        let tree4 = graph.tree_from_choices(EClassId::new(0), &choices4);
        assert_eq!(tree4.label(), "a");
        assert_eq!(tree4.children()[0].label(), "b2");
        assert_eq!(tree4.children()[0].children()[0].label(), "c2");
    }

    #[test]
    fn tree_from_choices_three_and_children() {
        // Test with three AND children
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![ENode::new("f".to_owned(), vec![eid(1), eid(2), eid(3)])],
                    dummy_ty(),
                ),
                EClass::new(vec![ENode::leaf("a".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("b".to_owned())], dummy_ty()),
                EClass::new(vec![ENode::leaf("c".to_owned())], dummy_ty()),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = vec![0, 0, 0, 0];
        let tree = graph.tree_from_choices(EClassId::new(0), &choices);

        assert_eq!(tree.label(), "f");
        assert_eq!(tree.ty().unwrap().label(), "0");
        assert_eq!(tree.children().len(), 3);
        assert_eq!(tree.children()[0].label(), "a");
        assert_eq!(tree.children()[1].label(), "b");
        assert_eq!(tree.children()[2].label(), "c");
    }

    #[test]
    fn tree_from_choices_matches_choice_iter() {
        // Verify that tree_from_choices produces correct trees for each choice vector
        let graph = EGraph::new(
            cfv(vec![
                EClass::new(
                    vec![
                        ENode::new("x".to_owned(), vec![eid(1)]),
                        ENode::leaf("y".to_owned()),
                    ],
                    dummy_ty(),
                ),
                EClass::new(
                    vec![ENode::leaf("a".to_owned()), ENode::leaf("b".to_owned())],
                    dummy_ty(),
                ),
            ]),
            EClassId::new(0),
            Vec::new(),
            HashMap::new(),
            dummy_nat_nodes(),
            HashMap::new(),
        );

        let choices = graph.choice_iter(0).collect::<Vec<_>>();

        // Should produce: x(a), x(b), y
        assert_eq!(choices.len(), 3);
        assert_eq!(choices[0], vec![0, 0]);
        assert_eq!(choices[1], vec![0, 1]);
        assert_eq!(choices[2], vec![1]);

        // Verify tree_from_choices produces expected trees
        let tree1 = graph.tree_from_choices(graph.root(), &choices[0]);
        assert_eq!(tree1.label(), "x");

        let tree2 = graph.tree_from_choices(graph.root(), &choices[1]);
        assert_eq!(tree2.label(), "x");

        let tree3 = graph.tree_from_choices(graph.root(), &choices[2]);
        assert_eq!(tree3.label(), "y");
    }

    #[test]
    fn deserialize_json_file() {
        let graph = EGraph::<String>::parse_from_file(Path::new(
            "data/egraph_jsons/ser_egraph_vectorization_SRL_step_2_iteration_0_root_150.json",
        ));

        // Verify root is correct
        assert_eq!(graph.root().to_index(), 150);

        // Verify we can access the root class
        let root_class = graph.class(graph.root());
        assert!(!root_class.nodes().is_empty());

        // Verify some classes exist
        assert!(!graph.classes.is_empty());

        // Verify nat nodes exist
        assert!(!graph.nat_nodes.is_empty());
    }
}
