use hashbrown::HashMap;

use crate::graph::EClass;
use crate::ids::{EClassId, ExprChildId, NatId, TypeChildId};
use crate::nodes::NatNode;
use crate::nodes::Label;
use crate::tree::TreeNode;

pub fn eid(i: usize) -> ExprChildId {
    ExprChildId::EClass(EClassId::new(i))
}

pub fn dummy_ty() -> TypeChildId {
    TypeChildId::Nat(NatId::new(0))
}

pub fn dummy_nat_nodes() -> HashMap<NatId, NatNode<String>> {
    let mut nats = HashMap::new();
    nats.insert(NatId::new(0), NatNode::leaf("0".to_owned()));
    nats
}

pub fn cfv(classes: Vec<EClass<String>>) -> HashMap<EClassId, EClass<String>> {
    classes
        .into_iter()
        .enumerate()
        .map(|(i, c)| (EClassId::new(i), c))
        .collect()
}

pub fn leaf<L: Label>(label: L) -> TreeNode<L> {
    TreeNode::leaf_untyped(label)
}

pub fn node<L: Label>(label: L, children: Vec<TreeNode<L>>) -> TreeNode<L> {
    TreeNode::new_untyped(label, children)
}
