use hashbrown::HashMap;

use crate::graph::Class;
use crate::ids::{EClassId, ExprChildId, NatId, TypeChildId};
use crate::nodes::Label;
use crate::nodes::NatNode;
use crate::tree::TypedTree;

pub fn eid(i: usize) -> ExprChildId {
    ExprChildId::EClass(EClassId::new(i))
}

#[expect(clippy::unnecessary_wraps)]
pub fn dummy_ty() -> Option<TypeChildId> {
    Some(TypeChildId::Nat(NatId::new(0)))
}

pub fn dummy_nat_nodes() -> HashMap<NatId, NatNode<String>> {
    let mut nats = HashMap::new();
    nats.insert(NatId::new(0), NatNode::leaf("0".to_owned()));
    nats
}

pub fn cfv(classes: Vec<Class<String>>) -> HashMap<EClassId, Class<String>> {
    classes
        .into_iter()
        .enumerate()
        .map(|(i, c)| (EClassId::new(i), c))
        .collect()
}

pub fn leaf<L: Label>(label: L) -> TypedTree<L> {
    TypedTree::leaf_untyped(label)
}

pub fn node<L: Label>(label: L, children: Vec<TypedTree<L>>) -> TypedTree<L> {
    TypedTree::new_untyped(label, children)
}
