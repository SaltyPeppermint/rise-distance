use hashbrown::HashMap;

use crate::graph::EGraph;
use crate::ids::{DataChildId, DataId, FunId, NatId, TypeChildId};
use crate::nodes::Label;

/// Cache for type node sizes to avoid repeated computation.
///
/// Built eagerly before parallelism via [`TypeSizeCache::build`] so that
/// the parallel phase only needs a shared `&TypeSizeCache` (no locking).
#[derive(Debug, Default, Clone)]
pub(crate) struct TypeSizeCache {
    nats: HashMap<NatId, usize>,
    data: HashMap<DataId, usize>,
    funs: HashMap<FunId, usize>,
}

impl TypeSizeCache {
    /// Pre-compute sizes for every nat, data, and fun type node in the e-graph.
    pub(crate) fn build<L: Label>(graph: &EGraph<L>) -> Self {
        let mut cache = Self::default();
        for id in graph.nat_ids() {
            cache.ensure_nat(graph, id);
        }
        for id in graph.data_ids() {
            cache.ensure_data(graph, id);
        }
        for id in graph.fun_ids() {
            cache.ensure_fun(graph, id);
        }
        cache
    }

    pub(crate) fn get_type_size(&self, id: TypeChildId) -> usize {
        match id {
            TypeChildId::Nat(nat_id) => self.nats[&nat_id],
            TypeChildId::Type(fun_id) => self.funs[&fun_id],
            TypeChildId::Data(data_id) => self.data[&data_id],
        }
    }

    pub(crate) fn get_nat_size(&self, id: NatId) -> usize {
        self.nats[&id]
    }

    pub(crate) fn get_data_size(&self, id: DataId) -> usize {
        self.data[&id]
    }

    // -- eager population helpers (called only during `build`) --

    fn ensure_nat<L: Label>(&mut self, graph: &EGraph<L>, id: NatId) {
        if self.nats.contains_key(&id) {
            return;
        }
        let node = graph.nat(id);
        for &child_id in node.children() {
            self.ensure_nat(graph, child_id);
        }
        let size = 1 + node
            .children()
            .iter()
            .map(|&c| self.nats[&c])
            .sum::<usize>();
        self.nats.insert(id, size);
    }

    fn ensure_data<L: Label>(&mut self, graph: &EGraph<L>, id: DataId) {
        if self.data.contains_key(&id) {
            return;
        }
        let node = graph.data_ty(id);
        for &child_id in node.children() {
            match child_id {
                DataChildId::Nat(nat_id) => self.ensure_nat(graph, nat_id),
                DataChildId::DataType(data_id) => self.ensure_data(graph, data_id),
            }
        }
        let size = 1 + node
            .children()
            .iter()
            .map(|&c| match c {
                DataChildId::Nat(nat_id) => self.nats[&nat_id],
                DataChildId::DataType(data_id) => self.data[&data_id],
            })
            .sum::<usize>();
        self.data.insert(id, size);
    }

    fn ensure_fun<L: Label>(&mut self, graph: &EGraph<L>, id: FunId) {
        if self.funs.contains_key(&id) {
            return;
        }
        let node = graph.fun_ty(id);
        for &child_id in node.children() {
            match child_id {
                TypeChildId::Nat(nat_id) => self.ensure_nat(graph, nat_id),
                TypeChildId::Data(data_id) => self.ensure_data(graph, data_id),
                TypeChildId::Type(fun_id) => self.ensure_fun(graph, fun_id),
            }
        }
        let size = 1 + node
            .children()
            .iter()
            .map(|&c| match c {
                TypeChildId::Nat(nat_id) => self.nats[&nat_id],
                TypeChildId::Data(data_id) => self.data[&data_id],
                TypeChildId::Type(fun_id) => self.funs[&fun_id],
            })
            .sum::<usize>();
        self.funs.insert(id, size);
    }
}
