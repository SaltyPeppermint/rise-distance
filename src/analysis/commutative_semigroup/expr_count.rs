use std::fmt::Debug;

use egg::{Analysis, DidMerge, EGraph, Id, Language};
use hashbrown::HashMap;

use super::CommutativeSemigroupAnalysis;
use crate::Counter;

#[derive(Debug, Copy, Clone)]
pub struct ExprCount {
    limit: usize,
}

impl ExprCount {
    #[must_use]
    pub const fn new(limit: usize) -> Self {
        Self { limit }
    }
}

impl<C, L, N> CommutativeSemigroupAnalysis<L, N, C> for ExprCount
where
    L: Language,
    N: Analysis<L>,
    C: Counter,
{
    // Size and number of programs of that size
    type Data = HashMap<usize, C>;

    fn make(
        &self,
        _egraph: &EGraph<L, N>,
        _eclass_id: Id,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data {
        let mut children_data = Vec::new();
        {
            for child_id in enode.children() {
                children_data.push(analysis_of.get(child_id).unwrap().clone());
            }
        }

        let mut tmp = Vec::new();

        children_data
            .into_iter()
            .fold(HashMap::from([(1, C::one())]), |mut acc, child_data| {
                tmp.extend(acc.drain());

                for (acc_size, acc_count) in &tmp {
                    for (child_size, child_count) in &child_data {
                        let combined_size = acc_size + child_size;
                        if combined_size > self.limit {
                            continue;
                        }
                        let combined_count = acc_count.to_owned() * child_count;
                        acc.entry(combined_size)
                            .and_modify(|c| *c += &combined_count)
                            .or_insert(combined_count);
                    }
                }

                tmp.clear();
                acc
            })
    }

    fn merge(&self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        if b.is_empty() {
            return DidMerge(false, false);
        }

        for (size, count) in b {
            a.entry(size)
                .and_modify(|c| {
                    *c += &count;
                })
                .or_insert(count);
        }
        DidMerge(true, false)
    }

    // fn new_data() -> Self::Data {
    //     Self::Data::new()
    // }

    // fn data_empty(data: &Self::Data) -> bool {
    //     data.is_empty()
    // }
}

#[cfg(test)]
mod tests {
    use egg::{EGraph, SymbolLang};
    use num::BigUint;

    use super::*;

    #[test]
    fn simple_term_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();

        let data = ExprCount::new(10).one_shot_analysis(&egraph);
        let root_data: &HashMap<usize, BigUint> = &data[&egraph.find(apb)];

        assert_eq!(root_data[&5], 1usize.into());
    }

    #[test]
    fn slightly_complicated_size_count() {
        let mut egraph = EGraph::<SymbolLang, ()>::default();
        let a = egraph.add(SymbolLang::leaf("a"));
        let b = egraph.add(SymbolLang::leaf("b"));
        let apb = egraph.add(SymbolLang::new("+", vec![a, b]));

        egraph.union(a, apb);
        egraph.rebuild();
        egraph.union(b, apb);
        egraph.rebuild();

        let data = ExprCount::new(10).one_shot_analysis(&egraph);

        let root_data: &HashMap<usize, BigUint> = &data[&egraph.find(apb)];
        assert_eq!(root_data[&5], 16usize.into());
    }
}
