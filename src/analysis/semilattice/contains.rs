use egg::{Analysis, DidMerge, EGraph, Id, Language};

use super::SemiLatticeAnalysis;
use crate::utils::HashMap;

#[derive(Debug)]
pub struct SatisfiesContainsAnalysis;

impl<L: Language, N: Analysis<L>> SemiLatticeAnalysis<L, N> for SatisfiesContainsAnalysis {
    type Data = bool;

    fn make(
        &mut self,
        _egraph: &EGraph<L, N>,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data {
        enode.any(|c| analysis_of[&c])
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let r = *a || b;
        let dm = DidMerge(r != *a, r != b);
        *a = r;
        dm
    }
}

#[derive(Debug)]
pub struct SatisfiesOnlyContainsAnalysis;
impl<L: Language, A: Analysis<L>> SemiLatticeAnalysis<L, A> for SatisfiesOnlyContainsAnalysis {
    type Data = bool;

    fn make(
        &mut self,
        _egraph: &EGraph<L, A>,
        enode: &L,
        analysis_of: &HashMap<Id, Self::Data>,
    ) -> Self::Data {
        if enode.is_leaf() {
            return false;
        }
        enode.all(|c| analysis_of[&c])
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let r = *a || b;
        let dm = DidMerge(r != *a, r != b);
        *a = r;
        dm
    }
}
