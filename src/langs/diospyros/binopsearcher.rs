use egg::{Analysis, EGraph, Id, Pattern, Rewrite, SearchMatches, Searcher, Var, rewrite as rw};

use std::str::FromStr;

use super::searchutils::{all_matches_to_substs, vec_fold_op, vec_with_var};
use super::veclang::VecLang;

#[derive(Debug, PartialEq, Clone)]
pub struct BinOpSearcher {
    pub left_var: String,
    pub right_var: String,
    pub full_pattern: Pattern<VecLang>,
    pub vec_pattern: Pattern<VecLang>,
    pub op_pattern: Pattern<VecLang>,
    pub zero_pattern: Pattern<VecLang>,
}

#[must_use]
#[expect(clippy::missing_panics_doc)]
pub fn build_binop_or_zero_rule(op_str: &str, vec_str: &str) -> Rewrite<VecLang, ()> {
    let left_var = "a".to_owned();
    let right_var = "b".to_owned();
    let full_pattern = vec_fold_op(op_str, &left_var, &right_var)
        .parse::<Pattern<VecLang>>()
        .unwrap();

    let vec_pattern = vec_with_var("x").parse::<Pattern<VecLang>>().unwrap();

    let op_pattern = format!("({op_str} ?{left_var} ?{right_var})")
        .parse::<Pattern<VecLang>>()
        .unwrap();

    let zero_pattern = "0".parse::<Pattern<VecLang>>().unwrap();

    let applier: Pattern<VecLang> = format!(
        "({} {} {})",
        vec_str,
        vec_with_var(&left_var),
        vec_with_var(&right_var)
    )
    .parse()
    .unwrap();

    let searcher = BinOpSearcher {
        left_var,
        right_var,
        full_pattern,
        vec_pattern,
        op_pattern,
        zero_pattern,
    };

    rw!(format!("{}_binop_or_zero", op_str); { searcher } => { applier })
}

impl BinOpSearcher {}

// We want each lane (?x0, ?x1, ?x2, ?x3) to match either:
//     (<binop> ?a ?b)
//     0               here, map ?a -> 0, ?b -> 0
impl<A: Analysis<VecLang>> Searcher<VecLang, A> for BinOpSearcher {
    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<VecLang, A>,
        eclass: Id,
        limit: usize,
    ) -> Option<SearchMatches<'_, VecLang>> {
        let mut result = self.search_eclass(egraph, eclass)?;
        // egg asserts substs.len() <= limit after this call; truncate to satisfy it
        result.substs.truncate(limit);
        Some(result)
    }

    fn search_eclass(
        &self,
        egraph: &EGraph<VecLang, A>,
        eclass: Id,
    ) -> Option<SearchMatches<'_, VecLang>> {
        let matches = self.vec_pattern.search_eclass(egraph, eclass)?;

        // Now we know the eclass is a Vec. The question is: does it
        // match a pattern compatible with this binary operation?
        let mut new_substs = Vec::new();
        let zero_id = egraph.lookup(VecLang::Num(0)).unwrap();

        // For each set of substitutions
        for substs in &matches.substs {
            let mut all_matches_found = true;
            let mut new_substs_options = Vec::new();

            // For each variable in (?x0, ?x1, ?x2, ?x3)
            // We use the index i to disambiguate lanes, so we can have,
            // for example, ?a0 through ?a3
            for (i, vec_var) in self.vec_pattern.vars().iter().enumerate() {
                // TODO: abstract this out to be prettier
                let mut new_var_substs = Vec::new();

                // Check if that variable matches the binop
                let child_eclass = substs.get(*vec_var).unwrap();
                if let Some(op_match) = self.op_pattern.search_eclass(egraph, *child_eclass) {
                    for s in &op_match.substs {
                        let mut subs = Vec::new();
                        for op_var in &self.op_pattern.vars() {
                            let new_v = Var::from_str(&format!("{}{}", *op_var, i)).unwrap();
                            subs.push((new_v, *s.get(*op_var).unwrap()));
                        }
                        new_var_substs.push(subs);
                    }
                // This lane is just 0
                } else if self
                    .zero_pattern
                    .search_eclass(egraph, *child_eclass)
                    .is_some()
                {
                    // ?a and ?b  map to zero
                    let subs = vec![
                        (
                            Var::from_str(&format!("?{}{}", self.left_var, i)).unwrap(),
                            zero_id,
                        ),
                        (
                            Var::from_str(&format!("?{}{}", self.right_var, i)).unwrap(),
                            zero_id,
                        ),
                    ];
                    new_var_substs.push(subs);

                // This lane isn't compatible, so whole Vec not a match
                } else {
                    all_matches_found = false;
                    break;
                }
                new_substs_options.push(new_var_substs);
            }
            if all_matches_found {
                // Now there is at least one match, but we need to make
                // potentially > 1 subst as children are combinatorial
                let mut all_substs = all_matches_to_substs(&new_substs_options);
                new_substs.append(&mut all_substs);
            }
        }
        if new_substs.is_empty() {
            None
        } else {
            Some(SearchMatches {
                eclass: matches.eclass,
                substs: new_substs,
                ast: None,
            })
        }
    }

    fn vars(&self) -> Vec<Var> {
        self.full_pattern.vars()
    }
}
