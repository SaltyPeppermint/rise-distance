use egg::{
    Analysis, EGraph, Id, Pattern, Rewrite, SearchMatches, Searcher, Subst, Var, rewrite as rw,
};

use std::str::FromStr;

use super::VECTOR_WIDTH;
use super::searchutils::{all_matches_to_substs, vec_with_var};
use super::veclang::VecLang;

#[derive(Debug, PartialEq, Clone)]
pub struct MacSearcher {
    pub acc_var: String,
    pub left_var: String,
    pub right_var: String,
    pub full_mac_pattern: Pattern<VecLang>,
    pub vec_pattern: Pattern<VecLang>,
    pub add_mul_pattern1: Pattern<VecLang>,
    pub add_mul_pattern2: Pattern<VecLang>,
    pub mul_pattern: Pattern<VecLang>,
    pub zero_pattern: Pattern<VecLang>,
}

#[must_use]
#[expect(clippy::missing_panics_doc)]
pub fn build_mac_rule() -> Rewrite<VecLang, ()> {
    let acc_var = "a".to_owned();
    let left_var = "b".to_owned();
    let right_var = "c".to_owned();

    let mut lanes: Vec<String> = Vec::with_capacity(VECTOR_WIDTH);
    for i in 0..VECTOR_WIDTH {
        lanes.push(format!(
            "(+ ?{acc_var}{i} (* ?{left_var}{i} ?{right_var}{i}))"
        ));
    }
    let full_mac_pattern = format!("(Vec {})", lanes.join(" "))
        .parse::<Pattern<VecLang>>()
        .unwrap();

    let vec_pattern = vec_with_var("x").parse::<Pattern<VecLang>>().unwrap();

    let add_mul_pattern1 = format!("(+ ?{acc_var} (* ?{left_var} ?{right_var}))")
        .parse::<Pattern<VecLang>>()
        .unwrap();

    let add_mul_pattern2 = format!("(+ (* ?{left_var} ?{right_var}) ?{acc_var})")
        .parse::<Pattern<VecLang>>()
        .unwrap();

    let mul_pattern = format!("(* ?{left_var} ?{right_var})")
        .parse::<Pattern<VecLang>>()
        .unwrap();

    let zero_pattern = "0".parse::<Pattern<VecLang>>().unwrap();

    let applier: Pattern<VecLang> = format!(
        "(VecMAC {} {} {})",
        vec_with_var(&acc_var),
        vec_with_var(&left_var),
        vec_with_var(&right_var)
    )
    .parse()
    .unwrap();

    let searcher = MacSearcher {
        acc_var,
        left_var,
        right_var,
        full_mac_pattern,
        vec_pattern,
        add_mul_pattern1,
        add_mul_pattern2,
        mul_pattern,
        zero_pattern,
    };

    rw!("vec-mac"; { searcher } => { applier })
}

impl MacSearcher {}

// We want each lane (?w, ?x, ?y, ?z) in (Vec ?w ?x ?y ?z) to match either:
//     (+ ?a (* ?b ?c))
//     (* ?b ?c)           here, map ?a -> 0
//     0                   here, map ?a -> 0, ?b -> 0, ?c -> 0
impl<A: Analysis<VecLang>> Searcher<VecLang, A> for MacSearcher {
    fn search_eclass_with_limit(
        &self,
        egraph: &EGraph<VecLang, A>,
        eclass: Id,
        _limit: usize,
    ) -> Option<SearchMatches<'_, VecLang>> {
        self.search_eclass(egraph, eclass)
    }

    fn search_eclass(
        &self,
        egraph: &EGraph<VecLang, A>,
        eclass: Id,
    ) -> Option<SearchMatches<'_, VecLang>> {
        let matches = self.vec_pattern.search_eclass(egraph, eclass)?;

        // Now we know the eclass is a Vec. The question is: does it
        // match a pattern compatible with a MAC?
        let mut new_substs: Vec<Subst> = Vec::new();
        let zero_id = egraph.lookup(VecLang::Num(0)).unwrap();

        // For each set of substitutions
        for substs in &matches.substs {
            let mut all_matches_found = true;
            let mut new_substs_options = Vec::new();

            // For each variable (?x0, ?x1, ?x2, ?x3)
            // We use the index i to disambiguate lanes, so we can have,
            // for example, ?a0 through ?a3
            for (i, vec_var) in self.vec_pattern.vars().iter().enumerate() {
                // TODO: abstract this out to be prettier
                let mut new_var_substs = Vec::new();

                // Check if that variable matches add/mul
                let child_eclass = substs.get(*vec_var).unwrap();
                if let Some(add_mul_match) =
                    self.add_mul_pattern1.search_eclass(egraph, *child_eclass)
                {
                    for s in &add_mul_match.substs {
                        let mut subs = Vec::new();
                        for add_mul_var in &self.add_mul_pattern1.vars() {
                            let new_v = Var::from_str(&format!("{}{}", *add_mul_var, i)).unwrap();
                            subs.push((new_v, *s.get(*add_mul_var).unwrap()));
                        }
                        new_var_substs.push(subs);
                    }
                } else if let Some(add_mul_match) =
                    self.add_mul_pattern2.search_eclass(egraph, *child_eclass)
                {
                    for s in &add_mul_match.substs {
                        let mut subs = Vec::new();
                        for add_mul_var in &self.add_mul_pattern2.vars() {
                            let new_v = Var::from_str(&format!("{}{}", *add_mul_var, i)).unwrap();
                            subs.push((new_v, *s.get(*add_mul_var).unwrap()));
                        }
                        new_var_substs.push(subs);
                    }
                // Check if that variable matches just a mul
                } else if let Some(mul_match) =
                    self.mul_pattern.search_eclass(egraph, *child_eclass)
                {
                    for s in &mul_match.substs {
                        let mut subs = Vec::new();
                        // for ?b and ?c
                        for mul_var in &self.mul_pattern.vars() {
                            let new_v = Var::from_str(&format!("{}{}", *mul_var, i)).unwrap();
                            subs.push((new_v, *s.get(*mul_var).unwrap()));
                        }
                        // ?a needs to map to a zero!
                        let var_a = Var::from_str(&format!("?{}{}", self.acc_var, i)).unwrap();
                        subs.push((var_a, zero_id));
                        new_var_substs.push(subs);
                    }
                // This lane is just 0
                } else if self
                    .zero_pattern
                    .search_eclass(egraph, *child_eclass)
                    .is_some()
                {
                    // ?a, ?b, and ?c all map to zero
                    let subs = vec![
                        (
                            Var::from_str(&format!("?{}{}", self.acc_var, i)).unwrap(),
                            zero_id,
                        ),
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

                // This lane isn't compatible, so the whole Vec can't
                // be a MAC
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
        self.full_mac_pattern.vars()
    }
}
