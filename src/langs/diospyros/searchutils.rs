use egg::{Id, Subst, Var};

use super::config::VECTOR_WIDTH;

#[must_use]
pub fn vec_with_op(op: &str, pre: &str) -> String {
    let joined = ids_with_prefix(pre, VECTOR_WIDTH).join(" ");
    format!("({op} {joined})")
}

#[must_use]
pub fn vec_with_var(pre: &str) -> String {
    vec_with_op("Vec", pre)
}

#[must_use]
pub fn vec_fold_op(op: &str, pre_left: &str, pre_right: &str) -> String {
    let mut ops: Vec<String> = Vec::with_capacity(VECTOR_WIDTH);
    for i in 0..VECTOR_WIDTH {
        ops.push(format!("({op} ?{pre_left}{i} ?{pre_right}{i})"));
    }
    let joined = ops.join(" ");
    format!("(Vec {joined})")
}

#[must_use]
pub fn vec_map_op(op: &String, pre: &String) -> String {
    let mut ops: Vec<String> = Vec::with_capacity(VECTOR_WIDTH);
    for i in 0..VECTOR_WIDTH {
        ops.push(format!("({op} ?{pre}{i})"));
    }
    let joined = ops.join(" ");
    format!("(Vec {joined})")
}

#[must_use]
pub fn ids_with_prefix(pre: &str, count: usize) -> Vec<String> {
    let mut ids: Vec<String> = Vec::with_capacity(count);
    for i in 0..count {
        ids.push(format!("?{pre}{i}"));
    }
    ids
}

// Combinatorial combination of match children
#[must_use]
pub fn all_matches_to_substs(all_matches: &[Vec<Vec<(Var, Id)>>]) -> Vec<Subst> {
    match all_matches.first() {
        None => vec![Subst::with_capacity(12)],
        Some(var_substs) => {
            let mut new_substs: Vec<Subst> = Vec::new();
            let substs = all_matches_to_substs(&all_matches[1..]);
            for var_match in var_substs {
                for sub in &substs {
                    let mut sub_clone = sub.clone();
                    for (var, id) in var_match {
                        sub_clone.insert(*var, *id);
                    }
                    new_substs.push(sub_clone);
                }
            }
            new_substs
        }
    }
}
