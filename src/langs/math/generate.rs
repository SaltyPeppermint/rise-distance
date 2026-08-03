use egg::{Id, Language, RecExpr, Symbol};
use hashbrown::HashSet;
use num::FromPrimitive;
use num::rational::Ratio;

use crate::generator::{Grammar, Samplable};
use crate::utils::id0;

use super::Math;

/// Grammar for random math terms, including scoped `Diff`/`Integral` binders.
impl Samplable for Math {
    fn grammar(leaf_symbols: Option<Vec<Self>>) -> Grammar<Self> {
        let leaves = leaf_symbols.unwrap_or_else(default_symbols);
        let vars = leaves
            .iter()
            .filter(|l| matches!(l, Math::Symbol(_)))
            .cloned()
            .collect();

        Grammar::new(
            vec![
                leaves,
                vec![
                    Math::Ln(id0()),
                    Math::Sqrt(id0()),
                    Math::Sin(id0()),
                    Math::Cos(id0()),
                ],
                vec![
                    Math::Add([id0(), id0()]),
                    Math::Sub([id0(), id0()]),
                    Math::Mul([id0(), id0()]),
                    Math::Div([id0(), id0()]),
                    Math::Pow([id0(), id0()]),
                ],
            ],
            vars,
            vec![Math::Diff([id0(), id0()]), Math::Integral([id0(), id0()])],
        )
    }

    fn free_var_indices(grammar: &Grammar<Self>, expr: &RecExpr<Self>) -> Vec<usize> {
        let free = free_vars(expr, expr.root());
        grammar
            .vars
            .iter()
            .enumerate()
            .filter(|(_, v)| matches!(v, Math::Symbol(s) if free.contains(s)))
            .map(|(i, _)| i)
            .collect()
    }
}

/// Collect the expression's free variables.
fn free_vars(expr: &RecExpr<Math>, id: Id) -> HashSet<Symbol> {
    match expr[id] {
        Math::Symbol(s) => [s].into_iter().collect(),
        Math::Diff(_) | Math::Integral(_) => {
            let c_ids = expr[id].children();
            let mut vars = free_vars(expr, c_ids[0]);
            if let Math::Symbol(bound) = expr[c_ids[1]] {
                vars.remove(&bound);
            }
            vars
        }
        _ => expr[id]
            .children()
            .iter()
            .flat_map(|c_id| free_vars(expr, *c_id))
            .collect(),
    }
}

fn default_symbols() -> Vec<Math> {
    vec![
        Math::Symbol("x".into()),
        Math::Symbol("y".into()),
        Math::Constant(Ratio::from_i64(0).unwrap()),
        Math::Constant(Ratio::from_i64(1).unwrap()),
        Math::Constant(Ratio::from_i64(2).unwrap()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::SizeUniformSampler;
    use egg::{AstSize, CostFunction};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn binders_valid(expr: &RecExpr<Math>) -> bool {
        fn rec(expr: &RecExpr<Math>, id: Id) -> bool {
            if let Math::Diff(_) | Math::Integral(_) = expr[id] {
                let children = expr[id].children();
                let body = children[0];
                let Math::Symbol(var) = expr[children[1]] else {
                    return false;
                };
                free_vars(expr, body).contains(&var) && rec(expr, body)
            } else {
                expr[id].children().iter().all(|id| rec(expr, *id))
            }
        }
        rec(expr, expr.root())
    }

    #[test]
    fn samples_have_exactly_the_target_size() {
        for target in [3, 5, 10, 15, 30] {
            let sampler = SizeUniformSampler::<Math>::new(target, None);
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            for expr in sampler.sample_many(&mut rng, 50) {
                assert_eq!(AstSize.cost_rec(&expr), target, "wrong size for {expr}");
            }
        }
    }

    #[test]
    fn binders_are_valid_by_construction() {
        let sampler = SizeUniformSampler::<Math>::new(15, None);
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        for expr in sampler.sample_many(&mut rng, 200) {
            assert!(
                binders_valid(&expr),
                "binder with non-free bound variable: {expr}"
            );
        }
    }

    #[test]
    fn binder_child1_is_always_a_variable() {
        let sampler = SizeUniformSampler::<Math>::new(15, None);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for expr in sampler.sample_many(&mut rng, 200) {
            for id in expr.ids() {
                if let Math::Diff(_) | Math::Integral(_) = expr[id] {
                    let children = expr[id].children();
                    assert!(
                        matches!(expr[children[1]], Math::Symbol(_)),
                        "binder child[1] is not a Symbol: {:?}",
                        expr[children[1]]
                    );
                }
            }
        }
    }

    #[test]
    fn sampling_is_diverse() {
        let sampler = SizeUniformSampler::<Math>::new(12, None);
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let distinct = sampler
            .sample_many(&mut rng, 200)
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<HashSet<_>>();
        assert!(distinct.len() > 190, "only {} distinct", distinct.len());
    }

    #[test]
    fn all_operators_are_reachable() {
        let sampler = SizeUniformSampler::<Math>::new(20, None);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let all = sampler
            .sample_many(&mut rng, 500)
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        for op in [
            "d", "i", "+", "-", "*", "/", "pow", "ln", "sqrt", "sin", "cos",
        ] {
            assert!(all.contains(op), "operator {op} never generated");
        }
    }

    fn parse(s: &str) -> RecExpr<Math> {
        s.parse().unwrap()
    }

    #[test]
    fn free_vars_single_symbol() {
        let expr = parse("x");
        assert_eq!(
            free_vars(&expr, expr.root()),
            ["x".into()].into_iter().collect()
        );
    }

    #[test]
    fn free_vars_binder_removes_bound_var() {
        let expr = parse("(d x x)");
        assert!(free_vars(&expr, expr.root()).is_empty());
    }

    #[test]
    fn free_vars_binder_keeps_other_vars() {
        let expr = parse("(d (+ x y) x)");
        let fv = free_vars(&expr, expr.root());
        assert!(!fv.contains(&Symbol::from("x")), "x should be bound");
        assert!(fv.contains(&Symbol::from("y")), "y should be free");
    }

    #[test]
    fn free_vars_nested_binders() {
        let expr = parse("(d (d (+ x y) x) y)");
        assert!(free_vars(&expr, expr.root()).is_empty());
    }

    #[test]
    fn free_vars_no_binders_collects_all_symbols() {
        let expr = parse("(+ x y)");
        assert_eq!(
            free_vars(&expr, expr.root()),
            ["x".into(), "y".into()].into_iter().collect()
        );
    }

    #[test]
    fn free_var_indices_maps_into_the_var_pool() {
        let grammar = Math::grammar(None);
        let idx = Math::free_var_indices(&grammar, &parse("(d (+ x y) x)"));
        assert_eq!(idx.len(), 1);
        assert_eq!(grammar.vars[idx[0]], Math::Symbol("y".into()));
    }

    #[test]
    fn free_var_indices_empty_for_constant_terms() {
        let grammar = Math::grammar(None);
        assert!(Math::free_var_indices(&grammar, &parse("(+ 1 2)")).is_empty());
    }

    #[test]
    fn binders_valid_simple_valid() {
        assert!(binders_valid(&parse("(d x x)")));
    }

    #[test]
    fn binders_valid_simple_invalid() {
        assert!(!binders_valid(&parse("(d x y)")));
    }

    #[test]
    fn binders_valid_no_binders() {
        assert!(binders_valid(&parse("(+ x y)")));
    }

    #[test]
    fn binders_valid_nested_outer_invalid() {
        assert!(!binders_valid(&parse("(d (d x x) y)")));
    }

    #[test]
    fn default_symbols_contains_expected_leaves() {
        let syms = default_symbols();
        assert!(syms.contains(&Math::Symbol("x".into())));
        assert!(syms.contains(&Math::Symbol("y".into())));
        assert!(syms.contains(&Math::Constant(Ratio::from_i64(0).unwrap())));
        assert!(syms.contains(&Math::Constant(Ratio::from_i64(1).unwrap())));
        assert!(syms.contains(&Math::Constant(Ratio::from_i64(2).unwrap())));
        assert_eq!(syms.len(), 5);
    }
}
