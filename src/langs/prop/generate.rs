use crate::generator::{Grammar, Samplable};
use crate::utils::id0;

use super::Prop;

/// Grammar for random propositional terms.
impl Samplable for Prop {
    fn grammar(leaf_symbols: Option<Vec<Self>>) -> Grammar<Self> {
        Grammar::new(
            vec![
                leaf_symbols.unwrap_or_else(default_symbols),
                vec![Prop::Not(id0())],
                vec![
                    Prop::And([id0(), id0()]),
                    Prop::Or([id0(), id0()]),
                    Prop::Implies([id0(), id0()]),
                ],
            ],
            Vec::new(),
            Vec::new(),
        )
    }
}

fn default_symbols() -> Vec<Prop> {
    (0..20)
        .map(|i| Prop::Symbol(format!("x{i}").into()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::SizeUniformSampler;
    use egg::{AstSize, CostFunction};
    use hashbrown::HashSet;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn samples_have_exactly_the_target_size() {
        for target in [1, 2, 5, 10, 15, 30] {
            let sampler = SizeUniformSampler::<Prop>::new(target, None);
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            for expr in sampler.sample_many(&mut rng, 50) {
                assert_eq!(AstSize.cost_rec(&expr), target, "wrong size for {expr}");
            }
        }
    }

    #[test]
    fn sampling_is_diverse() {
        let sampler = SizeUniformSampler::<Prop>::new(12, None);
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
        let sampler = SizeUniformSampler::<Prop>::new(20, None);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let all = sampler
            .sample_many(&mut rng, 500)
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(" ");
        for op in ["&", "~", "|", "->"] {
            assert!(all.contains(op), "operator {op} never generated");
        }
    }

    #[test]
    fn size_one_is_always_a_leaf() {
        let sampler = SizeUniformSampler::<Prop>::new(1, None);
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for expr in sampler.sample_many(&mut rng, 20) {
            assert!(matches!(expr[expr.root()], Prop::Symbol(_)), "{expr}");
        }
    }

    #[test]
    fn custom_leaf_pool_is_respected() {
        let leaves = vec![Prop::Symbol("a".into()), Prop::Symbol("b".into())];
        let sampler = SizeUniformSampler::<Prop>::new(7, Some(leaves));
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        for expr in sampler.sample_many(&mut rng, 50) {
            let s = expr.to_string();
            assert!(!s.contains("x0"), "unexpected default symbol in {s}");
        }
    }
}
