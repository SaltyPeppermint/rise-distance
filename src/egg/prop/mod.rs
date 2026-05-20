mod generate;

use std::sync::LazyLock;

use egg::{
    Analysis, DidMerge, EGraph, Id, PatternAst, RecExpr, Rewrite, Subst, Symbol, define_language,
    merge_option, rewrite,
};
use serde::{Deserialize, Serialize};

use crate::{MyAnalysis, MyLanguage, OriginLang};

pub use generate::BoltzmannSampler;

pub static RULES: LazyLock<Vec<Rewrite<Prop, ConstantFold>>> = LazyLock::new(rules);

define_language! {
    #[derive(Deserialize, Serialize)]
    pub enum Prop {
        Bool(bool),
        "&" = And([Id; 2]),
        "~" = Not(Id),
        "|" = Or([Id; 2]),
        "->" = Implies([Id; 2]),
        Symbol(Symbol),
    }
}

impl MyLanguage for Prop {
    fn type_of() -> Self {
        panic!("No types to see here");
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ConstantFold;

impl MyAnalysis<Prop> for ConstantFold {
    fn is_typed(_id: Id) -> bool {
        false
    }

    fn ty(_id: Id) -> Option<RecExpr<OriginLang<Prop>>> {
        None
    }
}

impl Analysis<Prop> for ConstantFold {
    type Data = Option<(bool, PatternAst<Prop>)>;
    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn make(egraph: &mut EGraph<Prop, ConstantFold>, enode: &Prop, _id: Id) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|c| c.0);
        match enode {
            Prop::Bool(c) => Some((*c, c.to_string().parse().unwrap())),
            Prop::Symbol(_) => None,
            Prop::And([a, b]) => Some((
                x(a)? && x(b)?,
                format!("(& {} {})", x(a)?, x(b)?).parse().unwrap(),
            )),
            Prop::Not(a) => Some((!x(a)?, format!("(~ {})", x(a)?).parse().unwrap())),
            Prop::Or([a, b]) => Some((
                x(a)? || x(b)?,
                format!("(| {} {})", x(a)?, x(b)?).parse().unwrap(),
            )),
            Prop::Implies([a, b]) => Some((
                !x(a)? || x(b)?,
                format!("(-> {} {})", x(a)?, x(b)?).parse().unwrap(),
            )),
        }
    }

    fn modify(egraph: &mut EGraph<Prop, ConstantFold>, id: Id) {
        if let Some(c) = egraph[id].data.clone() {
            egraph.union_instantiations(
                &c.1,
                &c.0.to_string().parse().unwrap(),
                &Subst::default(),
                "analysis".to_owned(),
            );
        }
    }
}

#[rustfmt::skip]
fn rules() -> Vec<Rewrite<Prop, ConstantFold>> {
    let mut rs = vec![
        rewrite!("assoc_or";       "(| ?a (| ?b ?c))" => "(| (| ?a ?b) ?c)"),
        rewrite!("dist_and_or";    "(& ?a (| ?b ?c))" => "(| (& ?a ?b) (& ?a ?c))"),
        rewrite!("dist_or_and";    "(| ?a (& ?b ?c))" => "(& (| ?a ?b) (| ?a ?c))"),
        rewrite!("comm_or";        "(| ?a ?b)"        => "(| ?b ?a)"),
        rewrite!("comm_and";       "(& ?a ?b)"        => "(& ?b ?a)"),
        rewrite!("lem";            "(| ?a (~ ?a))"    => "true"),
        rewrite!("or_true";        "(| ?a true)"      => "true"),
        rewrite!("and_true";       "(& ?a true)"      => "?a"),
        rewrite!("contrapositive"; "(-> ?a ?b)"       => "(-> (~ ?b) (~ ?a))"),
    ];
    rs.extend(rewrite!("def_imply";  "(-> ?a ?b)" <=> "(| (~ ?a) ?b)"));
    rs.extend(rewrite!("double_neg"; "(~ (~ ?a))" <=> "?a"));
    rs
}

// // this has to be a multipattern since (& (-> ?a ?b) (-> (~ ?a) ?c))  !=  (| ?b ?c)
// // see https://github.com/egraphs-good/egg/issues/185
// fn lem_imply() -> Rewrite<Prop, ConstantFold> {
//     multi_rewrite!(
//         "lem_imply";
//         "?value = true = (& (-> ?a ?b) (-> (~ ?a) ?c))"
//         =>
//         "?value = (| ?b ?c)"
//     )
// }

// fn prove_something(
//     name: &str,
//     start: &str,
//     mut rewrites: Vec<Rewrite<Prop, ConstantFold>>,
//     goals: &[&str],
// ) {
//     let _ = env_logger::builder().is_test(true).try_init();
//     println!("Proving {name}");

//     let start_expr: RecExpr<_> = start.parse().unwrap();
//     let goal_exprs: Vec<RecExpr<_>> = goals.iter().map(|g| g.parse().unwrap()).collect();

//     let mut runner = Runner::default()
//         .with_iter_limit(20)
//         .with_node_limit(5_000)
//         .with_expr(&start_expr);

//     // we are assume the input expr is true
//     // this is needed for the soundness of lem_imply
//     rewrites.push(lem_imply());
//     let true_id = runner.egraph.add(Prop::Bool(true));
//     let root = runner.roots[0];
//     runner.egraph.union(root, true_id);
//     runner.egraph.rebuild();

//     let egraph = runner.run(&rewrites).egraph;

//     for (i, (goal_expr, goal_str)) in goal_exprs.iter().zip(goals).enumerate() {
//         println!("Trying to prove goal {i}: {goal_str}");
//         let equivs = egraph.equivs(&start_expr, goal_expr);
//         assert!(!equivs.is_empty(), "Couldn't prove goal {i}: {goal_str}");
//     }
// }
