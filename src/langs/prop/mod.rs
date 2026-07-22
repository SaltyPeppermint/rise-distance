mod generate;

use egg::{
    Analysis, DidMerge, EGraph, Id, PatternAst, Rewrite, Subst, Symbol, define_language,
    merge_option, rewrite,
};
use serde::{Deserialize, Serialize};

pub use generate::PropSampler;

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

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct ConstantFold;

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
#[must_use]
pub fn rules() -> Vec<Rewrite<Prop, ConstantFold>> {
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
