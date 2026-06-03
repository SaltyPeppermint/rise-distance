use egg::{CostFunction, EGraph, Id, Language};

use super::veclang::VecLang;

pub struct VecCostFn<'a> {
    pub egraph: &'a EGraph<VecLang, ()>,
}

// &'a EGraph

impl CostFunction<VecLang> for VecCostFn<'_> {
    type Cost = f64;
    // you're passed in an enode whose children are costs instead of eclass ids
    #[expect(clippy::cast_precision_loss)]
    fn cost<C>(&mut self, enode: &VecLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        const LITERAL: f64 = 0.001;
        const STRUCTURE: f64 = 0.1;
        const VEC_OP: f64 = 1.;
        const OP: f64 = 1.;
        const BIG: f64 = 100.0;
        let op_cost = match enode {
            // You get literals for extremely cheap
            // And list structures for quite cheap
            VecLang::List(..) | VecLang::Concat(..) => STRUCTURE,

            // Vectors are cheap if they have literal values
            VecLang::Vec(vals) => {
                // For now, workaround to determine if children are num, symbol,
                // or get
                let non_literals = vals.iter().any(|&x| costs(x) > 3. * LITERAL);
                if non_literals { BIG } else { STRUCTURE }
            }
            VecLang::Num(..) | VecLang::Symbol(..) | VecLang::Get(..) | VecLang::LitVec(..) => {
                LITERAL
            }

            // But scalar and vector ops cost something
            VecLang::Add(vals) | VecLang::Mul(vals) | VecLang::Minus(vals) | VecLang::Div(vals) => {
                OP * (vals.len() as f64 - 1.)
            }
            VecLang::Sgn(..) | VecLang::Neg(..) | VecLang::Sqrt(..) => OP,
            _ => VEC_OP,
        };
        enode.fold(op_cost, |sum, id| sum + costs(id))
    }
}
