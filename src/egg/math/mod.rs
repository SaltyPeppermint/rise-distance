mod generate;

use std::sync::LazyLock;

use egg::{
    Analysis, DidMerge, Id, Language, PatternAst, Rewrite, Subst, Symbol, define_language,
    merge_option, rewrite,
};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

pub use generate::BoltzmannSampler;

use crate::egg::ToEgg;
use crate::{Label, TreeShaped};
// pub use label::MathLabel;

pub type Constant = NotNan<f64>;

define_language! {
    #[derive(Deserialize,Serialize)]
    pub enum Math {
        "d" = Diff([Id; 2]),
        "i" = Integral([Id; 2]),

        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),
        "ln" = Ln(Id),
        "sqrt" = Sqrt(Id),

        "sin" = Sin(Id),
        "cos" = Cos(Id),

        Constant(Constant),
        Symbol(Symbol),
    }
}

#[derive(Default, Clone)]
pub struct ConstantFold;
impl Analysis<Math> for ConstantFold {
    type Data = Option<(Constant, PatternAst<Math>)>;

    fn make(egraph: &mut egg::EGraph<Math, ConstantFold>, enode: &Math, _id: Id) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0);
        Some(match enode {
            Math::Constant(c) => (*c, format!("{c}").parse().unwrap()),
            Math::Add([a, b]) => (
                x(a)? + x(b)?,
                format!("(+ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Sub([a, b]) => (
                x(a)? - x(b)?,
                format!("(- {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Mul([a, b]) => (
                x(a)? * x(b)?,
                format!("(* {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            Math::Div([a, b]) if x(b) != Some(NotNan::new(0.0).unwrap()) => (
                x(a)? / x(b)?,
                format!("(/ {} {})", x(a)?, x(b)?).parse().unwrap(),
            ),
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(a.0, b.0, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut egg::EGraph<Math, ConstantFold>, id: Id) {
        let data = egraph[id].data.clone();
        if let Some((c, pat)) = data {
            if egraph.are_explanations_enabled() {
                egraph.union_instantiations(
                    &pat,
                    &format!("{c}").parse().unwrap(),
                    &Subst::default(),
                    "constant_fold".to_owned(),
                );
            } else {
                let added = egraph.add(Math::Constant(c));
                egraph.union(id, added);
            }
            // to not prune, comment this out
            egraph[id].nodes.retain(|n| n.is_leaf());

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

fn is_const_or_distinct_var(
    v: &str,
    w: &str,
) -> impl Fn(&mut egg::EGraph<Math, ConstantFold>, Id, &Subst) -> bool {
    let v = v.parse().unwrap();
    let w = w.parse().unwrap();
    move |egraph, _, subst| {
        egraph.find(subst[v]) != egraph.find(subst[w])
            && (egraph[subst[v]].data.is_some()
                || egraph[subst[v]]
                    .nodes
                    .iter()
                    .any(|n| matches!(n, Math::Symbol(..))))
    }
}

fn is_const(var: &str) -> impl Fn(&mut egg::EGraph<Math, ConstantFold>, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| egraph[subst[var]].data.is_some()
}

fn is_sym(var: &str) -> impl Fn(&mut egg::EGraph<Math, ConstantFold>, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        egraph[subst[var]]
            .nodes
            .iter()
            .any(|n| matches!(n, Math::Symbol(..)))
    }
}

fn is_not_zero(var: &str) -> impl Fn(&mut egg::EGraph<Math, ConstantFold>, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        if let Some(n) = &egraph[subst[var]].data {
            *(n.0) != 0.0
        } else {
            true
        }
    }
}

pub static RULES: LazyLock<Vec<Rewrite<Math, ConstantFold>>> = LazyLock::new(rules);

impl Label for Math {
    fn type_of() -> Self {
        panic!("No types to see here");
    }
}

impl<T: TreeShaped<Math>> ToEgg<Math> for T {
    fn add_node<F: FnMut(&Self, Math) -> Id>(&self, adder: &mut F) -> Id {
        let child_ids = self
            .children()
            .iter()
            .map(|c| c.add_node(adder))
            .collect::<Vec<_>>();
        let math_node = match self.label() {
            Math::Diff(_) => Math::Diff([child_ids[0], child_ids[1]]),
            Math::Integral(_) => Math::Integral([child_ids[0], child_ids[1]]),
            Math::Add(_) => Math::Add([child_ids[0], child_ids[1]]),
            Math::Sub(_) => Math::Sub([child_ids[0], child_ids[1]]),
            Math::Mul(_) => Math::Mul([child_ids[0], child_ids[1]]),
            Math::Div(_) => Math::Div([child_ids[0], child_ids[1]]),
            Math::Pow(_) => Math::Pow([child_ids[0], child_ids[1]]),
            Math::Ln(_) => Math::Ln(child_ids[0]),
            Math::Sqrt(_) => Math::Sqrt(child_ids[0]),
            Math::Sin(_) => Math::Sin(child_ids[0]),
            Math::Cos(_) => Math::Cos(child_ids[0]),
            Math::Constant(c) => Math::Constant(*c),
            Math::Symbol(s) => Math::Symbol(*s),
        };
        adder(self, math_node)
    }
}

#[rustfmt::skip]
#[must_use]
pub fn rules() -> Vec<Rewrite<Math, ConstantFold>> { vec![
    rewrite!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
    rewrite!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
    rewrite!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
    rewrite!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),

    rewrite!("sub-canon"; "(- ?a ?b)" => "(+ ?a (* -1 ?b))"),
    rewrite!("div-canon"; "(/ ?a ?b)" => "(* ?a (pow ?b -1))" if is_not_zero("?b")),
    // rewrite!("canon-sub"; "(+ ?a (* -1 ?b))"   => "(- ?a ?b)"),
    // rewrite!("canon-div"; "(* ?a (pow ?b -1))" => "(/ ?a ?b)" if is_not_zero("?b")),

    rewrite!("zero-add"; "(+ ?a 0)" => "?a"),
    rewrite!("zero-mul"; "(* ?a 0)" => "0"),
    rewrite!("one-mul";  "(* ?a 1)" => "?a"),

    rewrite!("add-zero"; "?a" => "(+ ?a 0)"),
    rewrite!("mul-one";  "?a" => "(* ?a 1)"),

    rewrite!("cancel-sub"; "(- ?a ?a)" => "0"),
    rewrite!("cancel-div"; "(/ ?a ?a)" => "1" if is_not_zero("?a")),

    rewrite!("distribute"; "(* ?a (+ ?b ?c))"        => "(+ (* ?a ?b) (* ?a ?c))"),
    rewrite!("factor"    ; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),

    rewrite!("pow-mul"; "(* (pow ?a ?b) (pow ?a ?c))" => "(pow ?a (+ ?b ?c))"),
    rewrite!("pow0"; "(pow ?x 0)" => "1"
        if is_not_zero("?x")),
    rewrite!("pow1"; "(pow ?x 1)" => "?x"),
    rewrite!("pow2"; "(pow ?x 2)" => "(* ?x ?x)"),
    rewrite!("pow-recip"; "(pow ?x -1)" => "(/ 1 ?x)"
        if is_not_zero("?x")),
    rewrite!("recip-mul-div"; "(* ?x (/ 1 ?x))" => "1" if is_not_zero("?x")),

    rewrite!("d-variable"; "(d ?x ?x)" => "1" if is_sym("?x")),
    rewrite!("d-constant"; "(d ?x ?c)" => "0" if is_sym("?x") if is_const_or_distinct_var("?c", "?x")),

    rewrite!("d-add"; "(d ?x (+ ?a ?b))" => "(+ (d ?x ?a) (d ?x ?b))"),
    rewrite!("d-mul"; "(d ?x (* ?a ?b))" => "(+ (* ?a (d ?x ?b)) (* ?b (d ?x ?a)))"),

    rewrite!("d-sin"; "(d ?x (sin ?x))" => "(cos ?x)"),
    rewrite!("d-cos"; "(d ?x (cos ?x))" => "(* -1 (sin ?x))"),

    rewrite!("d-ln"; "(d ?x (ln ?x))" => "(/ 1 ?x)" if is_not_zero("?x")),

    rewrite!("d-power";
        "(d ?x (pow ?f ?g))" =>
        "(* (pow ?f ?g)
            (+ (* (d ?x ?f)
                  (/ ?g ?f))
               (* (d ?x ?g)
                  (ln ?f))))"
        if is_not_zero("?f")
        if is_not_zero("?g")
    ),

    rewrite!("i-one"; "(i 1 ?x)" => "?x"),
    rewrite!("i-power-const"; "(i (pow ?x ?c) ?x)" =>
        "(/ (pow ?x (+ ?c 1)) (+ ?c 1))" if is_const("?c")),
    rewrite!("i-cos"; "(i (cos ?x) ?x)" => "(sin ?x)"),
    rewrite!("i-sin"; "(i (sin ?x) ?x)" => "(* -1 (cos ?x))"),
    rewrite!("i-sum"; "(i (+ ?f ?g) ?x)" => "(+ (i ?f ?x) (i ?g ?x))"),
    rewrite!("i-dif"; "(i (- ?f ?g) ?x)" => "(- (i ?f ?x) (i ?g ?x))"),
    rewrite!("i-parts"; "(i (* ?a ?b) ?x)" =>
        "(- (* ?a (i ?b ?x)) (i (* (d ?x ?a) (i ?b ?x)) ?x))"),
]}

#[cfg(test)]
mod tests {
    use egg::{RecExpr, Runner, SimpleScheduler, StopReason};

    use crate::{TypedTree, egg::id0};

    use super::*;

    egg::test_fn! {
        math_associate_adds, [
            rewrite!("comm-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
            rewrite!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        ],
        runner = Runner::default()
            .with_iter_limit(7)
            .with_scheduler(SimpleScheduler),
        "(+ 1 (+ 2 (+ 3 (+ 4 (+ 5 (+ 6 7))))))"
        =>
        "(+ 7 (+ 6 (+ 5 (+ 4 (+ 3 (+ 2 1))))))"
        @check |r: Runner<Math, ()>| assert_eq!(r.egraph.number_of_classes(), 127)
    }

    egg::test_fn! {
        #[should_panic(expected = "Could not prove goal 0")]
        math_fail, rules(),
        "(+ x y)" => "(/ x y)"
    }

    egg::test_fn! {math_simplify_add, rules(), "(+ x (+ x (+ x x)))" => "(* 4 x)" }
    egg::test_fn! {math_powers, rules(), "(* (pow 2 x) (pow 2 y))" => "(pow 2 (+ x y))"}

    egg::test_fn! {
        math_simplify_const, rules(),
        "(+ 1 (- a (* (- 2 1) a)))" => "1"
    }

    egg::test_fn! {
        math_simplify_root, rules(),
        runner = Runner::default().with_node_limit(75_000),
        "
    (/ 1
       (- (/ (+ 1 (sqrt five))
             2)
          (/ (- 1 (sqrt five))
             2)))"
        =>
        "(/ 1 (sqrt five))"
    }

    egg::test_fn! {
        math_simplify_factor, rules(),
        "(* (+ x 3) (+ x 1))"
        =>
        "(+ (+ (* x x) (* 4 x)) 3)"
    }

    egg::test_fn! {math_diff_same,      rules(), "(d x x)" => "1"}
    egg::test_fn! {math_diff_different, rules(), "(d x y)" => "0"}
    egg::test_fn! {math_diff_simple1,   rules(), "(d x (+ 1 (* 2 x)))" => "2"}
    egg::test_fn! {math_diff_simple2,   rules(), "(d x (+ 1 (* y x)))" => "y"}
    egg::test_fn! {math_diff_ln,        rules(), "(d x (ln x))" => "(/ 1 x)"}

    egg::test_fn! {
        diff_power_simple, rules(),
        "(d x (pow x 3))" => "(* 3 (pow x 2))"
    }

    egg::test_fn! {
        diff_power_harder, rules(),
        runner = Runner::default()
            .with_time_limit(std::time::Duration::from_secs(10))
            .with_iter_limit(60)
            .with_node_limit(100_000)
            .with_explanations_enabled()
            // HACK this needs to "see" the end expression
            .with_expr(&"(* x (- (* 3 x) 14))".parse().unwrap()),
        "(d x (- (pow x 3) (* 7 (pow x 2))))"
        =>
        "(* x (- (* 3 x) 14))"
    }

    egg::test_fn! {
        integ_one, rules(), "(i 1 x)" => "x"
    }

    egg::test_fn! {
        integ_sin, rules(), "(i (cos x) x)" => "(sin x)"
    }

    egg::test_fn! {
        integ_x, rules(), "(i (pow x 1) x)" => "(/ (pow x 2) 2)"
    }

    egg::test_fn! {
        integ_part1, rules(), "(i (* x (cos x)) x)" => "(+ (* x (sin x)) (cos x))"
    }

    egg::test_fn! {
        integ_part2, rules(),
        "(i (* (cos x) x) x)" => "(+ (* x (sin x)) (cos x))"
    }

    egg::test_fn! {
        integ_part3, rules(), "(i (ln x) x)" => "(- (* x (ln x)) x)"
    }

    #[test]
    fn assoc_mul_saturates() {
        let expr: RecExpr<Math> = "(* x 1)".parse().unwrap();

        let runner: Runner<Math, ConstantFold> = Runner::default()
            .with_iter_limit(3)
            .with_expr(&expr)
            .run(&rules());

        assert!(matches!(runner.stop_reason, Some(StopReason::Saturated)));
    }

    #[test]
    fn test_union_trusted() {
        let expr: RecExpr<Math> = "(+ (* x 1) y)".parse().unwrap();
        let expr2 = "20".parse().unwrap();
        let mut runner: Runner<Math, ConstantFold> = Runner::default()
            .with_explanations_enabled()
            .with_iter_limit(3)
            .with_expr(&expr)
            .run(&rules());
        let lhs = runner.egraph.add_expr(&expr);
        let rhs = runner.egraph.add_expr(&expr2);
        runner.egraph.union_trusted(lhs, rhs, "whatever");
        let proof = runner.explain_equivalence(&expr, &expr2).get_flat_strings();
        assert_eq!(proof, vec!["(+ (* x 1) y)", "(Rewrite=> whatever 20)"]);
    }

    #[test]
    fn math_ematching_bench() {
        let exprs = &[
            "(i (ln x) x)",
            "(i (+ x (cos x)) x)",
            "(i (* (cos x) x) x)",
            "(d x (+ 1 (* 2 x)))",
            "(d x (- (pow x 3) (* 7 (pow x 2))))",
            "(+ (* y (+ x y)) (- (+ x 2) (+ x x)))",
            "(/ 1 (- (/ (+ 1 (sqrt five)) 2) (/ (- 1 (sqrt five)) 2)))",
        ];

        let extra_patterns = &[
            "(+ ?a (+ ?b ?c))",
            "(+ (+ ?a ?b) ?c)",
            "(* ?a (* ?b ?c))",
            "(* (* ?a ?b) ?c)",
            "(+ ?a (* -1 ?b))",
            "(* ?a (pow ?b -1))",
            "(* ?a (+ ?b ?c))",
            "(pow ?a (+ ?b ?c))",
            "(+ (* ?a ?b) (* ?a ?c))",
            "(* (pow ?a ?b) (pow ?a ?c))",
            "(* ?x (/ 1 ?x))",
            "(d ?x (+ ?a ?b))",
            "(+ (d ?x ?a) (d ?x ?b))",
            "(d ?x (* ?a ?b))",
            "(+ (* ?a (d ?x ?b)) (* ?b (d ?x ?a)))",
            "(d ?x (sin ?x))",
            "(d ?x (cos ?x))",
            "(* -1 (sin ?x))",
            "(* -1 (cos ?x))",
            "(i (cos ?x) ?x)",
            "(i (sin ?x) ?x)",
            "(d ?x (ln ?x))",
            "(d ?x (pow ?f ?g))",
            "(* (pow ?f ?g) (+ (* (d ?x ?f) (/ ?g ?f)) (* (d ?x ?g) (ln ?f))))",
            "(i (pow ?x ?c) ?x)",
            "(/ (pow ?x (+ ?c 1)) (+ ?c 1))",
            "(i (+ ?f ?g) ?x)",
            "(i (- ?f ?g) ?x)",
            "(+ (i ?f ?x) (i ?g ?x))",
            "(- (i ?f ?x) (i ?g ?x))",
            "(i (* ?a ?b) ?x)",
            "(- (* ?a (i ?b ?x)) (i (* (d ?x ?a) (i ?b ?x)) ?x))",
        ];

        egg::test::bench_egraph("math", rules(), exprs, extra_patterns);
    }

    #[test]
    fn test_basic_egraph_union_intersect() {
        let mut egraph1 = egg::EGraph::new(ConstantFold {}).with_explanations_enabled();
        let mut egraph2 = egg::EGraph::new(ConstantFold {}).with_explanations_enabled();
        egraph1.union_instantiations(
            &"x".parse().unwrap(),
            &"y".parse().unwrap(),
            &Subst::default(),
            "",
        );
        egraph1.union_instantiations(
            &"y".parse().unwrap(),
            &"z".parse().unwrap(),
            &Subst::default(),
            "",
        );
        egraph2.union_instantiations(
            &"x".parse().unwrap(),
            &"y".parse().unwrap(),
            &Subst::default(),
            "",
        );
        egraph2.union_instantiations(
            &"x".parse().unwrap(),
            &"a".parse().unwrap(),
            &Subst::default(),
            "",
        );

        let mut egraph3 = egraph1.egraph_intersect(&egraph2, ConstantFold {});

        egraph2.egraph_union(&egraph1);

        assert_eq!(
            egraph2.add_expr(&"x".parse().unwrap()),
            egraph2.add_expr(&"y".parse().unwrap())
        );
        assert_eq!(
            egraph3.add_expr(&"x".parse().unwrap()),
            egraph3.add_expr(&"y".parse().unwrap())
        );

        assert_eq!(
            egraph2.add_expr(&"x".parse().unwrap()),
            egraph2.add_expr(&"z".parse().unwrap())
        );
        assert_ne!(
            egraph3.add_expr(&"x".parse().unwrap()),
            egraph3.add_expr(&"z".parse().unwrap())
        );
        assert_eq!(
            egraph2.add_expr(&"x".parse().unwrap()),
            egraph2.add_expr(&"a".parse().unwrap())
        );
        assert_ne!(
            egraph3.add_expr(&"x".parse().unwrap()),
            egraph3.add_expr(&"a".parse().unwrap())
        );

        assert_eq!(
            egraph2.add_expr(&"y".parse().unwrap()),
            egraph2.add_expr(&"a".parse().unwrap())
        );
        assert_ne!(
            egraph3.add_expr(&"y".parse().unwrap()),
            egraph3.add_expr(&"a".parse().unwrap())
        );
    }

    #[test]
    fn test_intersect_basic() {
        let mut egraph1 = egg::EGraph::new(ConstantFold {}).with_explanations_enabled();
        let mut egraph2 = egg::EGraph::new(ConstantFold {}).with_explanations_enabled();
        egraph1.union_instantiations(
            &"(+ x 0)".parse().unwrap(),
            &"(+ y 0)".parse().unwrap(),
            &Subst::default(),
            "",
        );
        egraph2.union_instantiations(
            &"x".parse().unwrap(),
            &"y".parse().unwrap(),
            &Subst::default(),
            "",
        );
        egraph2.add_expr(&"(+ x 0)".parse().unwrap());
        egraph2.add_expr(&"(+ y 0)".parse().unwrap());

        let mut egraph3 = egraph1.egraph_intersect(&egraph2, ConstantFold {});

        assert_ne!(
            egraph3.add_expr(&"x".parse().unwrap()),
            egraph3.add_expr(&"y".parse().unwrap())
        );
        assert_eq!(
            egraph3.add_expr(&"(+ x 0)".parse().unwrap()),
            egraph3.add_expr(&"(+ y 0)".parse().unwrap())
        );
    }

    #[test]
    fn test_medium_intersect() {
        let mut egraph1 = egg::EGraph::<Math, ()>::new(());

        egraph1.add_expr(&"(sqrt (ln 1))".parse().unwrap());
        let ln = egraph1.add_expr(&"(ln 1)".parse().unwrap());
        let a = egraph1.add_expr(&"(sqrt (sin pi))".parse().unwrap());
        let b = egraph1.add_expr(&"(* 1 pi)".parse().unwrap());
        let pi = egraph1.add_expr(&"pi".parse().unwrap());
        egraph1.union(a, b);
        egraph1.union(a, pi);
        let c = egraph1.add_expr(&"(+ pi pi)".parse().unwrap());
        egraph1.union(ln, c);
        let k = egraph1.add_expr(&"k".parse().unwrap());
        let one = egraph1.add_expr(&"1".parse().unwrap());
        egraph1.union(k, one);
        egraph1.rebuild();

        assert_eq!(
            egraph1.add_expr(&"(ln k)".parse().unwrap()),
            egraph1.add_expr(&"(+ (* k pi) (* k pi))".parse().unwrap())
        );

        let mut egraph2 = egg::EGraph::<Math, ()>::new(());
        let ln2 = egraph2.add_expr(&"(ln 2)".parse().unwrap());
        let k2 = egraph2.add_expr(&"k".parse().unwrap());
        let mk1 = egraph2.add_expr(&"(* k 1)".parse().unwrap());
        egraph2.union(mk1, k2);
        let two = egraph2.add_expr(&"2".parse().unwrap());
        egraph2.union(mk1, two);
        let mul2pi = egraph2.add_expr(&"(+ (* 2 pi) (* 2 pi))".parse().unwrap());
        egraph2.union(ln2, mul2pi);
        egraph2.rebuild();

        assert_eq!(
            egraph2.add_expr(&"(ln k)".parse().unwrap()),
            egraph2.add_expr(&"(+ (* k pi) (* k pi))".parse().unwrap())
        );

        let mut egraph3 = egraph1.egraph_intersect(&egraph2, ());

        assert_eq!(
            egraph3.add_expr(&"(ln k)".parse().unwrap()),
            egraph3.add_expr(&"(+ (* k pi) (* k pi))".parse().unwrap())
        );
    }

    fn leaf(label: Math) -> TypedTree<Math> {
        TypedTree::leaf_untyped(label)
    }

    fn node(label: Math, children: Vec<TypedTree<Math>>) -> TypedTree<Math> {
        TypedTree::new_untyped(label, children)
    }

    fn sym(s: &str) -> TypedTree<Math> {
        leaf(Math::Symbol(s.into()))
    }

    /// Build tree, convert to `RecExpr`, check it matches the directly parsed `RecExpr`.
    fn assert_eq_recexpr(tree: &TypedTree<Math>, expected_str: &str) {
        let from_tree: RecExpr<Math> = (tree).to_rec_expr();
        let direct: RecExpr<Math> = expected_str.parse().unwrap();
        assert_eq!(from_tree, direct, "mismatch for {expected_str}");
    }

    #[test]
    fn leaf_symbol() {
        assert_eq_recexpr(&sym("x"), "x");
    }

    #[test]
    fn leaf_constant() {
        assert_eq_recexpr(&leaf(Math::Constant("42".parse().unwrap())), "42");
    }

    #[test]
    fn binary_add() {
        assert_eq_recexpr(
            &node(Math::Add([id0(), id0()]), vec![sym("x"), sym("y")]),
            "(+ x y)",
        );
    }

    #[test]
    fn unary_ln() {
        assert_eq_recexpr(&node(Math::Ln(id0()), vec![sym("x")]), "(ln x)");
    }

    #[test]
    fn unary_sqrt() {
        assert_eq_recexpr(&node(Math::Sqrt(id0()), vec![sym("x")]), "(sqrt x)");
    }

    #[test]
    fn nested() {
        // (+ (* x 1) y)
        let one = leaf(Math::Constant("1".parse().unwrap()));
        let mul = node(Math::Mul([id0(), id0()]), vec![sym("x"), one]);
        let add = node(Math::Add([id0(), id0()]), vec![mul, sym("y")]);
        assert_eq_recexpr(&add, "(+ (* x 1) y)");
    }

    #[test]
    fn deeply_nested() {
        // (d (sin (+ x y)) x)
        let sum = node(Math::Add([id0(), id0()]), vec![sym("x"), sym("y")]);
        let sin = node(Math::Sin(id0()), vec![sum]);
        let diff = node(Math::Diff([id0(), id0()]), vec![sin, sym("x")]);
        assert_eq_recexpr(&diff, "(d (sin (+ x y)) x)");
    }

    #[test]
    fn all_binary_ops() {
        let ops = [
            (Math::Add([id0(), id0()]), "+"),
            (Math::Sub([id0(), id0()]), "-"),
            (Math::Mul([id0(), id0()]), "*"),
            (Math::Div([id0(), id0()]), "/"),
            (Math::Pow([id0(), id0()]), "pow"),
            (Math::Diff([id0(), id0()]), "d"),
            (Math::Integral([id0(), id0()]), "i"),
        ];
        for (label, op_str) in ops {
            let tree = node(label, vec![sym("x"), sym("y")]);
            assert_eq_recexpr(&tree, &format!("({op_str} x y)"));
        }
    }

    #[test]
    fn all_unary_ops() {
        let ops = [
            (Math::Ln(id0()), "ln"),
            (Math::Sqrt(id0()), "sqrt"),
            (Math::Sin(id0()), "sin"),
            (Math::Cos(id0()), "cos"),
        ];
        for (label, op_str) in ops {
            let tree = node(label, vec![sym("x")]);
            assert_eq_recexpr(&tree, &format!("({op_str} x)"));
        }
    }
}
