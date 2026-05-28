use std::fmt::Write;

use egg::{AstSize, Language, RecExpr, rewrite};

use crate::sketch::{SketchLang, eclass_contains, eclass_extract};
use crate::utils::grow_egraph_until;

type Lang = egg::SymbolLang;
type EGraph = egg::EGraph<Lang, ()>;
type Rewrite = egg::Rewrite<Lang, ()>;
type Expr = egg::RecExpr<Lang>;
type Sketch = RecExpr<SketchLang<Lang>>;

// f o g = (o f g)
// semantic: \x. f (g x)
// map n f = (m n f)
// semantic: [x1, .., xn]
//        => [f x1, .., f xn]
// transpose = T
// semantic: [[x11, .., x1n], .., [xm1, .., xmn]]
//        => [[x11, .., xm1], .., [x1n, .., xmn]]
// split n = (s n)
// semantic: [x1, .., xn, .., xm]
//        => [[x1, .., xn], .., [.., xm]]
// join = j
// semantic: [[x11, .., x1n], .., [xm1, .., xmn]]
//        => [x11, .., x1n, .., xm1, .., xmn]

// T o T = id
// j o (s n) = id

#[must_use]
pub fn common_rules() -> Vec<Rewrite> {
    vec![
        rewrite!("o-assoc1"; "(o (o ?f ?g) ?h)" => "(o ?f (o ?g ?h))"),
        rewrite!("o-assoc2"; "(o ?f (o ?g ?h))" => "(o (o ?f ?g) ?h)"),
        // rewrite!("map-fusion"; "(o (m ?n ?f) (m ?n ?g))" => "(m ?n (o ?f ?g))"),
        rewrite!("map-fission"; "(m ?n (o ?f ?g))" =>  "(o (m ?n ?f) (m ?n ?g))"),
        // unused rules:
        // rewrite!("transpose-before-maps"; "(o T (m ?n1 (m ?n2 ?f)))" => "(o (m ?n1 (m ?n2 ?f)) T)"),
        // rewrite!("transpose-after-maps"; "(o (m ?n1 (m ?n2 ?f)) T)" => "(o T (m ?n1 (m ?n2 ?f)))"),
    ]
}

#[must_use]
pub fn transpose_maps() -> Vec<Rewrite> {
    vec![
        rewrite!("transpose-maps"; "(m ?n1 (m ?n2 ?f))" => "(o T (o (m ?n2 (m ?n1 ?f)) T))"),
        // shortcut rules:
        // rewrite!("transpose-maps-2"; "(m ?n1 (m ?n2 (m ?n3 ?f)))" => "(o (m ?n1 T) (o (m ?n1 (m ?n3 (m ?n2 ?f))) (m ?n1 T))"),
        // rewrite!("transpose-maps-3"; "(m ?n1 (m ?n2 (m ?n3 (m ?n4 ?f))))" => "(o (m ?n1 (m ?n2 T)) (o (m ?n1 (m ?n2 (m ?n4 (m ?n3 ?f)))) (m ?n1 (m ?n2 T))"),
    ]
}

#[must_use]
pub fn split_map() -> Vec<Rewrite> {
    vec![
        rewrite!("split-map"; "(m (* ?n1 ?n2) ?f)" => "(o j (o (m ?n1 (m ?n2 ?f)) (s ?n2)))"), // rewrite!("split-map-32"; "(m ?n ?f)" => "(o j (o (m (/ ?n 32) (m 32 ?f)) (s 32)))"),
                                                                                               // n = (n1 * n2) / n2 = n1
                                                                                               // rewrite!("mul-div-id"; "(/ (* ?n1 ?n2) ?n2)" => "?n1"),
    ]
}

#[derive(Copy, Clone, Debug, clap::ValueEnum)]
pub enum TilingSearch {
    Split,
    Reorder,
    Tile,
}

#[must_use]
pub fn tile_1d(ts: TilingSearch) -> Vec<Expr> {
    tile(
        ts,
        "1d",
        // 1 nested map that we want to tile (split + reoder):
        "(m (* n1 32) f)",
        // sketches for the splitted map nests we are looking for:
        &["(contains (m n1 (m 32 f)))"],
        &["(o j (o (m n1 (m 32 f)) (s 32)))"],
        // there's nothing to reorder in 1d:
        &["(contains (m n1 (m 32 f)))"],
        &["(o j (o (m n1 (m 32 f)) (s 32)))"],
    )
}

#[must_use]
pub fn tile_2d(ts: TilingSearch) -> Vec<Expr> {
    tile(
        ts,
        "2d",
        // 2 nested maps that we want to tile (split + reorder):
        "(m (* n1 32) (m (* n2 32) f))",
        // sketches for the splitted map nests we are looking for:
        &["(contains (m n1 (m 32 (m n2 (m 32 f)))))"],
        // the corresponding full programs that we expect to find:
        &[
            "(o (o (o (m (* n1 32) j) j) (o (m n1 (m 32 (m n2 (m 32 f)))) (m n1 (m 32 (s 32))))) (s 32))",
        ],
        // sketches for the tiled map nests we are looking for:
        &["(contains (m n1 (m n2 (m 32 (m 32 f)))))"],
        &[
            "(o (o (o (o (m (* n1 32) j) j) (m n1 T)) (o (m n1 (m n2 (m 32 (m 32 f)))) (m n1 T))) (o (m n1 (m 32 (s 32))) (s 32)))",
        ],
    )
}

#[must_use]
pub fn tile_3d(ts: TilingSearch) -> Vec<Expr> {
    tile(
        ts,
        "3d",
        // 3 nested maps that we want to tile (split + reorder):
        "(m (* n1 32) (m (* n2 32) (m (* n3 32) f)))",
        // sketches for the splitted map nests we are looking for:
        &[
            /* "(contains (m n1 (m 32 (m (* n2 32) (m (* n3 32) f)))))",
            "(contains (m (* n1 32) (m n2 (m 32 (m (* n3 32) f)))))",
            "(contains (m (* n1 32) (m (* n2 32) (m n3 (m 32 f)))))",
            "(contains (m n1 (m 32 (contains (m n2 (m 32 (m (* n3 32) f))))))))",
            "(contains (m (* n1 32) (contains (m n2 (m 32 (contains (m n3 (m 32 f)))))))))", */
            "(contains (m n1 (m 32 (m n2 (m 32 (m n3 (m 32 f))))))))",
        ],
        // the corresponding full programs that we expect to find:
        &[
            /* "(o (o j (m n1 (m 32 (m (* n2 32) (m (* n3 32) f))))) (s 32))",
            "(o (m (* n1 32) j) (o (m (* n1 32) (m n2 (m 32 (m (* n3 32) f)))) (m (* n1 32) (s 32))))",
            "(o (m (* n1 32) (m (* n2 32) j)) (o (m (* n1 32) (m (* n2 32) (m n3 (m 32 f)))) (m (* n1 32) (m (* n2 32) (s 32)))))",
            "(o (o j (m n1 (m 32 (o j (o (m n2 (m 32 (m (* n3 32) f))) (s 32)))))) (s 32))",
            "(m (* n1 32) (o j (o (m n2 (m 32 (o j (o (m n3 (m 32 f)) (s 32))))) (s 32))))", */
            // "(o (o j (m n1 (m 32 (o j (o (m n2 (m 32 (o j (o (m n3 (m 32 f)) (s 32))))) (s 32)))))) (s 32))",
            "(o (m (* n1 32) (o (m (* n2 32) j) j)) (o (o j (o (m n1 (m 32 (m n2 (m 32 (m n3 (m 32 f)))))) (s 32))) (m (* n1 32) (o (m n2 (m 32 (s 32))) (s 32)))))",
        ],
        // sketches for the tiled map nests we are looking for:
        &[
            /* "(contains (m n1 (m 32 (m (* n2 32) (m (* n3 32) f)))))",
            "(contains (m (* n1 32) (m n2 (m 32 (m (* n3 32) f)))))",
            "(contains (m (* n1 32) (m (* n2 32) (m n3 (m 32 f)))))",
            "(contains (m n1 (contains (m n2 (m 32 (m 32 (m (* n3 32) f))))))))",
            "(contains (m (* n1 32) (contains (m n2 (contains (m n3 (m 32 (m 32 f)))))))))", */
            "(contains (m n1 (m n2 (m n3 (m 32 (m 32 (m 32 f))))))))",
        ],
        // the corresponding full programs that we expect to find:
        &[
            /* "(o j (o (m n1 (m 32 (m (* n2 32) (m (* n3 32) f)))) (s 32)))",
            "(o (m (* n1 32) j) (o (m (* n1 32) (m n2 (m 32 (m (* n3 32) f)))) (m (* n1 32) (s 32))))",
            "(o (o (m (* n1 32) (m (* n2 32) j)) (m (* n1 32) (m (* n2 32) (m n3 (m 32 f))))) (m (* n1 32) (m (* n2 32) (s 32))))",
            "(o j (o (m n1 (o (m 32 j) (o (o T (m n2 (m 32 (m 32 (m (* n3 32) f))))) (o T (m 32 (s 32)))))) (s 32)))",
            "(m (* n1 32) (o j (o (m n2 (o (o (o (m 32 j) T) (m n3 (m 32 (m 32 f)))) (o T (m 32 (s 32))))) (s 32))))", */
            // "(o j (o (m n1 (o (o (o (m 32 j) T) (o (m n2 (o (m 32 (o (m 32 j) T)) (o (o T (o (m n3 (m 32 (m 32 (m 32 f)))) T)) (m 32 (o T (m 32 (s 32))))))) T)) (m 32 (s 32)))) (s 32)))",
            "(o (o (m (* n1 32) (o (m (* n2 32) j) j)) j) (o (o (m n1 (o T (m n2 (o (m 32 T) T)))) (o (m n1 (m n2 (m n3 (m 32 (m 32 (m 32 f)))))) (m n1 (m n2 T)))) (o (o (m n1 (o (m n2 (m 32 T)) T)) (s 32)) (m (* n1 32) (o (m n2 (m 32 (s 32))) (s 32))))))",
        ],
    )
}

#[must_use]
pub fn tile_4d(ts: TilingSearch) -> Vec<Expr> {
    tile(
        ts,
        "4d",
        // 4 nested maps that we want to tile (split + reorder):
        "(m (* n1 32) (m (* n2 32) (m (* n3 32) (m (* n4 32) f))))",
        // sketches for the splitted map nests we are looking for:
        &["(contains (m n1 (m 32 (m n2 (m 32 (m n3 (m 32 (m n4 (m 32 f)))))))))"],
        // the corresponding full programs that we expect to find:
        &[
            "(o (m (* n1 32) (o (m (* n2 32) (o (m (* n3 32) j) j)) j)) (o (o j (o (m n1 (m 32 (m n2 (m 32 (m n3 (m 32 (m n4 (m 32 f)))))))) (s 32))) (m (* n1 32) (o (m n2 (m 32 (o (m n3 (m 32 (s 32))) (s 32)))) (s 32)))))",
        ],
        // sketches for the tiled map nests we are looking for:
        &["(contains (m n1 (m n2 (m n3 (m n4 (m 32 (m 32 (m 32 (m 32 f))))))))))"],
        // the corresponding full programs that we expect to find:
        &[
            "f", // ???
        ],
    )
}

#[must_use]
#[expect(clippy::missing_panics_doc)]
pub fn tile(
    ts: TilingSearch,
    name: &str,
    start: &str,
    split_sketches: &[&str],
    split_expected: &[&str],
    reorder_sketches: &[&str],
    reorder_expected: &[&str],
) -> Vec<Expr> {
    let parse_expr = |s: &&str| {
        let e: Expr = s.parse().unwrap();
        println!("LaTeX Expr: {}", latex_of_expr(&e));
        e
    };
    let parse_sketch = |s: &&str| {
        let e: Sketch = s.parse().unwrap();
        println!("LaTeX Sketch: {}", latex_of_expr(&e));
        e
    };

    let s = &[parse_expr(&start)];

    match ts {
        TilingSearch::Split => {
            let mut split_rules = common_rules();
            split_rules.extend(split_map());
            split_rules.extend(transpose_maps()); // <<< unused
            let ss = split_sketches.iter().map(parse_sketch).collect::<Vec<_>>();
            let se = split_expected.iter().map(parse_expr).collect::<Vec<_>>();
            reach_sketches_from_exprs(&format!("tile_{name}_s"), s, &split_rules, &ss, &se)
        }
        TilingSearch::Reorder => {
            let mut reorder_rules = common_rules();
            reorder_rules.extend(split_map()); // <<< unused
            reorder_rules.extend(transpose_maps());
            let rs = reorder_sketches
                .iter()
                .map(parse_sketch)
                .collect::<Vec<_>>();
            let se = split_expected.iter().map(parse_expr).collect::<Vec<_>>();
            let re = reorder_expected.iter().map(parse_expr).collect::<Vec<_>>();
            reach_sketches_from_exprs(&format!("tile_{name}_r"), &se, &reorder_rules, &rs, &re)
        }
        TilingSearch::Tile => {
            let mut tile_rules = common_rules();
            tile_rules.extend(split_map());
            tile_rules.extend(transpose_maps());
            let rs: Vec<Sketch> = reorder_sketches.iter().map(parse_sketch).collect();
            reach_sketches_from_exprs(
                &format!("tile_{name}"),
                s,
                &tile_rules,
                &rs,
                &[],
                // may find different programs: &re[..],
            )
        }
    }
}

#[must_use]
#[expect(clippy::missing_panics_doc)]
pub fn reorder_3d() -> Vec<Expr> {
    let mut rules = common_rules();
    rules.extend(transpose_maps());

    reach_sketches_from_exprs(
        "reorder_3d",
        // 3 nested maps that we want to reorder:
        &["(m n1 (m n2 (m n3 f)))".parse().unwrap()],
        &rules,
        // sketches for the reordered map nests we are looking for:
        &[
            "(contains (m n1 (m n3 (m n2 f))))".parse().unwrap(),
            "(contains (m n2 (m n1 (m n3 f))))".parse().unwrap(),
            "(contains (m n2 (m n3 (m n1 f))))".parse().unwrap(),
            "(contains (m n3 (m n2 (m n1 f))))".parse().unwrap(),
            "(contains (m n3 (m n1 (m n2 f))))".parse().unwrap(),
        ],
        // the corresponding full programs that we expect to find:
        &[
            "(o (m n1 T) (o (m n1 (m n3 (m n2 f))) (m n1 T)))"
                .parse()
                .unwrap(),
            "(o T (o (m n2 (m n1 (m n3 f))) T))".parse().unwrap(),
            "(o (o T (m n2 T)) (o (m n2 (m n3 (m n1 f))) (o (m n2 T) T)))"
                .parse()
                .unwrap(),
            "(o (o (o (o T (m n2 T)) T) (o (m n3 (m n2 (m n1 f))) (o T (m n2 T)))) T)"
                .parse()
                .unwrap(),
            "(o (m n1 T) (o (o T (o (m n3 (m n1 (m n2 f))) T)) (m n1 T)))"
                .parse()
                .unwrap(),
        ],
    )
}

#[must_use]
#[expect(clippy::missing_panics_doc)]
pub fn reach_sketches_from_exprs(
    search_name: &str,
    starts: &[Expr],
    rules: &[Rewrite],
    sketch_goals: &[Sketch],
    expected_goals: &[Expr], // may be empty to avoid checks
) -> Vec<Expr> {
    let mut egraph = EGraph::default();
    let eclass = starts
        .iter()
        .map(|e| egraph.add_expr(e))
        .collect::<Vec<_>>()
        .into_iter()
        .reduce(|a, b| {
            egraph.union(a, b);
            a
        })
        .expect("need at least one starting expression");
    let sketches_hook = sketch_goals.to_owned();
    let egraph = grow_egraph_until(search_name, egraph, rules, move |r| {
        let cano_eclass = r.egraph.find(eclass);
        sketches_hook.iter().all(|s| {
            // eclass_extract_sketch(s, egg::AstSize, &r.egraph, cano_eclass).is_some()
            eclass_contains(s, &r.egraph, cano_eclass)
        })
    });

    let cano_eclass = egraph.find(eclass);
    // FIXME: will return empty vec if expected goals is empty
    sketch_goals
        .iter()
        .zip(expected_goals.iter())
        .map(|(sketch, _expected)| {
            eclass_extract(sketch, AstSize, &egraph, cano_eclass)
                .unwrap()
                .1
        })
        .collect()
}

#[must_use]
pub fn string_of_expr(e: &Expr, flatten_o: bool) -> String {
    let mut res = String::new();
    string_of_expr_rec(e.as_ref(), e.as_ref().len() - 1, flatten_o, &mut res);
    res
}

pub fn string_of_expr_rec(nodes: &[Lang], i: usize, flatten_o: bool, acc: &mut String) {
    let node = &nodes[i];
    let op = node.to_string();

    if flatten_o && op == "o" {
        let cs = node.children();
        string_of_expr_rec(nodes, usize::from(cs[0]), flatten_o, acc);
        write!(acc, " o ").unwrap();
        string_of_expr_rec(nodes, usize::from(cs[1]), flatten_o, acc);
        return;
    }

    if node.is_leaf() {
        write!(acc, "{op}").unwrap();
        return;
    }

    write!(acc, "({op}").unwrap();
    for child in node.children().iter().map(|inner_i| usize::from(*inner_i)) {
        write!(acc, " ").unwrap();
        string_of_expr_rec(nodes, child, flatten_o, acc);
    }
    write!(acc, ")").unwrap();
}

#[must_use]
pub fn latex_of_expr<L: Language + std::fmt::Display>(e: &egg::RecExpr<L>) -> String {
    let mut res = String::new();
    latex_of_expr_rec(e.as_ref(), e.as_ref().len() - 1, &mut res);
    res
}

pub fn latex_of_expr_rec<L: Language + std::fmt::Display>(nodes: &[L], i: usize, acc: &mut String) {
    let node = &nodes[i];
    let op = node.to_string();

    let (is_infix, parenthesis, expanded_op) = match op.as_str() {
        "contains" => (false, true, "contains"),
        "o" => (true, false, "\\circ"),
        "*" => (true, true, "\\times"),
        "m" => (false, true, "map"),
        "s" => (false, true, "split"),
        "j" => (false, false, "join"),
        "T" => (false, false, "transpose"),
        op => (false, !node.is_leaf(), op),
    };

    if parenthesis {
        write!(acc, "(").unwrap();
    }
    let cs = node.children();
    if is_infix {
        latex_of_expr_rec(nodes, usize::from(cs[0]), acc);
        write!(acc, " {expanded_op} ").unwrap();
        latex_of_expr_rec(nodes, usize::from(cs[1]), acc);
    } else {
        write!(acc, "{expanded_op}").unwrap();
        for child in cs.iter().map(|i_inner| usize::from(*i_inner)) {
            write!(acc, "~").unwrap();
            latex_of_expr_rec(nodes, child, acc);
        }
    }
    if parenthesis {
        write!(acc, ")").unwrap();
    }
}
