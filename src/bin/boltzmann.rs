use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser};

use rise_distance::{
    EGraph, Expr, Label, TreeNode, UnitCost, find_min_boltzmann_zs, tree_distance_unit,
};

#[derive(Parser)]
#[command(about = "Find the closest tree in an e-graph to a reference tree")]
#[command(after_help = "\
Examples:
  # Reference tree from file
  boltzmann graph.json -f trees.txt -n blocking_goal

  # Reference tree from command line
  boltzmann graph.json -e '(+ 1 2)'

  # With revisits and quiet mode
  boltzmann graph.json -e '(foo bar)' -r 2 -q
")]
struct Args {
    /// Path to the serialized e-graph JSON file
    graph: String,

    #[command(flatten)]
    reference: RefSource,

    /// Target weight
    #[arg(short, long, default_value_t = 0)]
    target_weight: usize,

    /// Number of samples
    #[arg(short = 's', long, default_value_t = 10000)]
    samples: usize,

    /// Include the types in the comparison
    #[arg(short, long)]
    with_types: bool,

    /// Use raw string labels instead of Rise-typed labels (for regression testing)
    #[arg(long)]
    raw_strings: bool,
    // /// Use structural distance instead of Zhang-Shasha tree edit distance
    // #[arg(short, long)]
    // structural: bool,

    // /// Ignore the labels when using the structural option
    // #[arg(short, long, requires_all = ["structural"])]
    // ignore_labels: bool,
}

#[derive(ClapArgs)]
struct RefSource {
    /// Reference tree as an s-expression
    #[arg(short = 'e', long = "expr", conflicts_with_all = ["file", "name"])]
    expr: Option<String>,

    /// Path to file containing named trees
    #[arg(short = 'f', long, requires = "name")]
    file: Option<String>,

    /// Name of the reference tree (requires --file)
    #[arg(short = 'n', long, requires = "file")]
    name: Option<String>,
}

fn main() {
    let args = Args::parse();

    if args.raw_strings {
        run(&args, |sexpr| {
            TreeNode::<String>::from_str(sexpr).expect("Failed to parse s-expression")
        });
    } else {
        run(&args, |sexpr| {
            sexpr
                .parse::<Expr>()
                .expect("Failed to parse Rise expression")
                .to_tree()
        });
    }
}

fn run<L, F>(args: &Args, parse_tree: F)
where
    L: Label,
    F: Fn(&str) -> TreeNode<L>,
{
    eprintln!("Loading e-graph from: {}", args.graph);
    let graph = EGraph::<L>::parse_from_file(Path::new(&args.graph));
    eprintln!("Root e-class: {:?}", graph.root());
    let ref_tree = parse_ref(args, parse_tree);
    run_extraction(&graph, &ref_tree, args);
}

fn parse_ref<L, F>(args: &Args, parse_tree: F) -> TreeNode<L>
where
    L: Label,
    F: Fn(&str) -> TreeNode<L>,
{
    if let Some(expr) = &args.reference.expr {
        eprintln!("Parsing reference tree from command line...");
        return parse_tree(expr);
    }
    let file = args.reference.file.as_ref().unwrap();
    let name = args.reference.name.as_ref().unwrap();
    eprintln!("Parsing reference tree '{name}' from file...");
    let content =
        fs::read_to_string(file).unwrap_or_else(|e| panic!("Failed to read '{file}': {e}"));
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .find_map(|line| {
            let (n, sexpr) = line.split_once(':').expect("Line must be 'Name: sexpr'");
            if n.trim() == name {
                Some(parse_tree(sexpr.trim()))
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("No tree with name {name} found"))
}

#[expect(clippy::cast_precision_loss)]
fn run_extraction<L: Label>(graph: &EGraph<L>, ref_tree: &TreeNode<L>, args: &Args) {
    let ref_node_count = ref_tree.size_with_types();
    let ref_stripped_count = ref_tree.size();
    eprintln!("Reference tree has {ref_node_count} nodes ({ref_stripped_count} without types)");

    let start = Instant::now();
    eprintln!("Zhang-Shasha extraction (with lower-bound pruning)");
    if let (Some(result), stats) = find_min_boltzmann_zs(
        graph,
        ref_tree,
        &UnitCost,
        args.with_types,
        args.samples,
        args.target_weight,
        100,
    ) {
        let best = &result.0;
        eprintln!("Best distance: {}", result.1);
        eprintln!("Time: {:.2?}", start.elapsed());
        eprintln!("Statistics:");
        eprintln!("  Trees enumerated:   {}", stats.trees_enumerated);
        eprintln!(
            "  Trees size pruned:  {} ({:.1}%)",
            stats.size_pruned,
            100.0 * stats.size_pruned as f64 / stats.trees_enumerated as f64
        );
        eprintln!(
            "  Trees euler pruned: {} ({:.1}%)",
            stats.euler_pruned,
            100.0 * stats.euler_pruned as f64 / stats.trees_enumerated as f64
        );
        eprintln!(
            "  Full comparisons:   {} ({:.1}%)",
            stats.full_comparisons,
            100.0 * stats.full_comparisons as f64 / stats.trees_enumerated as f64
        );

        let zs_dist = tree_distance_unit(&best.flatten(true), &ref_tree.flatten(true));
        eprintln!("ZS distance to ref: {zs_dist}");
        eprintln!(
            "Best tree size: {} with types, {} without",
            best.size_with_types(),
            best.size()
        );
        eprintln!(
            "Ref tree size: {} with types, {} without",
            ref_tree.size_with_types(),
            ref_tree.size()
        );

        println!("{best}");
    } else {
        eprintln!("No result found!");
    }
}
