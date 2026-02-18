use std::fs;
use std::path::Path;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser};
use num::BigUint;
use num::ToPrimitive;

use rise_distance::{
    EClassId, EGraph, Expr, RiseLabel, Stats, TermCount, TreeNode, UnitCost, find_min_count_zs,
    tree_distance_unit,
};

#[derive(Parser)]
#[command(about = "Count and sample terms per size in an e-graph")]
#[expect(clippy::struct_excessive_bools)]
#[command(after_help = "\
Examples:
  # Count terms up to size 20, print root histogram
  count graph.json -l 20 --hist

  # Also print histograms for specific e-classes
  count graph.json -l 20 --hist 0 3 7

  # Pretty-print with bar chart, then sample
  count graph.json -l 20 --hist --pretty -s 100 -e '(+ 1 2)'

  # Sample only, from sizes 5 to 20
  count graph.json -l 20 -m 5 -s 100 -e '(+ 1 2)'

  # Compare to a named tree from a file
  count graph.json -l 20 -m 5 -s 100 -f trees.txt -n blocking_goal
")]
struct Args {
    /// Path to the serialized e-graph JSON file
    egraph: String,

    /// Maximum term size to count
    #[arg(short, long, default_value_t = 10)]
    limit: usize,

    /// Include type annotations in size calculations
    #[arg(short = 't', long)]
    with_types: bool,

    /// Print term-count histograms
    #[arg(short, long)]
    histogram: bool,

    /// Pretty-print histograms with bar charts (requires --histogram)
    #[arg(long, requires = "histogram")]
    pretty: bool,

    /// Display counts in scientific notation (requires --histogram)
    #[arg(long, requires = "histogram")]
    scientific: bool,

    /// Width of the bar chart in characters (requires --pretty)
    #[arg(short = 'w', long, default_value_t = 40, requires = "pretty")]
    bar_width: usize,

    /// Additional e-class IDs (numeric) to print histograms for (requires --histogram)
    eclass_ids: Vec<usize>,

    /// Minimum term size to sample (defaults to smallest size with terms)
    #[arg(short = 'm', long)]
    min_size: Option<usize>,

    /// Number of terms to sample per size
    #[arg(short = 's', long)]
    samples: Option<u64>,

    #[command(flatten)]
    reference: RefSource,
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

    eprintln!("Loading e-graph from: {}", args.egraph);
    let graph = EGraph::parse_from_file(Path::new(&args.egraph));

    let root = graph.root();
    eprintln!("Root e-class: {root:?}");
    eprintln!("Limit: {}", args.limit);
    eprintln!("With types: {}", args.with_types);

    let start = Instant::now();
    let term_count = TermCount::<BigUint, RiseLabel>::new(args.limit, args.with_types, &graph);
    eprintln!("Counting completed in {:.2?}", start.elapsed());

    if args.histogram {
        let fmt = DisplayConfig {
            pretty: args.pretty,
            scientific: args.scientific,
            bar_width: args.bar_width,
        };

        print_histogram("Root", root, &term_count, &fmt);

        for &id in &args.eclass_ids {
            print_histogram(
                &format!("EClassId({id})"),
                EClassId::new(id),
                &term_count,
                &fmt,
            );
        }
    }

    if let Some(samples) = args.samples {
        let ref_tree = parse_ref(&args.reference);
        run_extraction(
            &term_count,
            &ref_tree,
            args.min_size.unwrap_or_else(|| {
                term_count
                    .of_eclass(root)
                    .and_then(|h| h.keys().copied().min())
                    .expect("Root e-class has no terms")
            }),
            args.limit,
            samples,
        );
    }
}

fn parse_ref(source: &RefSource) -> TreeNode<RiseLabel> {
    if let Some(expr) = &source.expr {
        eprintln!("Parsing reference tree from command line...");
        expr.parse::<Expr>()
            .expect("Failed to parse Rise expression")
            .to_tree()
    } else {
        let file = source.file.as_ref().unwrap();
        let name = source.name.as_ref().unwrap();
        eprintln!("Parsing reference tree '{name}' from file...");
        let content =
            fs::read_to_string(file).unwrap_or_else(|e| panic!("Failed to read '{file}': {e}"));
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .find_map(|line| {
                let (n, sexpr) = line.split_once(':').expect("Line must be 'Name: sexpr'");
                if n.trim() == name {
                    Some(
                        sexpr
                            .trim()
                            .parse::<Expr>()
                            .expect("Failed to parse Rise expression")
                            .to_tree(),
                    )
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("No tree with name {name} found"))
    }
}

fn run_extraction(
    term_count: &TermCount<BigUint, RiseLabel>,
    ref_tree: &TreeNode<RiseLabel>,
    min_size: usize,
    max_size: usize,
    samples_per_size: u64,
) {
    let ref_node_count = ref_tree.size_with_types();
    let ref_stripped_count = ref_tree.size();
    eprintln!("Reference tree has {ref_node_count} nodes ({ref_stripped_count} without types)");

    let start = Instant::now();
    eprintln!("Zhang-Shasha extraction (count-based sampling, with lower-bound pruning)");
    eprintln!("Sampling {samples_per_size} terms per size from {min_size} to {max_size}");

    if let (Some(result), stats) = find_min_count_zs(
        term_count,
        ref_tree,
        &UnitCost,
        min_size,
        max_size,
        samples_per_size,
    ) {
        print_stats(&result, ref_tree, &stats, start.elapsed());
    } else {
        eprintln!("No result found!");
    }
}

fn print_stats(
    result: &(TreeNode<RiseLabel>, usize),
    ref_tree: &TreeNode<RiseLabel>,
    stats: &Stats,
    elapsed: std::time::Duration,
) {
    eprintln!("Best distance: {}", result.1);
    eprintln!("Time: {elapsed:.2?}");
    eprintln!("Statistics:");
    eprintln!("  Trees enumerated:   {}", stats.trees_enumerated);
    #[expect(clippy::cast_precision_loss)]
    {
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
    }

    let best = &result.0;
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
}

struct DisplayConfig {
    pretty: bool,
    scientific: bool,
    bar_width: usize,
}

fn format_count(n: &BigUint, scientific: bool) -> String {
    if !scientific {
        return n.to_string();
    }
    let s = n.to_string();
    if s.len() <= 6 {
        return s;
    }
    // e.g. "123456789" -> "1.23456e8"
    let exponent = s.len() - 1;
    let mut mantissa = String::with_capacity(7);
    mantissa.push(s.as_bytes()[0] as char);
    mantissa.push('.');
    mantissa.push_str(&s[1..6.min(s.len())]);
    format!("{mantissa}e{exponent}")
}

fn print_histogram(
    label: &str,
    id: EClassId,
    data: &TermCount<BigUint, RiseLabel>,
    fmt: &DisplayConfig,
) {
    eprintln!("--- {label} ---");
    let Some(histogram) = data.of_eclass(id) else {
        eprintln!("  (no data)");
        return;
    };

    let mut sizes = histogram.iter().collect::<Vec<_>>();
    sizes.sort_by_key(|(size, _)| *size);
    let total = histogram.values().sum::<BigUint>();

    if fmt.pretty {
        print_bar_chart(&sizes, &total, fmt.scientific, fmt.bar_width);
    } else {
        let count_width = sizes
            .iter()
            .map(|(_, c)| format_count(c, fmt.scientific).len())
            .max()
            .unwrap_or(0);
        for (size, count) in &sizes {
            eprintln!(
                "  size {size:>4}: {:>width$}",
                format_count(count, fmt.scientific),
                width = count_width
            );
        }
        eprintln!(
            "  total:     {:>width$}",
            format_count(&total, fmt.scientific),
            width = count_width
        );
    }
}

fn print_bar_chart(
    sizes: &[(&usize, &BigUint)],
    total: &BigUint,
    scientific: bool,
    bar_width: usize,
) {
    let max_count = sizes
        .iter()
        .map(|(_, c)| *c)
        .max()
        .cloned()
        .unwrap_or_default();

    let total_str = format_count(total, scientific);
    let count_width = sizes
        .iter()
        .map(|(_, c)| format_count(c, scientific).len())
        .chain(std::iter::once(total_str.len()))
        .max()
        .unwrap_or(0);

    let max_f64 = max_count.to_f64().unwrap_or(f64::INFINITY);

    for (size, count) in sizes {
        let fraction = count.to_f64().unwrap_or(0.0) / max_f64;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let bar_len = if fraction.is_finite() {
            (fraction * bar_width as f64).round() as usize
        } else {
            0
        };
        let bar = "#".repeat(bar_len);
        eprintln!(
            "  size {size:>4} | {bar:<width$} | {:>cw$}",
            format_count(count, scientific),
            width = bar_width,
            cw = count_width
        );
    }
    eprintln!("  total: {total_str}");
}
