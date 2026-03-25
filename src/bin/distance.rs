use std::fs;
use std::path::Path;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser, Subcommand};
use indicatif::ParallelProgressIterator;
use num::{BigUint, ToPrimitive};
use rayon::prelude::*;

use rise_distance::cli::{DistanceMetric, SizeDistribution};
use rise_distance::count::TermCount;
use rise_distance::sampling::Sampler;
use rise_distance::sampling::count::CountSampler;
use rise_distance::{
    EClassId, Expr, Graph, Label, NumericId, RiseLabel, StructuralDistance, TreeNode, TreeShaped,
    UnitCost, ZSStats, find_min_struct, find_min_zs, prune_by_ref_tree, tree_distance_unit,
};

#[derive(Parser)]
#[command(about = "Find the closest tree in an e-graph to a reference tree")]
struct Cli {
    /// Path to the serialized e-graph JSON file
    egraph: String,

    #[command(flatten)]
    reference: RefSource,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Count-based sampling by exact term size
    #[command(after_help = "\
Examples:
  # Count terms up to size 20, print root histogram
  distance graph.json sample -l 20 --histogram

  # Pretty-print histogram with bar chart
  distance graph.json sample -l 20 --histogram --pretty

  # Sample 100 terms uniformly, compute Zhang-Shasha distance
  distance graph.json -e '(+ 1 2)' sample -l 20 -b 100

  # Proportional sampling with min 10 per size
  distance graph.json -e '(+ 1 2)' sample -l 20 -b 500 -p proportional:10

  # Normal distribution sampling with sigma=2.5
  distance graph.json -e '(+ 1 2)' sample -l 20 -b 500 -p normal:2.5

  # Use structural distance metric
  distance graph.json -e '(+ 1 2)' sample -l 20 -b 100 -d structural

  # Overlap sampling locks in shared structure with the reference tree
  distance graph.json -e '(+ 1 2)' sample -l 20 -b 100 --overlap

  # Compare to a named tree from a file
  distance graph.json -f trees.txt -n goal sample -l 20 -b 100
")]
    Sample {
        /// Include type annotations in size calculations
        #[arg(short = 't', long)]
        with_types: bool,

        /// Distance metric to use
        #[arg(short, long, default_value_t = DistanceMetric::ZhangShasha)]
        distance: DistanceMetric,

        /// Maximum term size to count
        #[arg(short, long)]
        max_size: Option<usize>,

        #[command(flatten)]
        display: DisplayConfig,

        /// Number of terms in total to take
        #[arg(short = 'b', long)]
        budget: Option<usize>,

        /// How to distribute the sample budget across sizes.
        /// Options: uniform, proportional:<`min_per_size`>, normal:<sigma>
        #[arg(short = 'p', long, requires = "budget", default_value_t = SizeDistribution::Uniform)]
        distribution: SizeDistribution,
    },

    /// Prune the e-graph by overlap with a reference tree and output the result
    #[command(after_help = "\
Examples:
  # Prune and print result to stdout
  distance graph.json -e '(+ 1 2)' prune

  # Prune and write to a file
  distance graph.json -e '(+ 1 2)' prune -o pruned.json

  # Prune using a named tree from a file
  distance graph.json -f trees.txt -n goal prune -o pruned.json
")]
    Prune {
        /// Write pruned graph to a file instead of printing to stdout.
        /// The output filename will encode the root e-class ID.
        #[arg(short, long)]
        output: Option<String>,
    },
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
    let cli = Cli::parse();

    std::eprintln!("Loading e-graph from: {}", cli.egraph);
    let graph = Graph::<RiseLabel>::parse_from_file(Path::new(&cli.egraph));
    let root = graph.root();
    std::eprintln!("Root e-class: {root:?}");

    match &cli.command {
        Command::Sample {
            with_types,
            distance,
            max_size,
            display,
            budget: samples,
            distribution,
        } => {
            let ref_tree = parse_ref(&cli.reference);
            let max_size = max_size.unwrap_or_else(|| ref_tree.size(*with_types));
            std::eprintln!("Limit: {max_size}");

            let start = Instant::now();
            let term_count = TermCount::<BigUint>::new(max_size, *with_types, &graph);
            std::eprintln!("Counting completed in {:.2?}", start.elapsed());

            if display.histogram {
                print_histogram("Root", root, &term_count, display);

                for &id in &display.additional_histogram_ids {
                    print_histogram(
                        &std::format!("EClassId({id})"),
                        EClassId::new(id),
                        &term_count,
                        display,
                    );
                }
            }

            if let Some(sample_count) = samples {
                let Some(min_size) = term_count
                    .get(&graph.root())
                    .and_then(|h| h.keys().min().copied())
                else {
                    std::panic!(
                        "Root e-class {root:?} has no terms up to size limit {max_size}. \
                         The smallest representable term likely exceeds this limit \
                         (try increasing -l)."
                    );
                };
                run_count_extraction(
                    &graph,
                    &term_count,
                    &ref_tree,
                    &CountSampleConfig {
                        min_size,
                        max_size,
                        total_samples: *sample_count,
                        distribution: *distribution,
                        with_types: *with_types,
                        distance: *distance,
                    },
                );
            }
        }
        Command::Prune { output } => {
            let ref_tree = parse_ref(&cli.reference);
            let start = Instant::now();
            let (pruned, pruned_ids) = prune_by_ref_tree(&graph, root, &ref_tree);
            std::eprintln!("Pruning completed in {:.2?}", start.elapsed());
            std::eprintln!("Pruned {} e-classes", pruned_ids.len());
            let pruned_str = std::format!(
                "[{}]",
                pruned_ids
                    .iter()
                    .map(|&i| i.to_index().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            if let Some(path) = output {
                let out_path = Path::new(path);
                let file = fs::File::create(out_path)
                    .unwrap_or_else(|e| std::panic!("Failed to create '{path}': {e}"));
                serde_json::to_writer(std::io::BufWriter::new(file), &pruned)
                    .expect("Failed to serialize pruned e-graph");
                std::println!("Root: {root:?}; Pruned: {pruned_str}; Path: {path}");
            } else {
                let json =
                    serde_json::to_string(&pruned).expect("Failed to serialize pruned e-graph");
                std::println!("Root: {root:?}; Pruned: {pruned_str}; EGraph: {json}");
            }
        }
    }
}

fn parse_ref(source: &RefSource) -> TreeNode<RiseLabel> {
    let parse_tree = |sexpr: &str| {
        sexpr
            .parse::<Expr>()
            .expect("Failed to parse Rise expression")
            .to_tree()
    };
    if let Some(expr) = &source.expr {
        eprintln!("Parsing reference tree from command line...");
        return parse_tree(expr);
    }
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
                Some(parse_tree(sexpr.trim()))
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("No tree with name {name} found"))
}

// --- Count-based extraction ---

#[derive(Debug, Clone)]
struct CountSampleConfig {
    min_size: usize,
    max_size: usize,
    total_samples: usize,
    distribution: SizeDistribution,
    with_types: bool,
    distance: DistanceMetric,
}

fn run_count_extraction<L: Label>(
    graph: &Graph<L>,
    term_count: &TermCount<BigUint>,
    ref_tree: &TreeNode<L>,
    config: &CountSampleConfig,
) {
    let CountSampleConfig {
        min_size,
        max_size,
        total_samples,
        distribution,
        with_types,
        distance,
    } = *config;
    let ref_node_count = ref_tree.size_with_types();
    let ref_stripped_count = ref_tree.size_without_types();
    eprintln!("Reference tree has {ref_node_count} nodes ({ref_stripped_count} without types)");

    let start = Instant::now();
    eprintln!("{distance} extraction (count-based sampling)");

    let histogram = term_count
        .get(&graph.root())
        .expect("Root e-class has no terms");
    #[expect(clippy::cast_precision_loss)]
    let normal_center = (ref_tree.size(with_types) - min_size) as f64;
    let samples_per_size =
        distribution.samples_per_size(histogram, min_size, max_size, total_samples, normal_center);
    let candidates = CountSampler::new(term_count, graph).sample_constrained_root(
        min_size,
        max_size,
        &samples_per_size,
    );
    let n_candidates = candidates.len();
    eprintln!("{n_candidates} unique candidates");

    let iter = candidates
        .into_par_iter()
        .map(|x| x.into())
        .progress_count(n_candidates.try_into().unwrap());

    match distance {
        DistanceMetric::ZhangShasha => {
            if let (Some(result), stats) = find_min_zs(iter, ref_tree, &UnitCost, with_types) {
                print_zs_result(&result, ref_tree, &stats, start.elapsed(), with_types);
            } else {
                eprintln!("No result found!");
            }
        }
        DistanceMetric::Structural => {
            if let Some(result) = find_min_struct(iter, ref_tree, &UnitCost, with_types) {
                print_struct_result(&result, ref_tree, start.elapsed(), with_types);
            } else {
                eprintln!("No result found!");
            }
        }
    }
}

// --- Shared result printing ---

#[expect(clippy::cast_precision_loss)]
fn print_zs_result<L: Label>(
    result: &(TreeNode<L>, usize),
    ref_tree: &TreeNode<L>,
    stats: &ZSStats,
    elapsed: std::time::Duration,
    with_types: bool,
) {
    let best = &result.0;
    eprintln!("Best distance: {}", result.1);
    eprintln!("Time: {elapsed:.2?}");
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

    let zs_dist = tree_distance_unit(&best.flatten(with_types), &ref_tree.flatten(with_types));
    eprintln!("ZS distance to ref: {zs_dist}");
    print_tree_sizes(best, ref_tree);
    println!("{best}");
}

fn print_struct_result<L: Label>(
    result: &(TreeNode<L>, StructuralDistance),
    ref_tree: &TreeNode<L>,
    elapsed: std::time::Duration,
    with_types: bool,
) {
    let best = &result.0;
    eprintln!("Best structural overlap: {}", result.1.overlap());
    eprintln!("Best structural zs_sum: {}", result.1.zs_sum());
    eprintln!("Time: {elapsed:.2?}");

    let zs_dist = tree_distance_unit(&best.flatten(with_types), &ref_tree.flatten(with_types));
    eprintln!("Raw ZS distance to ref: {zs_dist}");
    print_tree_sizes(best, ref_tree);
    println!("{best}");
}

fn print_tree_sizes<L: Label>(best: &TreeNode<L>, ref_tree: &TreeNode<L>) {
    eprintln!(
        "Best tree size: {} with types, {} without",
        best.size_with_types(),
        best.size_without_types()
    );
    eprintln!(
        "Ref tree size: {} with types, {} without",
        ref_tree.size_with_types(),
        ref_tree.size_without_types()
    );
}

// --- Histogram display (count subcommand only) ---

#[derive(ClapArgs)]
struct DisplayConfig {
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
    #[arg(long, default_value_t = 40, requires = "pretty")]
    bar_width: usize,

    /// Additional e-class IDs (numeric) to print histograms for (requires --histogram)
    #[arg(long, requires = "histogram")]
    additional_histogram_ids: Vec<usize>,
}

fn format_count(n: &BigUint, scientific: bool) -> String {
    if !scientific {
        return n.to_string();
    }
    let s = n.to_string();
    if s.len() <= 6 {
        return s;
    }
    let exponent = s.len() - 1;
    let mut mantissa = String::with_capacity(7);
    mantissa.push(s.as_bytes()[0] as char);
    mantissa.push('.');
    mantissa.push_str(&s[1..6.min(s.len())]);
    format!("{mantissa}e{exponent}")
}

fn print_histogram(label: &str, id: EClassId, data: &TermCount<BigUint>, fmt: &DisplayConfig) {
    eprintln!("--- {label} ---");
    let Some(histogram) = data.get(&id) else {
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
        #[expect(clippy::cast_precision_loss)]
        let bar_len = if fraction.is_finite() {
            (fraction * bar_width as f64).round().to_usize().unwrap()
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
