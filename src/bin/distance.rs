use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser, Subcommand};
use indicatif::ParallelProgressIterator;
use num::BigUint;
use num::ToPrimitive;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rayon::prelude::*;

use rise_distance::{
    DistanceMetric, EClassId, EGraph, Expr, FixpointSampler, FixpointSamplerConfig, Label,
    RiseLabel, Stats, TermCount, TreeNode, UnitCost, find_min_zs_par, tree_distance_unit,
};

#[derive(Parser)]
#[command(about = "Find the closest tree in an e-graph to a reference tree")]
struct Cli {
    /// Use raw string labels instead of Rise-typed labels (for regression testing)
    #[arg(long)]
    raw_strings: bool,

    /// Distance metric to use
    #[arg(short, long, default_value_t = DistanceMetric::Zs)]
    distance: DistanceMetric,

    /// Path to the serialized e-graph JSON file
    egraph: String,

    /// Include type annotations in size calculations
    #[arg(short = 't', long)]
    with_types: bool,

    #[command(flatten)]
    reference: RefSource,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Count-based uniform sampling by exact term size
    #[command(after_help = "\
Examples:
  # Count terms up to size 20, print root histogram
  distance graph.json count -l 20 --histogram

  # Pretty-print with bar chart, then sample
  distance graph.json count -l 20 --histogram --pretty -s 100 -e '(+ 1 2)'

  # Sample only, from sizes 5 to 20
  distance graph.json -e '(+ 1 2)' count -l 20 -m 5 -s 100

  # Compare to a named tree from a file
  distance graph.json -f trees.txt -n blocking_goal count -l 20 -m 5 -s 100
")]
    Count {
        /// Maximum term size to count
        #[arg(short, long, default_value_t = 10)]
        limit: usize,

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
    },

    /// Boltzmann sampling-based search
    #[command(after_help = "\
Examples:
  # Reference tree from file
  distance graph.json -f trees.txt -n blocking_goal boltzmann

  # Reference tree from command line
  distance graph.json -e '(+ 1 2)' boltzmann -s 1000
")]
    Boltzmann {
        /// Target weight
        #[arg(short = 'w', long, default_value_t = 0)]
        target_weight: usize,

        /// Number of samples
        #[arg(short = 's', long, default_value_t = 10000)]
        samples: usize,
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

    if cli.raw_strings {
        run::<String>(&cli, |sexpr| {
            TreeNode::<String>::from_str(sexpr).expect("Failed to parse s-expression")
        });
    } else {
        run::<RiseLabel>(&cli, |sexpr| {
            sexpr
                .parse::<Expr>()
                .expect("Failed to parse Rise expression")
                .to_tree()
        });
    }
}

fn run<L: Label>(cli: &Cli, parse_tree: impl Fn(&str) -> TreeNode<L>) {
    eprintln!("Loading e-graph from: {}", cli.egraph);
    let graph = EGraph::<L>::parse_from_file(Path::new(&cli.egraph));
    let root = graph.root();
    eprintln!("Root e-class: {root:?}");
    eprintln!("With types: {}", cli.with_types);

    match &cli.command {
        Command::Count {
            limit,
            histogram,
            pretty,
            scientific,
            bar_width,
            eclass_ids,
            min_size,
            samples,
        } => {
            eprintln!("Limit: {limit}");

            let start = Instant::now();
            let term_count = TermCount::<BigUint, L>::new(*limit, cli.with_types, &graph);
            eprintln!("Counting completed in {:.2?}", start.elapsed());

            if *histogram {
                let fmt = DisplayConfig {
                    pretty: *pretty,
                    scientific: *scientific,
                    bar_width: *bar_width,
                };

                print_histogram("Root", root, &term_count, &fmt);

                for &id in eclass_ids {
                    print_histogram(
                        &format!("EClassId({id})"),
                        EClassId::new(id),
                        &term_count,
                        &fmt,
                    );
                }
            }

            if let Some(sample_count) = samples {
                let ref_tree = parse_ref(&cli.reference, &parse_tree);
                run_count_extraction(
                    &term_count,
                    &ref_tree,
                    min_size.unwrap_or_else(|| {
                        term_count
                            .of_eclass(root)
                            .and_then(|h| h.keys().copied().min())
                            .expect("Root e-class has no terms")
                    }),
                    *limit,
                    *sample_count,
                );
            }
        }
        Command::Boltzmann {
            target_weight,
            samples,
        } => {
            let ref_tree = parse_ref(&cli.reference, &parse_tree);
            run_boltzmann_extraction(&graph, &ref_tree, cli.with_types, *samples, *target_weight);
        }
    }
}

fn parse_ref<L: Label>(
    source: &RefSource,
    parse_tree: impl Fn(&str) -> TreeNode<L>,
) -> TreeNode<L> {
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

fn run_count_extraction<L: Label>(
    term_count: &TermCount<BigUint, L>,
    ref_tree: &TreeNode<L>,
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

    let candidates = term_count.sample_unique(min_size, max_size, samples_per_size);
    let n_candidates = candidates.len();
    eprintln!("{n_candidates} unique candidates");

    let iter = candidates
        .into_par_iter()
        .progress_count(n_candidates as u64);

    if let (Some(result), stats) =
        find_min_zs_par(iter, ref_tree, &UnitCost, term_count.with_types())
    {
        print_result(&result, ref_tree, &stats, start.elapsed());
    } else {
        eprintln!("No result found!");
    }
}

// --- Boltzmann-based extraction ---

fn run_boltzmann_extraction<L: Label>(
    graph: &EGraph<L>,
    ref_tree: &TreeNode<L>,
    with_types: bool,
    samples: usize,
    target_weight: usize,
) {
    let ref_node_count = ref_tree.size_with_types();
    let ref_stripped_count = ref_tree.size();
    eprintln!("Reference tree has {ref_node_count} nodes ({ref_stripped_count} without types)");

    let start = Instant::now();
    eprintln!("Zhang-Shasha extraction (Boltzmann sampling, with lower-bound pruning)");

    let mut rng = ChaCha12Rng::seed_from_u64(100);
    let config = FixpointSamplerConfig::builder().build();
    let (fp_sampler, lambda, expected_size) =
        FixpointSampler::for_target_size(graph, target_weight, &config, &mut rng).unwrap();
    eprintln!("LAMBDA IS {lambda}");
    eprintln!("EXPECTED SIZE IS {expected_size}");

    let samples_vec = fp_sampler.take(samples).collect::<Vec<_>>();
    let iter = samples_vec.into_par_iter().progress_count(samples as u64);

    if let (Some(result), stats) = find_min_zs_par(iter, ref_tree, &UnitCost, with_types) {
        print_result(&result, ref_tree, &stats, start.elapsed());
    } else {
        eprintln!("No result found!");
    }
}

// --- Shared result printing ---

#[expect(clippy::cast_precision_loss)]
fn print_result<L: Label>(
    result: &(TreeNode<L>, usize),
    ref_tree: &TreeNode<L>,
    stats: &Stats,
    elapsed: std::time::Duration,
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

// --- Histogram display (count subcommand only) ---

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
    let exponent = s.len() - 1;
    let mut mantissa = String::with_capacity(7);
    mantissa.push(s.as_bytes()[0] as char);
    mantissa.push('.');
    mantissa.push_str(&s[1..6.min(s.len())]);
    format!("{mantissa}e{exponent}")
}

fn print_histogram<L: Label>(
    label: &str,
    id: EClassId,
    data: &TermCount<BigUint, L>,
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
