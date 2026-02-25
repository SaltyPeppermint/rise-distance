use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser, Subcommand};
use hashbrown::HashMap;
use indicatif::ParallelProgressIterator;
use num::BigUint;
use num::ToPrimitive;
use rayon::prelude::*;

use rise_distance::StructuralDistance;
use rise_distance::{
    EClassId, EGraph, Expr, Label, RiseLabel, TermCount, TreeNode, UnitCost, ZSStats,
    find_min_struct, find_min_zs, tree_distance_unit,
};

#[derive(Parser)]
#[command(about = "Find the closest tree in an e-graph to a reference tree")]
struct Cli {
    /// Use raw string labels instead of Rise-typed labels (for regression testing)
    #[arg(long)]
    raw_strings: bool,

    /// Distance metric to use
    #[arg(short, long, default_value_t = DistanceMetric::ZhangShasha)]
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
  distance graph.json count -l 20 --histogram --pretty -b 100 -e '(+ 1 2)'

  # Sample only, from sizes 5 to 20
  distance graph.json -e '(+ 1 2)' count -l 20 -m 5 -b 100

  # Proportional sampling with min 10 per size
  distance graph.json -e '(+ 1 2)' count -l 20 -b 500 -p proportional:10

  # Normal distribution sampling with sigma=2.5
  distance graph.json -e '(+ 1 2)' count -l 20 -b 500 -p normal:2.5

  # Compare to a named tree from a file
  distance graph.json -f trees.txt -n blocking_goal count -l 20 -m 5 -s 100
")]
    Sample {
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
        #[arg(short = 'p', long, requires = "budget", default_value_t = SampleDistribution::Uniform)]
        distribution: SampleDistribution,

        /// Use overlap sampling: lock in shared structure with the reference tree
        /// before sampling holes
        #[arg(short = 'o', long, requires = "budget")]
        overlap: bool,
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

/// Available distance metrics for tree comparison.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum DistanceMetric {
    /// Zhang-Shasha tree edit distance
    ZhangShasha,
    /// Structural tree edit distance
    Structural,
}

impl std::fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZhangShasha => write!(f, "ZhangShasha"),
            Self::Structural => write!(f, "Structural"),
        }
    }
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
        Command::Sample {
            max_size,
            display,
            budget: samples,
            distribution,
            overlap,
        } => {
            let ref_tree = parse_ref(&cli.reference, &parse_tree);
            let max_size = max_size.unwrap_or_else(|| ref_tree.size(cli.with_types));
            eprintln!("Limit: {max_size}");

            let start = Instant::now();
            let term_count = TermCount::<BigUint, L>::new(max_size, cli.with_types, &graph);
            eprintln!("Counting completed in {:.2?}", start.elapsed());

            if display.histogram {
                print_histogram("Root", root, &term_count, display);

                for &id in &display.additional_histogram_ids {
                    print_histogram(
                        &format!("EClassId({id})"),
                        EClassId::new(id),
                        &term_count,
                        display,
                    );
                }
            }

            if let Some(sample_count) = samples {
                let Some(min_size) = term_count
                    .of_eclass(root)
                    .and_then(|h| h.keys().min().copied())
                else {
                    panic!(
                        "Root e-class {root:?} has no terms up to size limit {max_size}. \
                         The smallest representable term likely exceeds this limit \
                         (try increasing -l)."
                    );
                };
                run_count_extraction(
                    &term_count,
                    &ref_tree,
                    &CountSampleConfig {
                        min_size,
                        max_size,
                        total_samples: *sample_count,
                        distribution: *distribution,
                        with_types: cli.with_types,
                        distance: cli.distance,
                        overlap: *overlap,
                    },
                );
            }
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

#[derive(Debug, Clone)]
struct CountSampleConfig {
    min_size: usize,
    max_size: usize,
    total_samples: usize,
    distribution: SampleDistribution,
    with_types: bool,
    distance: DistanceMetric,
    overlap: bool,
}

#[derive(Debug, Clone, Copy)]
enum SampleDistribution {
    /// Uniform accross the term sizes
    Uniform,
    /// Proportional to the number of terms of that size with a minimum number per size
    Proportional(usize),
    /// As a normal distribution centered in the middle between the smallest and goal term
    /// value = sigma
    Normal(f64),
}

impl FromStr for SampleDistribution {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "uniform" {
            return Ok(Self::Uniform);
        }
        if s == "proportional" {
            return Ok(Self::Proportional(10));
        }

        if let Some(rest) = s.strip_prefix("proportional:") {
            let min = rest
                .parse::<usize>()
                .map_err(|e| format!("invalid min_per_size in 'proportional:{rest}': {e}"))?;
            return Ok(Self::Proportional(min));
        }

        if s == "normal" {
            return Ok(Self::Normal(2.6));
        }
        if let Some(rest) = s.strip_prefix("normal:") {
            let sigma = rest
                .parse::<f64>()
                .map_err(|e| format!("invalid sigma in 'normal:{rest}': {e}"))?;
            return Ok(Self::Normal(sigma));
        }
        Err(format!(
            "unknown distribution '{s}': expected 'uniform', 'proportional:<min>', or 'normal:<sigma>'"
        ))
    }
}

impl std::fmt::Display for SampleDistribution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform => write!(f, "uniform"),
            Self::Proportional(min) => write!(f, "proportional:{min}"),
            Self::Normal(sigma) => write!(f, "normal:{sigma}"),
        }
    }
}

fn run_count_extraction<L: Label>(
    term_count: &TermCount<BigUint, L>,
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
        overlap,
    } = *config;
    let ref_node_count = ref_tree.size_with_types();
    let ref_stripped_count = ref_tree.size_without_types();
    eprintln!("Reference tree has {ref_node_count} nodes ({ref_stripped_count} without types)");

    let start = Instant::now();
    eprintln!("{distance} extraction (count-based sampling)");

    if overlap {
        eprintln!("Using overlap sampling");
    }

    let samples_per_size = samples_per_size(
        term_count,
        ref_tree,
        min_size,
        max_size,
        total_samples,
        distribution,
        with_types,
    );
    let candidates = if overlap {
        term_count.sample_unique_root_overlap(ref_tree, min_size, max_size, &samples_per_size)
    } else {
        term_count.sample_unique_root(min_size, max_size, &samples_per_size)
    };
    let n_candidates = candidates.len();
    eprintln!("{n_candidates} unique candidates");

    let iter = candidates
        .into_par_iter()
        .progress_count(n_candidates as u64);

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

fn samples_per_size<L: Label>(
    term_count: &TermCount<BigUint, L>,
    ref_tree: &TreeNode<L>,
    min_size: usize,
    max_size: usize,
    total_samples: usize,
    distribution: SampleDistribution,
    with_types: bool,
) -> HashMap<usize, u64> {
    match distribution {
        SampleDistribution::Proportional(min_per_size) => {
            let histogram = term_count.of_root().expect("Root e-class has no terms");
            let total_terms = (min_size..=max_size)
                .filter_map(|s| histogram.get(&s))
                .sum::<BigUint>();
            eprintln!(
                "Sampling {total_samples} terms proportionally from {min_size} to {max_size}"
            );
            let budget = BigUint::from(total_samples);
            (min_size..max_size)
                .map(|size| {
                    let n = histogram.get(&size).map_or(0, |count| {
                        (count * &budget / &total_terms)
                            .to_u64()
                            .unwrap_or(u64::MAX)
                            .max(min_per_size as u64)
                    });
                    eprintln!("Sampling {n} terms for size {size}");
                    (size, n)
                })
                .collect()
        }
        SampleDistribution::Normal(sigma) => {
            #[expect(clippy::cast_precision_loss)]
            let center = (ref_tree.size(with_types) - min_size) as f64;
            eprintln!(
                "Sampling {total_samples} terms with normal distribution (center={center}, sigma={sigma}) from {min_size} to {max_size}"
            );
            // Compute unnormalized weights for each size
            let weights = (min_size..=max_size)
                .map(|s| {
                    #[expect(clippy::cast_precision_loss)]
                    let z = (s as f64 - center) / sigma;
                    (s, (-0.5 * z * z).exp())
                })
                .collect::<HashMap<_, _>>();
            let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
            (min_size..max_size)
                .map(|size| {
                    let w = *weights.get(&size).unwrap_or(&0.0);
                    #[expect(
                        clippy::cast_precision_loss,
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss
                    )]
                    let n = (w / total_weight * total_samples as f64).round() as u64;
                    eprintln!("Sampling {n} terms for size {size}");
                    (size, n)
                })
                .collect()
        }
        SampleDistribution::Uniform => {
            eprintln!("Sampling {total_samples} terms per size from {min_size} to {max_size}");
            let s = (total_samples / (max_size - min_size)) as u64;
            (min_size..max_size).map(|size| (size, s)).collect()
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
