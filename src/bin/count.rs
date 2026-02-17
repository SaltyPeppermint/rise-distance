use std::fs;
use std::path::Path;
use std::time::Instant;

use clap::{Args as ClapArgs, Parser, Subcommand};
use num::BigUint;
use num::ToPrimitive;

use rise_distance::{
    EClassId, EGraph, Expr, RiseLabel, Stats, TermCount, TreeNode, UnitCost, find_min_count_zs,
};

#[derive(Parser)]
#[command(about = "Count and sample terms per size in an e-graph")]
struct Args {
    /// Path to the serialized e-graph JSON file
    egraph: String,

    /// Maximum term size to count
    #[arg(short, long, default_value_t = 10)]
    limit: usize,

    /// Include type annotations in size calculations
    #[arg(short = 't', long)]
    with_types: bool,

    #[command(subcommand)]
    mode: Mode,
}

#[derive(Subcommand)]
enum Mode {
    /// Print term-count histograms
    #[command(after_help = "\
Examples:
  # Count terms up to size 20, print root histogram
  count graph.json -l 20 hist

  # Also print histograms for specific e-classes
  count graph.json -l 20 hist 0 3 7

  # Pretty-print with bar chart
  count graph.json -l 20 hist --pretty

  # Scientific notation for large numbers
  count graph.json -l 50 hist --scientific
")]
    Hist {
        /// Pretty-print histograms with bar charts
        #[arg(short, long)]
        pretty: bool,

        /// Display counts in scientific notation
        #[arg(short, long)]
        scientific: bool,

        /// Width of the bar chart (in characters)
        #[arg(short = 'w', long, default_value_t = 40, requires = "pretty")]
        bar_width: usize,

        /// Additional e-class IDs (numeric) to print histograms for
        eclass_ids: Vec<usize>,
    },

    /// Sample terms uniformly at random for each size in a range, find closest to a reference
    #[command(after_help = "\
Examples:
  # Sample 100 terms per size from sizes 5 to 20, compare to expression
  count graph.json -l 20 sample -m 5 -s 100 -e '(+ 1 2)'

  # Sample from the smallest possible size
  count graph.json -l 20 sample -s 100 -e '(+ 1 2)'

  # Compare to a named tree from a file
  count graph.json -l 20 sample -m 5 -s 100 -f trees.txt -n blocking_goal
")]
    Sample {
        /// Minimum term size to sample (defaults to smallest size with terms)
        #[arg(short = 'm', long)]
        min_size: Option<usize>,

        /// Number of terms to sample per size
        #[arg(short = 's', long, default_value_t = 10)]
        samples: u64,

        #[command(flatten)]
        reference: RefSource,
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
    let args = Args::parse();

    println!("Loading e-graph from: {}", args.egraph);
    let graph = EGraph::parse_from_file(Path::new(&args.egraph));

    let root = graph.root();
    println!("  Root e-class: {root:?}");
    println!("  Limit: {}", args.limit);
    println!("  With types: {}", args.with_types);

    let start = Instant::now();
    let term_count = TermCount::<BigUint, RiseLabel>::new(args.limit, args.with_types, &graph);
    println!("  Counting completed in {:.2?}", start.elapsed());

    match args.mode {
        Mode::Hist {
            pretty,
            scientific,
            bar_width,
            ref eclass_ids,
        } => {
            let fmt = DisplayConfig {
                pretty,
                scientific,
                bar_width,
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
        Mode::Sample {
            min_size,
            samples,
            ref reference,
        } => {
            let ref_tree = parse_ref(reference, args.with_types);
            run_extraction(
                &term_count,
                &ref_tree,
                min_size.unwrap_or_else(|| {
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
}

fn parse_ref(source: &RefSource, with_types: bool) -> TreeNode<RiseLabel> {
    if let Some(expr) = &source.expr {
        println!("Parsing reference tree from command line...");
        expr.parse::<Expr>()
            .expect("Failed to parse Rise expression")
            .to_tree(with_types)
    } else {
        let file = source.file.as_ref().unwrap();
        let name = source.name.as_ref().unwrap();
        println!("Parsing reference tree '{name}' from file...");
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
                            .to_tree(with_types),
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
    let ref_node_count = ref_tree.size();
    let ref_stripped_count = ref_tree.strip_types().size();
    println!("  Reference tree has {ref_node_count} nodes ({ref_stripped_count} without types)");

    let start = Instant::now();
    println!("\n--- Zhang-Shasha extraction (count-based sampling, with lower-bound pruning) ---");
    println!("  Sampling {samples_per_size} terms per size from {min_size} to {max_size}");

    if let (Some(result), stats) = find_min_count_zs(
        term_count,
        ref_tree,
        &UnitCost,
        min_size,
        max_size,
        samples_per_size,
    ) {
        print_stats(&result, &stats, start.elapsed());
    } else {
        println!("  No result found!");
    }
}

fn print_stats(result: &(TreeNode<RiseLabel>, usize), stats: &Stats, elapsed: std::time::Duration) {
    println!("  Best distance: {}", result.1);
    println!("  Time: {elapsed:.2?}");
    println!("\n  Statistics:");
    println!("    Trees enumerated:   {}", stats.trees_enumerated);
    #[expect(clippy::cast_precision_loss)]
    {
        println!(
            "    Trees size pruned:  {} ({:.1}%)",
            stats.size_pruned,
            100.0 * stats.size_pruned as f64 / stats.trees_enumerated as f64
        );
        println!(
            "    Trees euler pruned: {} ({:.1}%)",
            stats.euler_pruned,
            100.0 * stats.euler_pruned as f64 / stats.trees_enumerated as f64
        );
        println!(
            "    Full comparisons:   {} ({:.1}%)",
            stats.full_comparisons,
            100.0 * stats.full_comparisons as f64 / stats.trees_enumerated as f64
        );
    }
    println!("\n  Best tree:");
    println!("{}", result.0);
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
    println!("\n--- {label} ---");
    let Some(histogram) = data.of_eclass(id) else {
        println!("  (no data)");
        return;
    };

    let mut sizes: Vec<_> = histogram.iter().collect();
    sizes.sort_by_key(|(size, _)| *size);
    let total: BigUint = histogram.values().sum();

    if fmt.pretty {
        print_bar_chart(&sizes, &total, fmt.scientific, fmt.bar_width);
    } else {
        let count_width = sizes
            .iter()
            .map(|(_, c)| format_count(c, fmt.scientific).len())
            .max()
            .unwrap_or(0);
        for (size, count) in &sizes {
            println!(
                "  size {size:>4}: {:>width$}",
                format_count(count, fmt.scientific),
                width = count_width
            );
        }
        println!(
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
        println!(
            "  size {size:>4} | {bar:<width$} | {:>cw$}",
            format_count(count, scientific),
            width = bar_width,
            cw = count_width
        );
    }
    println!("\n  total: {total_str}");
}
