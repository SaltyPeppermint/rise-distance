use std::path::Path;
use std::time::Instant;

use clap::Parser;
use num::BigUint;
use num::ToPrimitive;

use rise_distance::{EClassId, EGraph, TermCount};

#[derive(Parser)]
#[command(about = "Count terms per size in an e-graph")]
#[command(after_help = "\
Examples:
  # Count terms up to size 20, print root histogram
  count graph.json -l 20

  # Also print histograms for specific e-classes
  count graph.json -l 20 0 3 7

  # Include type annotations in size calculation
  count graph.json -l 10 --with-types

  # Pretty-print with bar chart
  count graph.json -l 20 --pretty

  # Scientific notation for large numbers
  count graph.json -l 50 --scientific
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
}

fn main() {
    let args = Args::parse();

    println!("Loading e-graph from: {}", args.egraph);
    let graph = EGraph::<String>::parse_from_file(Path::new(&args.egraph));

    let root = graph.root();
    println!("  Root e-class: {root:?}");
    println!("  Limit: {}", args.limit);
    println!("  With types: {}", args.with_types);

    let counter = TermCount::new(args.limit, args.with_types);

    let start = Instant::now();
    let data = counter.analyze::<_, BigUint>(&graph);
    println!("  Analysis completed in {:.2?}", start.elapsed());

    let fmt = DisplayConfig {
        pretty: args.pretty,
        scientific: args.scientific,
        bar_width: args.bar_width,
    };

    print_histogram("Root", root, &data, &fmt);

    for &id in &args.eclass_ids {
        print_histogram(&format!("EClassId({id})"), EClassId::new(id), &data, &fmt);
    }
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
    data: &hashbrown::HashMap<EClassId, hashbrown::HashMap<usize, BigUint>>,
    fmt: &DisplayConfig,
) {
    println!("\n--- {label} ---");
    let Some(histogram) = data.get(&id) else {
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
