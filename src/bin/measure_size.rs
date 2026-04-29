use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use csv::{Reader, Writer};
use egg::{RecExpr, Runner, SimpleScheduler};
use indicatif::{ProgressIterator, ProgressStyle};
use rise_distance::egg::math::{Math, RULES};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Parser)]
#[command(
    about = "Measure the heap size of running eqsat on each term in a CSV via dhat, \
             appending a measured_size column in place."
)]
struct Cli {
    /// Path to CSV file with a `term` column. Edited in place.
    #[arg(long)]
    path: PathBuf,

    /// Iter limit for the runner
    #[arg(long, default_value_t = 11)]
    max_iters: usize,

    /// Node limit for the runner
    #[arg(long, default_value_t = 100_000)]
    max_nodes: usize,

    /// Time limit for the runner (seconds)
    #[arg(long, default_value_t = 1.0)]
    max_time: f64,
}

fn measure_term(expr: &RecExpr<Math>, max_iters: usize, max_nodes: usize, max_time: f64) -> usize {
    let runner = Runner::default()
        .with_expr(expr)
        .with_iter_limit(max_iters)
        .with_node_limit(max_nodes)
        .with_time_limit(Duration::from_secs_f64(max_time))
        .with_scheduler(SimpleScheduler)
        .run(&*RULES);

    let before_drop = dhat::HeapStats::get();
    drop(runner);
    let after_drop = dhat::HeapStats::get();
    before_drop.curr_bytes - after_drop.curr_bytes
}

fn main() {
    let _profiler = dhat::Profiler::new_heap();

    let cli = Cli::parse();

    let mut rdr = Reader::from_path(&cli.path)
        .unwrap_or_else(|e| panic!("Failed to open CSV {}: {e}", cli.path.display()));
    let headers = rdr.headers().expect("CSV must have a header row").clone();
    let term_idx = headers
        .iter()
        .position(|h| h == "term")
        .expect("CSV must contain a `term` column");

    let rows = rdr
        .records()
        .map(|rec| rec.expect("CSV read error"))
        .collect::<Vec<_>>();

    let style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} terms ({eta})",
    )
    .expect("valid template")
    .progress_chars("=>-");

    let measurements: Vec<usize> = rows
        .iter()
        .progress_with_style(style)
        .map(|rec| {
            let term = &rec[term_idx];
            let expr = term
                .parse::<RecExpr<Math>>()
                .unwrap_or_else(|e| panic!("Failed to parse term '{term}': {e}"));
            measure_term(&expr, cli.max_iters, cli.max_nodes, cli.max_time)
        })
        .collect();

    let mut writer = Writer::from_path(&cli.path).expect("Failed to open CSV for writing");
    let mut new_headers: Vec<&str> = headers.iter().collect();
    new_headers.push("measured_size");
    writer.write_record(&new_headers).unwrap();
    for (rec, mem) in rows.iter().zip(measurements) {
        let mut new_rec: Vec<String> = rec.iter().map(str::to_owned).collect();
        new_rec.push(mem.to_string());
        writer.write_record(&new_rec).unwrap();
    }
    writer.flush().unwrap();
}
