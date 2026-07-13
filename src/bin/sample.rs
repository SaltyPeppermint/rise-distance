//! Produce the guide-candidate menu for one seed/goal pair.
//!
//! This is the "first half" of the old `guide` binary: it replays the
//! guide-phase eqsat for one seed, builds a [`PrecomputePackage`], and samples
//! the four-strategy guide menu. The individual search legs (the second half)
//! are driven by `driver.py`, which feeds chosen guide subsets to `verify`.
//!
//! A stdin/stdout filter that touches no files: `driver.py` owns all file I/O.
//! It reads `args.json` (for `--language` and the eqsat flags, both on argv) and
//! the goal-enriched `terms.json`, then passes this binary a single seed's
//! per-seed replay inputs (`seed`, `guide_iters`, `max_size`, `goals`) as a JSON
//! request on stdin. The `SeedSamples` record is printed as a one-element JSON
//! array to stdout (empty array on failure); all logging goes to stderr.

use std::collections::BTreeMap;
use std::io::Read;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use num::BigUint;
use serde::Deserialize;
use time::OffsetDateTime;

use rise_distance::cli::{EqsatArgs, GuideExpr, SeedSamples, Strategy};
use rise_distance::eqsat::{EqsatConfig, run_eqsat};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::sampling::{PrecomputePackage, TermSampleDist};
use rise_distance::{MyAnalysis, MyLanguage, OriginLang};

#[derive(Parser)]
#[command(
    about = "Sample the guide-candidate menu for one seed/goal pair (feeds driver.py)",
    after_help = "\
Reads a JSON `SampleRequest` on stdin (per-seed replay inputs) and prints a
one-element `[SeedSamples]` array to stdout (empty on failure); logs go to
stderr. `--language` and the eqsat limits come from argv. `driver.py` parses
terms.json and fans these out.
Example:
  echo '{\"seed\":\"(+ x 0)\",\"guide_iters\":38,\"max_size\":24,\"goals\":[...]}' \\
    | sample --language math --max-iters 200 --max-nodes 1000000 --max-time 10 \\
      --backoff-scheduler
"
)]
struct Args {
    /// Which language's rules to run under (from the folder's `args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    #[command(flatten)]
    eqsat: EqsatArgs,

    /// How to distribute the guide sample budget across sizes.
    #[arg(long, default_value_t = TermSampleDist::GREEDY)]
    size_distribution: TermSampleDist,

    /// How many guide candidates to draw per sampling strategy. The menu must be
    /// large enough for the driver to pick its widest `k` and to reshuffle
    /// across restarts. `Smallest` always contributes exactly one term.
    #[arg(long, default_value_t = 1000)]
    samples_per_strategy: usize,

    /// How much to grow `max_size` on each precompute retry.
    #[arg(long, default_value_t = 5)]
    retry_step: usize,

    /// How many times to retry precompute with a larger `max_size` before
    /// giving up on a seed.
    #[arg(long, default_value_t = 20)]
    max_retries: usize,

    /// How many sizes need to be present in the precomputed histogram of root
    #[arg(long, default_value_t = 5)]
    sample_sizes: usize,
}

/// One seed's replay inputs, read from stdin as JSON. These are exactly the
/// fields `sample` used to pull out of the goal-enriched `terms.json`; the
/// driver now parses that file and passes them here.
#[derive(Deserialize)]
struct SampleRequest {
    /// The seed s-expression to replay the guide phase from.
    seed: String,
    /// Guide-phase iteration count recorded by `goal`; the replay runs to
    /// exactly this many iters.
    guide_iters: usize,
    /// `max_size` recorded by `goal`, copied straight onto the output record.
    max_size: usize,
    /// Lowered goal s-expressions recorded by `goal`, passed through to output.
    goals: Vec<String>,
}

fn main() {
    let args = Args::parse();

    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .expect("read sample request from stdin");
    let request: SampleRequest = serde_json::from_str(&buf).expect("parse sample request JSON");

    eprintln!("Starting at {}", OffsetDateTime::now_local().unwrap());
    eprintln!("Language: {:?}", args.language);
    eprintln!("Distribution: {}", args.size_distribution);
    eprintln!("Seed: {}", request.seed);

    match args.language {
        AvailableLanguages::Diospyros => {
            main_inner::<_, ()>(&args, &request, &diospyros::rules(false, false));
        }
        AvailableLanguages::Math => {
            main_inner::<_, math::ConstantFold>(&args, &request, &math::rules());
        }
        AvailableLanguages::Prop => {
            main_inner::<_, prop::ConstantFold>(&args, &request, &prop::rules());
        }
    }

    eprintln!("Finished at {}", OffsetDateTime::now_local().unwrap());
}

/// Sample the requested seed for one concrete language and print the
/// `SeedSamples` record as a one-element JSON array to stdout (empty array on
/// failure). The output is emitted here (rather than returned) because the
/// `SeedSamples<L>` type is language-specific and the caller's `match` arms
/// can't unify on it; the JSON bytes are type-erased on the wire.
fn main_inner<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    request: &SampleRequest,
    rules: &[Rewrite<L, N>],
) {
    eprintln!(
        "\n=== Seed: {} (max_size={}, guide_iters={}) ===",
        request.seed, request.max_size, request.guide_iters
    );

    let mut out = Vec::new();
    match sample_seed(args, request, rules) {
        Ok(record) => out.push(record),
        Err(e) => eprintln!("ERROR OCCURRED:\n{e}"),
    }
    eprintln!("Finished seed at {}", OffsetDateTime::now_local().unwrap());

    serde_json::to_writer(std::io::stdout(), &out).expect("write samples JSON");
    println!();
}

fn sample_seed<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    request: &SampleRequest,
    rules: &[Rewrite<L, N>],
) -> Result<SeedSamples<L>, String> {
    let seed_expr = request
        .seed
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed '{}': {e}", request.seed));

    // Replay the guide phase with the recorded iteration count, under the same
    // eqsat config `goal` ran so the replay matches the stored `guide_iters`.
    let replay_config = EqsatConfig {
        max_iters: request.guide_iters,
        ..args.eqsat.into()
    };

    let result = run_eqsat(&seed_expr, rules.iter(), &replay_config).ok_or("Eqsat failed")?;
    eprintln!("Guide replay stop reason: {:?}", result.stop_reason());

    let guide_nodes = result.curr().total_number_of_nodes();
    let guide_classes = result.curr().classes().len();
    let guide_iters = result.data().len();
    let guide_time = result.data().iter().map(|i| i.total_time).sum();
    eprintln!("Guide egraph (replay): {guide_nodes} nodes, {guide_classes} classes");

    let mut root_log = String::new();

    let start_size = AstSize.cost_rec(&seed_expr);
    let (max_size, pc) = PrecomputePackage::<BigUint, _, _>::backoff_precompute(
        &result,
        start_size,
        args.max_retries,
        args.retry_step,
        args.sample_sizes,
        &mut root_log,
    )
    .map_err(|tried_max_size| {
        format!(
            "goal precompute returned None after {} retries (max_size={})",
            args.max_retries, tried_max_size
        )
    })?;
    eprintln!("PC computation succeeded with max_size {max_size}!");
    pc.log_root(&mut root_log);
    eprint!("{root_log}");

    let mut candidates = BTreeMap::new();
    for strategy in Strategy::ALL {
        let terms = draw_candidates(args, strategy, &pc);
        candidates.insert(
            strategy.name().to_owned(),
            terms.iter().map(GuideExpr::from_recexpr).collect(),
        );
    }

    Ok(SeedSamples {
        seed: request.seed.clone(),
        goals: request.goals.clone(),
        candidates,
        max_size: request.max_size,
        guide_nodes,
        guide_classes,
        guide_iters,
        guide_time,
        stop_reason: format!("{:?}", result.stop_reason()),
    })
}

/// Draw the guide-candidate pool for one strategy. The four sampling strategies
/// draw `samples_per_strategy` terms in a single batch; `Smallest` contributes
/// the one smallest (novel or overall) root term.
fn draw_candidates<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    strategy: Strategy,
    pc: &PrecomputePackage<BigUint, L, N>,
) -> Vec<RecExpr<OriginLang<L>>> {
    match strategy {
        // Replacement is a driver concern (how it re-draws subsets from the
        // pool across restarts); the pool itself is one novel sampled batch
        // either way. Sampling strategies always draw novel terms.
        Strategy::Sample(s) => pc
            .sample_frontier_terms(
                args.samples_per_strategy,
                args.size_distribution,
                s,
                [args.samples_per_strategy as u64, strategy.seed_of()],
                true,
            )
            .unwrap_or_else(|| {
                eprintln!(
                    "WARNING: strategy {} drew 0 candidates (empty novel frontier); \
                     driver legs for this strategy will have no guides to pick from",
                    strategy.name()
                );
                Vec::new()
            }),
        Strategy::Smallest { novel } => vec![pc.smallest(pc.root(), novel)],
    }
}
