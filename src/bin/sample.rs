//! Produce the guide-candidate menu for one seed.
//!
//! This is the "first half" of the old `guide` binary: it replays the
//! guide-phase eqsat for one seed, builds a [`PrecomputePackage`], and samples
//! the four-strategy guide menu. The individual search legs (the second half)
//! are driven by `driver.py`, which feeds chosen guide subsets to `verify`.
//!
//! Touches no files and reads no stdin: everything comes on argv. `driver.py`
//! owns all file I/O — it computes the effective replay limits (search-phase
//! eqsat limits overridden by any `stop_*` keys and the recorded
//! `guide_egraph.iters`) and invokes this binary once per seed; the replay ends
//! at whichever limit trips first. The recorded goals and `max_size` stay
//! Python-side (they were only ever echoed back here), so `SeedSamples` is pure
//! sampling output. It is printed as a one-element JSON array to stdout (empty
//! on failure); logs go to stderr.

use std::collections::BTreeMap;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use num::BigUint;
use time::OffsetDateTime;

use rise_distance::cli::{GuideExpr, SeedSamples, Strategy};
use rise_distance::eqsat::{EqsatConfig, run_eqsat};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::sampling::{PrecomputePackage, TermSampleDist};
use rise_distance::{MyAnalysis, MyLanguage, OriginLang};

#[derive(Parser)]
#[command(
    about = "Sample the guide-candidate menu for one seed (feeds driver.py)",
    after_help = "\
Reads nothing on stdin: `--seed`, `--language`, and the replay's eqsat limits
come from argv. Prints a one-element `[SeedSamples]` array to stdout (empty on
failure); logs go to stderr. `driver.py` fans out one invocation per seed,
passing the effective replay limits (search-phase limits overridden by any
`stop_*` keys, `--max-iters` set to the recorded `guide_egraph.iters`); the
replay ends at whichever limit trips first.
Example:
  sample --language math --seed '(+ x 0)' \\
    --max-iters 38 --max-nodes 1000000 --max-time 10 \\
    --backoff-scheduler --max-memory 2000000000
"
)]
struct Args {
    /// Which language's rules to run under (from the folder's `args.json`).
    #[arg(long)]
    language: AvailableLanguages,

    /// The seed s-expression to replay the guide phase from.
    #[arg(long)]
    seed: String,

    /// The replay's eqsat limits (the driver passes the effective values, see
    /// the module doc).
    #[command(flatten)]
    eqsat: EqsatConfig,

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

    /// How many novel sizes the precompute must find (the size scan stops at
    /// the `sample_sizes`-th one).
    #[arg(long, default_value_t = 5)]
    sample_sizes: usize,
}

fn main() {
    let args = Args::parse();

    eprintln!("Starting at {}", OffsetDateTime::now_local().unwrap());
    eprintln!("Language: {:?}", args.language);
    eprintln!("Distribution: {}", args.size_distribution);
    eprintln!("Seed: {}", args.seed);

    match args.language {
        AvailableLanguages::Diospyros => {
            main_inner::<_, ()>(&args, &diospyros::rules(false, false));
        }
        AvailableLanguages::Math => {
            main_inner::<_, math::ConstantFold>(&args, &math::rules());
        }
        AvailableLanguages::Prop => {
            main_inner::<_, prop::ConstantFold>(&args, &prop::rules());
        }
    }

    eprintln!("Finished at {}", OffsetDateTime::now_local().unwrap());
}

/// Sample the seed named by `--seed` for one concrete language and print the
/// `SeedSamples` record as a one-element JSON array to stdout (empty array on
/// failure). The output is emitted here (rather than returned) because the
/// `SeedSamples<L>` type is language-specific and the caller's `match` arms
/// can't unify on it; the JSON bytes are type-erased on the wire.
fn main_inner<L: MyLanguage, N: MyAnalysis<L>>(args: &Args, rules: &[Rewrite<L, N>]) {
    eprintln!(
        "\n=== Seed: {} (max-iters={}) ===",
        args.seed, args.eqsat.max_iters
    );

    let mut out = Vec::new();
    match sample_seed(args, rules) {
        Ok(record) => out.push(record),
        Err(e) => eprintln!("ERROR OCCURRED:\n{e}"),
    }
    eprintln!("Finished seed at {}", OffsetDateTime::now_local().unwrap());

    serde_json::to_writer(std::io::stdout(), &out).expect("write samples JSON");
    println!();
}

fn sample_seed<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    rules: &[Rewrite<L, N>],
) -> Result<SeedSamples<L>, String> {
    let seed_expr = args
        .seed
        .parse::<RecExpr<L>>()
        .unwrap_or_else(|e| panic!("Failed to parse seed '{}': {e}", args.seed));

    // Replay the guide phase under the effective limits the driver computed;
    // the replay ends at whichever limit trips first.
    let result = run_eqsat(&seed_expr, rules.iter(), &args.eqsat).ok_or("Eqsat failed")?;
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
        seed: args.seed.clone(),
        candidates,
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
