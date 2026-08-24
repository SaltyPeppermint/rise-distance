//! Produce the guide-candidate menu for one seed.
//!
//! Replays guide-phase eqsat and samples the requested candidate pools.
//! Arguments come from `guided_search.py`; output is a one-element JSON array
//! on stdout, or an empty array on failure. Logs go to stderr.

use std::collections::BTreeMap;

use clap::Parser;
use egg::{AstSize, CostFunction, RecExpr, Rewrite};
use num::BigUint;
use time::OffsetDateTime;

use rise_distance::cli::{CandidatePool, GuideExpr, SeedSamples};
use rise_distance::eqsat::{EqsatConfig, run_eqsat};
use rise_distance::langs::{AvailableLanguages, diospyros, math, prop};
use rise_distance::sampling::{Distribution, PrecomputePackage};
use rise_distance::{MyAnalysis, MyLanguage, OriginLang};

#[derive(Parser)]
#[command(
    about = "Sample the guide-candidate menu for one seed (feeds guided_search.py)",
    after_help = "\
Reads nothing on stdin: `--seed`, `--language`, and the replay's eqsat limits
come from argv. Prints a one-element `[SeedSamples]` array to stdout (empty on
failure); logs go to stderr. `guided_search.py` fans out one invocation per seed,
passing the effective replay limits (search-phase limits overridden by its
`--stop-*` budget flags); the replay ends at whichever limit trips first.
Example:
  sample --language math --seed '(+ x 0)' \\
    --max-iters 38 --max-nodes 1000000 --max-time 10 \\
    --max-memory 2000000000 \\
    --candidate-pool sample_balanced
"
)]
struct Args {
    /// Language used for eqsat.
    #[arg(long)]
    language: AvailableLanguages,

    /// Seed s-expression.
    #[arg(long)]
    seed: String,

    /// Guide replay limits.
    #[command(flatten)]
    eqsat: EqsatConfig,

    /// How to distribute the guide sample budget across sizes.
    #[arg(long, default_value_t = Distribution::Greedy)]
    size_distribution: Distribution,

    /// Candidates per sampling strategy. `Smallest` contributes one.
    #[arg(long, default_value_t = 1000)]
    samples_per_strategy: usize,

    /// Candidate-sampling seed, independent of the batch size.
    #[arg(long, default_value_t = 0)]
    sampling_seed: u64,

    /// How much to grow `max_size` on each precompute retry.
    #[arg(long, default_value_t = 5)]
    retry_step: usize,

    /// Number of precompute size increments.
    #[arg(long, default_value_t = 20)]
    max_retries: usize,

    /// Number of novel sizes to find.
    #[arg(long, default_value_t = 5)]
    sample_sizes: usize,

    /// Candidate pool to emit. Repeat for a shared multi-pool manifest.
    #[arg(long = "candidate-pool", value_enum, required = true)]
    candidate_pools: Vec<CandidatePool>,
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

/// Sample one language-specific seed and print its JSON record.
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
    let stop_reason = format!("{:?}", result.stop_reason());
    eprintln!("Guide replay stop reason: {stop_reason}");

    // Absolute process live heap, sampled before precompute and sampling below
    // allocate further.
    let guide_memory = result.allocated();
    let guide_peak_live_heap = result.peak_allocated();
    let guide_nodes = result.curr().total_number_of_nodes();
    let guide_classes = result.curr().classes().len();
    let guide_iters = result.data().len();
    let guide_time = result.data().iter().map(|i| i.total_time).sum();
    eprintln!(
        "Guide egraph (replay): {guide_nodes} nodes, {guide_classes} classes, \
         {guide_memory} live-heap bytes"
    );

    let mut root_log = String::new();

    let start_size = AstSize.cost_rec(&seed_expr);
    let (max_size, pc) = PrecomputePackage::<BigUint, _, _>::backoff_precompute(
        result,
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
    for pool in args.candidate_pools.iter().copied() {
        if candidates.contains_key(pool.name()) {
            continue;
        }
        let terms = draw_candidates(args, pool, &pc);
        candidates.insert(
            pool.name().to_owned(),
            terms.into_iter().map(GuideExpr::from_recexpr).collect(),
        );
    }

    Ok(SeedSamples {
        seed: args.seed.clone(),
        candidates,
        guide_nodes,
        guide_classes,
        guide_iters,
        guide_time,
        guide_memory,
        guide_peak_live_heap,
        stop_reason,
    })
}

/// Draw one candidate pool; smallest-term pools contain one term.
fn draw_candidates<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    pool: CandidatePool,
    pc: &PrecomputePackage<BigUint, L, N>,
) -> Vec<RecExpr<OriginLang<L>>> {
    match pool.sample_strategy() {
        // Replacement is a driver concern (how it re-draws subsets from the
        // pool across restarts); the pool itself is one novel sampled batch
        // either way. Sampling strategies always draw novel terms.
        Some(strategy) => pc
            .sample_frontier_terms(
                args.samples_per_strategy,
                args.size_distribution,
                strategy,
                [args.sampling_seed, pool.seed_of()],
            )
            .unwrap_or_else(|| {
                eprintln!(
                    "WARNING: strategy {} drew 0 candidates (empty novel frontier); \
                     driver legs for this strategy will have no guides to pick from",
                    pool.name()
                );
                Vec::new()
            }),
        None => vec![pc.smallest(pc.root(), matches!(pool, CandidatePool::SmallestNovel))],
    }
}
