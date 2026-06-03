use core::panic;

use clap::Parser;
use egg::{AstDepth, AstSize, Iteration, IterationData, RecExpr, Rewrite, Runner};
use hashbrown::HashMap;
use rise_distance::eqsat::EqsatConfig;
use rise_distance::langs::AvailableLanguages;
use rise_distance::langs::diospyros;
use serde::Serialize;

use rise_distance::cheapest;
use rise_distance::langs::math::{self, AddCheap, AddExpensive, Math, SillyCheap, TinyConstant};
use rise_distance::langs::prop::{self, Prop};
use rise_distance::{MyAnalysis, MyLanguage};

#[derive(Parser)]
#[command(about = "Run eqsat on a single term see how the cost evolves.")]
struct Args {
    /// Term to run eqsat on (s-expression).
    #[arg(long)]
    term: String,

    /// Iter limit for the runner
    #[arg(long, default_value_t = 11)]
    max_iters: usize,

    /// Node limit for the runner
    #[arg(long, default_value_t = 100_000)]
    max_nodes: usize,

    /// Time limit for the runner (seconds)
    #[arg(long, default_value_t = 1.0)]
    max_time: f64,

    /// Use egg's `BackoffScheduler` instead of the `SimpleScheduler`
    #[arg(long, default_value_t = false)]
    backoff_scheduler: bool,

    /// Language to sample terms from
    #[arg(long)]
    language: AvailableLanguages,
}

fn main() {
    let args = Args::parse();

    let costs = match args.language {
        AvailableLanguages::Diospyros => {
            let expr = args
                .term
                .parse()
                .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));
            run(&args, &expr, &diospyros::rules(false, false))
        }
        AvailableLanguages::Math => {
            let expr = args
                .term
                .parse()
                .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));
            run(&args, &expr, &math::silly_rules())
        }
        AvailableLanguages::Prop => {
            let expr = args
                .term
                .parse()
                .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));
            run(&args, &expr, &prop::rules())
        }
    };

    println!("{}", serde_json::to_string(&costs).unwrap());
}

#[derive(Debug, Serialize)]
struct CostEvolution {
    monotonic: Vec<HashMap<&'static str, usize>>,
    novelty: Vec<HashMap<&'static str, bool>>,
}

impl CostEvolution {
    fn from_iterations<L: MyLanguage, N: MyAnalysis<L>>(
        iterations: Vec<Iteration<CostThisRound<L>>>,
    ) -> CostEvolution
    where
        CostThisRound<L>: IterationData<L, N>,
    {
        let novelty = CostEvolution::is_new_extract(&iterations);
        let monotonic = iterations
            .into_iter()
            .map(|i| i.data.monotonic_costs)
            .collect::<Vec<_>>();

        CostEvolution { monotonic, novelty }
    }

    fn is_new_extract<L: MyLanguage, N: MyAnalysis<L>>(
        iterations: &[Iteration<CostThisRound<L>>],
    ) -> Vec<HashMap<&'static str, bool>>
    where
        CostThisRound<L>: IterationData<L, N>,
    {
        iterations
            .iter()
            .scan(None::<&HashMap<&'static str, RecExpr<L>>>, |prev, iter| {
                let current = &iter.data.ilp_extracts;
                let newness = current
                    .iter()
                    .map(|(k, v)| {
                        let is_new = prev
                            .and_then(|p| p.get(k))
                            .is_none_or(|prev_v| prev_v.to_string() != v.to_string());
                        (*k, is_new)
                    })
                    .collect();
                *prev = Some(current);
                Some(newness)
            })
            .collect()
    }
}

fn run<L: MyLanguage, N: MyAnalysis<L>>(
    args: &Args,
    expr: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
) -> CostEvolution
where
    CostThisRound<L>: IterationData<L, N>,
{
    eprintln!("Now running {expr}");
    let config = EqsatConfig {
        max_iters: args.max_iters,
        max_nodes: args.max_nodes,
        max_time: args.max_time,
        backoff_scheduler: args.backoff_scheduler,
    };
    let runner = config
        .build_runner::<_, _, CostThisRound<L>>(expr)
        .run(rules);

    CostEvolution::from_iterations(runner.iterations)
}

#[derive(Debug, Serialize)]
struct CostThisRound<L: MyLanguage> {
    monotonic_costs: HashMap<&'static str, usize>,
    ilp_extracts: HashMap<&'static str, RecExpr<L>>,
}

impl<N: MyAnalysis<Math>> IterationData<Math, N> for CostThisRound<Math> {
    fn make(runner: &Runner<Math, N, Self>) -> Self {
        eprintln!("Now running iteration {}", runner.iterations.len());
        let mut monotonic_costs = HashMap::new();
        monotonic_costs.insert("AstSize", cheapest(runner, AstSize));
        monotonic_costs.insert("AstDepth", cheapest(runner, AstDepth));
        monotonic_costs.insert("AddExpensive", cheapest(runner, AddExpensive));
        monotonic_costs.insert("AddCheap", cheapest(runner, AddCheap));
        monotonic_costs.insert("SillyCheap", cheapest(runner, SillyCheap));
        monotonic_costs.insert("TinyConstant", cheapest(runner, TinyConstant));

        let ilp_extracts = HashMap::new();
        // ilp_extracts.insert("AstSize", cheapest_ilp(runner, AstSize));

        CostThisRound {
            monotonic_costs,
            ilp_extracts,
        }
    }
}

impl<N: MyAnalysis<Prop>> IterationData<Prop, N> for CostThisRound<Prop> {
    fn make(runner: &Runner<Prop, N, Self>) -> Self {
        eprintln!("Now running iteration {}", runner.iterations.len());
        let mut monotonic_costs = HashMap::new();
        monotonic_costs.insert("AstSize", cheapest(runner, AstSize));
        monotonic_costs.insert("AstDepth", cheapest(runner, AstDepth));

        let ilp_extracts = HashMap::new();

        CostThisRound {
            monotonic_costs,
            ilp_extracts,
        }
    }
}

impl IterationData<diospyros::VecLang, ()> for CostThisRound<diospyros::VecLang> {
    fn make(runner: &Runner<diospyros::VecLang, (), Self>) -> Self {
        eprintln!("Now running iteration {}", runner.iterations.len());
        let mut monotonic_costs = HashMap::new();
        monotonic_costs.insert("AstSize", cheapest(runner, AstSize));
        monotonic_costs.insert("AstDepth", cheapest(runner, AstDepth));

        CostThisRound {
            monotonic_costs,
            ilp_extracts: HashMap::new(),
        }
    }
}
