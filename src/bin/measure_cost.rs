use core::panic;
use std::time::Duration;

use clap::Parser;
use egg::{
    AstDepth, AstSize, BackoffScheduler, CostFunction, Extractor, Id, Iteration, IterationData,
    Language as _, LpCostFunction, LpExtractor, RecExpr, Rewrite, Runner, SimpleScheduler,
};
use hashbrown::HashMap;
use serde::Serialize;

use rise_distance::egg::math::{self, Math};
use rise_distance::egg::prop::{self, Prop};
use rise_distance::{MyAnalysis, MyLanguage, cli::argparse::Language};

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
    language: Language,
}

fn main() {
    let args = Args::parse();

    let costs = match args.language {
        Language::Math => {
            let expr: RecExpr<Math> = args
                .term
                .parse()
                .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));
            run(&args, &expr, &math::RULES)
        }
        Language::Prop => panic!("Not implemented"), // Language::Prop => {
                                                     //     let expr: RecExpr<Prop> = args
                                                     //         .term
                                                     //         .parse()
                                                     //         .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));
                                                     //     run::<_, _, CostThisRound<Prop>>(&args, &expr, &prop::RULES)
                                                     // }
    };

    println!("{}", serde_json::to_string(&costs).unwrap());
}

#[derive(Debug, Serialize)]
struct CostEvolution {
    monotonic: Vec<HashMap<&'static str, usize>>,
    novelty: Vec<HashMap<&'static str, bool>>,
}

impl CostEvolution {
    fn from_iterations<L, N>(iterations: Vec<Iteration<CostThisRound<L>>>) -> CostEvolution
    where
        L: MyLanguage,
        N: MyAnalysis<L>,
        CostThisRound<L>: IterationData<L, N>,
    {
        let novelty = CostEvolution::is_new_extract(&iterations);
        let monotonic = iterations
            .into_iter()
            .map(|i| i.data.monotonic_costs)
            .collect::<Vec<_>>();

        CostEvolution { monotonic, novelty }
    }

    fn is_new_extract<L, N>(
        iterations: &[Iteration<CostThisRound<L>>],
    ) -> Vec<HashMap<&'static str, bool>>
    where
        L: MyLanguage,
        N: MyAnalysis<L>,
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

fn run<L, N>(args: &Args, expr: &RecExpr<L>, rules: &[Rewrite<L, N>]) -> CostEvolution
where
    L: MyLanguage,
    N: MyAnalysis<L>,
    CostThisRound<L>: IterationData<L, N>,
{
    let runner = Runner::new(Default::default())
        .with_expr(expr)
        .with_iter_limit(args.max_iters)
        .with_node_limit(args.max_nodes)
        .with_time_limit(Duration::from_secs_f64(args.max_time));
    let runner = if args.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    };
    let runner = runner.run(rules);

    CostEvolution::from_iterations(runner.iterations)
}

#[derive(Debug, Serialize)]
struct CostThisRound<L: MyLanguage> {
    monotonic_costs: HashMap<&'static str, usize>,
    ilp_extracts: HashMap<&'static str, RecExpr<L>>,
}

impl<N: MyAnalysis<Math>> IterationData<Math, N> for CostThisRound<Math> {
    fn make(runner: &Runner<Math, N, Self>) -> Self {
        let mut monotonic_costs = HashMap::new();
        monotonic_costs.insert("AstSize", cheapest(runner, AstSize));
        monotonic_costs.insert("AstDepth", cheapest(runner, AstDepth));
        monotonic_costs.insert("AddExpensive", cheapest(runner, AddExpensive));
        monotonic_costs.insert("AddCheap", cheapest(runner, AddCheap));

        let mut ilp_extracts = HashMap::new();
        ilp_extracts.insert("AstSize", cheapest_ilp(runner, AstSize));

        CostThisRound {
            monotonic_costs,
            ilp_extracts,
        }
    }
}

fn cheapest<CF, L, N, C>(runner: &Runner<L, N, C>, cf: CF) -> usize
where
    CF: CostFunction<L, Cost = usize>,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    Extractor::new(&runner.egraph, cf).find_best_cost(runner.roots[0])
}

fn cheapest_ilp<CF, L, N, C>(runner: &Runner<L, N, C>, cf: CF) -> RecExpr<L>
where
    CF: LpCostFunction<L, N>,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    LpExtractor::new(&runner.egraph, cf).solve(runner.roots[0])
}

pub struct DiffIntExpensive;
impl CostFunction<Math> for DiffIntExpensive {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            Math::Diff(..) | Math::Integral(..) => 100,
            _ => 1,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}

pub struct DiffIntCheap;
impl CostFunction<Math> for DiffIntCheap {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            Math::Diff(..) | Math::Integral(..) => 1,
            _ => 100,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}

pub struct AddExpensive;
impl CostFunction<Math> for AddExpensive {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            Math::Add(..) => 100,
            _ => 1,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}
pub struct AddCheap;
impl CostFunction<Math> for AddCheap {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            Math::Add(..) => 1,
            _ => 100,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}
