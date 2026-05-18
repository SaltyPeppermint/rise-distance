use std::time::Duration;

use clap::Parser;
use egg::{
    AstDepth, AstSize, BackoffScheduler, CostFunction, Extractor, Id, IterationData, Language as _,
    RecExpr, Rewrite, Runner, SimpleScheduler,
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
            run::<_, _, MathCostThisRound>(&args, &expr, &math::RULES)
        }
        Language::Prop => {
            let expr: RecExpr<Prop> = args
                .term
                .parse()
                .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));
            run::<_, _, PropCostThisRound>(&args, &expr, &prop::RULES)
        }
    };

    println!("{}", serde_json::to_string(&costs).unwrap());
}

fn run<L: MyLanguage, N: MyAnalysis<L>, I: CostStat<L, N>>(
    args: &Args,
    expr: &RecExpr<L>,
    rules: &[Rewrite<L, N>],
) -> Vec<HashMap<&'static str, usize>> {
    let runner = Runner::<_, _, I>::new(Default::default())
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

    runner
        .iterations
        .into_iter()
        .map(|i| i.data.get())
        .collect::<Vec<_>>()
}

trait CostStat<L: MyLanguage, N: MyAnalysis<L>>: IterationData<L, N> {
    fn get(self) -> HashMap<&'static str, usize>;
}

#[derive(Debug, Serialize)]
struct MathCostThisRound(HashMap<&'static str, usize>);
impl IterationData<Math, math::ConstantFold> for MathCostThisRound {
    fn make(runner: &Runner<Math, math::ConstantFold, Self>) -> Self {
        let mut costs = HashMap::new();
        costs.insert("AstSize", cheapest(runner, AstSize));
        costs.insert("AstDepth", cheapest(runner, AstDepth));
        costs.insert("AddExpensive", cheapest(runner, AddExpensive));
        costs.insert("AddCheap", cheapest(runner, AddCheap));
        MathCostThisRound(costs)
    }
}

impl CostStat<Math, math::ConstantFold> for MathCostThisRound {
    fn get(self) -> HashMap<&'static str, usize> {
        self.0
    }
}

#[derive(Debug, Serialize)]
struct PropCostThisRound(HashMap<&'static str, usize>);
impl IterationData<Prop, prop::ConstantFold> for PropCostThisRound {
    fn make(runner: &egg::Runner<Prop, prop::ConstantFold, PropCostThisRound>) -> Self {
        let mut costs = HashMap::new();
        costs.insert("AstSize", cheapest(runner, AstSize));
        costs.insert("AstDepth", cheapest(runner, AstDepth));
        PropCostThisRound(costs)
    }
}

impl CostStat<Prop, prop::ConstantFold> for PropCostThisRound {
    fn get(self) -> HashMap<&'static str, usize> {
        self.0
    }
}

fn cheapest<CF, L, N, C: CostStat<L, N>>(runner: &Runner<L, N, C>, cf: CF) -> usize
where
    CF: CostFunction<L, Cost = usize>,
    L: MyLanguage,
    N: MyAnalysis<L>,
{
    Extractor::new(&runner.egraph, cf).find_best_cost(runner.roots[0])
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
