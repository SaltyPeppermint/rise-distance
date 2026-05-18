use std::time::Duration;

use clap::Parser;
use egg::{
    Analysis, AstDepth, AstSize, BackoffScheduler, CostFunction, Extractor, Id, IterationData,
    Language, RecExpr, Runner, SimpleScheduler,
};
use hashbrown::HashMap;
use serde::Serialize;

use rise_distance::egg::math::{ConstantFold, Math, RULES};

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
}

fn main() {
    let args = Args::parse();

    let expr: RecExpr<Math> = args
        .term
        .parse()
        .unwrap_or_else(|e| panic!("Failed to parse term '{}': {e}", args.term));

    let runner = Runner::<_, _, CostThisRound>::new(ConstantFold)
        .with_expr(&expr)
        .with_iter_limit(args.max_iters)
        .with_node_limit(args.max_nodes)
        .with_time_limit(Duration::from_secs_f64(args.max_time));
    let runner = if args.backoff_scheduler {
        runner.with_scheduler(BackoffScheduler::default())
    } else {
        runner.with_scheduler(SimpleScheduler)
    };
    let runner = runner.run(&*RULES);

    let costs = runner
        .iterations
        .into_iter()
        .map(|i| i.data.0)
        .collect::<Vec<_>>();

    println!("{}", serde_json::to_string(&costs).unwrap());
}

#[derive(Debug, Serialize)]
struct CostThisRound(HashMap<&'static str, usize>);
impl IterationData<Math, ConstantFold> for CostThisRound {
    fn make(runner: &Runner<Math, ConstantFold, Self>) -> Self {
        let mut costs = HashMap::new();
        costs.insert("AstSize", cheapest(runner, AstSize));
        costs.insert("AstDepth", cheapest(runner, AstDepth));
        costs.insert("DiffIntExpensive", cheapest(runner, DiffIntExpensive));
        costs.insert("DiffIntCheap", cheapest(runner, DiffIntCheap));
        costs.insert("AddExpensive", cheapest(runner, AddExpensive));
        costs.insert("AddCheap", cheapest(runner, AddCheap));
        CostThisRound(costs)
    }
}

fn cheapest<CF, L, N>(runner: &Runner<L, N, CostThisRound>, cf: CF) -> usize
where
    CF: CostFunction<L, Cost = usize>,
    L: Language,
    N: Analysis<L>,
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
