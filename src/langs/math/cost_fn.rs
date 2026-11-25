use egg::{CostFunction, Id, Language};

use super::Math;

const CHEAP: usize = 1;
const EXPENSIVE: usize = 100;

pub struct DiffIntExpensive;
impl CostFunction<Math> for DiffIntExpensive {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            Math::Diff(..) | Math::Integral(..) => EXPENSIVE,
            _ => CHEAP,
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
            Math::Diff(..) | Math::Integral(..) => CHEAP,
            _ => EXPENSIVE,
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
            Math::Add(..) => EXPENSIVE,
            _ => CHEAP,
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
            Math::Add(..) => CHEAP,
            _ => EXPENSIVE,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}

pub struct SillyCheap;
impl CostFunction<Math> for SillyCheap {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            Math::SillyAdd(..) => CHEAP,
            _ => EXPENSIVE,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}

pub struct TinyConstant;
impl CostFunction<Math> for TinyConstant {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let op_cost = match enode {
            Math::TinyConst(_) => CHEAP,
            _ => EXPENSIVE,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}
