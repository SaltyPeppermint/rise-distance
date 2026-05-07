mod count;
mod naive;

pub use count::CountWeigher;
pub use naive::NaiveWeigher;

use crate::count::Counter;

pub trait Weigher<C: Counter>: Sync + Send {
    fn node_weight(&self, count: &C) -> C;
    fn child_weight(&self, child_count: &C, rest_count: &C) -> C;
}
