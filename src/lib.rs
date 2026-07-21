/// jemalloc as the process-wide allocator. Every binary in this workspace links
/// this crate, so declaring it here installs jemalloc for all of them (and for
/// the lib's own test harness) without a per-binary `#[global_allocator]`.
/// [`utils::live_heap_bytes`] reads jemalloc's live-heap stat and only returns
/// meaningful numbers because of this.
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod analysis;
pub mod cli;
mod counter;
pub mod eqsat;
pub mod generator;
pub mod langs;
mod origin;
pub mod sampling;
pub mod search;
pub mod sketch;
#[cfg(test)]
pub mod test_utils;
pub mod utils;
mod zs;

pub use counter::Counter;
pub use langs::{MyAnalysis, MyLanguage};
pub use origin::{OriginLang, lower};
pub use utils::{cheapest, cheapest_ilp, id0, stack_children};
pub use zs::{find_min_zs, tree_distance_unit};
