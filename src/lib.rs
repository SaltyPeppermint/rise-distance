/// jemalloc as the process-wide allocator. Every binary in this workspace links
/// this crate, so declaring it here installs jemalloc for all of them (and for
/// the lib's own test harness) without a per-binary `#[global_allocator]`.
/// [`utils::live_heap_bytes`] reads jemalloc's live-heap stat and only returns
/// meaningful numbers because of this.
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod analysis;
pub mod cli;
pub mod eqsat;
pub mod generator;
pub mod langs;
pub mod origin;
mod previous;
pub mod sampling;
pub mod search;
mod sketch;
pub mod utils;
pub mod zs;
