//! Guide-candidate construction from an e-graph.

mod allocation;

pub mod count;
pub mod draw;
mod package;

pub use allocation::{SelectionPolicy, SizeAllocation, uniform_candidate_allocation};

pub use package::ExactCandidatePackage;
