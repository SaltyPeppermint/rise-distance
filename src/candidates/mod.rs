//! Guide-candidate construction from an e-graph.

mod allocation;

pub mod count;
pub mod draw;
mod package;

pub use allocation::{ExactSelectionPolicy, SizeAllocation, uniform_candidate_allocation};

pub use draw::BalanceConfig;
pub use package::ExactCandidatePackage;
