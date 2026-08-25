//! Guide-candidate construction from an e-graph.
//!
//! The [`exact`] path uses complete term counts to draw candidates directly.

mod allocation;
pub mod exact;

pub use allocation::{ExactSelectionPolicy, SizeAllocation, uniform_candidate_allocation};
pub use exact::{BalanceConfig, ExactCandidatePackage};
