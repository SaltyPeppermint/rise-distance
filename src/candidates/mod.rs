//! Guide-candidate construction from an e-graph.
//!
//! The [`exact`] path uses complete term counts to draw candidates directly.
//! The rejection path uses low-memory proposal engines followed by exact
//! previous-boundary membership filtering.

mod allocation;
pub mod exact;
pub mod rejection;

pub use allocation::{ExactSelectionPolicy, SizeAllocation, uniform_candidate_allocation};
pub use exact::{BalanceConfig, ExactCandidatePackage};
pub use rejection::{
    FeasibilityEngine, ProposalEngine, RandomWalkEngine, RejectionBatch, RejectionCandidatePackage,
    RejectionLimits, RejectionStats, SizeRejectionStats,
};
