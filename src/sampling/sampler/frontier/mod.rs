//! Frontier-constrained sampling policies.
//!
//! [`space`] owns the shared frontier automaton and feasible derivations.
//! [`IndependentFrontierSampler`] and [`BalancedFrontierSampler`] choose
//! different distributions over that same constrained space.

mod balanced;
mod independent;
mod space;

pub use balanced::{BalanceConfig, BalancedFrontierSampler};
pub use independent::IndependentFrontierSampler;
