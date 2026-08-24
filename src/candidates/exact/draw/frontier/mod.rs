//! Frontier-constrained exact drawing.
//!
//! [`space`] owns the shared frontier automaton and feasible derivations.
//! [`IndependentFrontierDrawer`] and [`BalancedFrontierDrawer`] choose
//! different distributions over that same constrained space.

mod balanced;
mod independent;
mod space;

pub use balanced::{BalanceConfig, BalancedFrontierDrawer};
pub use independent::IndependentFrontierDrawer;
