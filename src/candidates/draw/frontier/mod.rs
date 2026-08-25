//! Frontier-constrained exact drawing.
//!
//! [`space`] owns the shared frontier automaton and feasible derivations.

mod independent;
mod space;

pub use independent::IndependentFrontierDrawer;
