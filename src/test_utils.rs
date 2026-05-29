use crate::langs::math::Math;

#[must_use]
pub fn sym(name: &str) -> Math {
    Math::Symbol(name.into())
}
