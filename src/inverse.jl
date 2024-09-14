"""
    Inverse{T<:Transformations}

An `Inverse` represents a *lazy* inverse of a `Transformations` t.

`inverse(t)` is a lazy inverse that defaults to `inv(t)` when evaluated.
`transform(inverse(t), x)` is equivalent to `inverse_transform(t, x)`.
This allows for specialized inverse transform implementations that don't
require the inverse to be computed explicitly.
"""
struct Inverse{T<:Transformations} <: Transformations
    t::T
end

inverse(t::Transformations) = Inverse(t)
inverse(t::Inverse) = t.t

transform(t::Inverse, x) = inverse_transform(t.t, x)

Base.inv(t::Inverse) = t.t
