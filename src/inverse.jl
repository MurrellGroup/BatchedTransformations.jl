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

@inline inverse(t::Transformations) = Inverse(t)
@inline inverse(t::Inverse) = t.t

@inline transform(t::Inverse, x) = inverse_transform(t.t, x)

@inline Base.inv(t::Inverse) = t.t
