"""
    InverseTransformations{T<:Transformations}

A `InverseTransformations` represents a *lazy* inverse of a `Transformations` t.

`inverse(t)` is a lazy inverse that defaults to `inv(t)` when evaluated.
`transform(inverse(t), x)` is equivalent to `inverse_transform(t, x)`.
This allows for specialized inverse transform implementations,
that don't require the inverse to be computed explicitly.
"""
struct InverseTransformations{T<:Transformations} <: Transformations
    t::T
end

@inline inverse(t::Transformations) = InverseTransformations(t)
@inline inverse(t::InverseTransformations) = t.t

transform(t::InverseTransformations, x::AbstractArray) = inverse_transform(t.t, x)

Base.inv(t::InverseTransformations) = t.t