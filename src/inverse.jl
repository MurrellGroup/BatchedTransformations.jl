"""
    Inverse{T<:Transformation}

An `Inverse` represents a *lazy* inverse of a `Transformation` t.

`inverse(t)` is a lazy inverse that defaults to `inv(t)` when evaluated.
`transform(inverse(t), x)` is equivalent to `inverse_transform(t, x)`.
This allows for specialized inverse transform implementations that don't
require the inverse to be computed explicitly.
"""
struct Inverse{T<:Transformation} <: Transformation
    parent::T
end

Base.:(==)(t1::Inverse, t2::Inverse) = t1.parent == t2.parent

batchsize(t::Inverse) = batchsize(t.parent)

@inline inverse(t::Transformation) = Inverse(t)
@inline inverse(t::Inverse) = t.parent

@inline transform(t::Inverse, x) = inverse_transform(t.parent, x)

@inline Base.inv(t::Inverse) = t.parent

function compose(t2::Inverse{T}, t1::T) where T<:Transformation
    t2.parent === t1 && return Identity()
    Composed(t2, t1)
end

function compose(t2::T, t1::Inverse{T}) where T<:Transformation
    t2 === t1.parent && return Identity()
    Composed(t2, t1)
end
