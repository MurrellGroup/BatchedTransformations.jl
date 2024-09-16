"""
    Transformation

An abstract type whose concrete subtypes contain batches of transformations
that can be applied to an array. A `Transformation` `t` can be applied to
`x` with `transform(t, x)`, `t * x`, and t(x).
"""
abstract type Transformation end

function compose end
function batchsize end

"""
    transform(t, x)
    t * x
    t(x)
"""
transform(t::Transformation, x) = error("transform not defined for $(typeof(t)) and $(typeof(x))")

@inline Base.:(*)(t::Transformation, x) = transform(t, x)
@inline (t::Transformation)(x) = transform(t, x)

Base.inv(t::Transformation) = error("inverse not defined for $(typeof(t))")

@inline inverse_transform(t::Transformation, x) = transform(inv(t), x)

Base.show(io::IO, ::MIME"text/plain", t::Transformation) = print(io, summary(t))


"""
    Identity <: Transformation
"""
struct Identity <: Transformation end

transform(::Identity, x) = x
inverse_transform(::Identity, x) = x

Base.inv(::Identity) = Identity()

@inline compose(::Identity, ::Identity) = Identity()
@inline compose(::Identity, t::Transformation) = t
@inline compose(t::Transformation, ::Identity) = t


"""
    Composed{Outer<:Transformation,Inner<:Transformation}

A `Composed` contains two transformations `outer` and `inner` that are composed,
where `inner` gets applied first, and then `outer`..
It can be constructed with `compose(outer, inner)` or `outer ∘ inner`, unless
the `compose` function is overloaded for the specific types.
"""
struct Composed{Outer<:Transformation,Inner<:Transformation} <: Transformation
    outer::Outer
    inner::Inner
end

"""
    compose(t2, t1)
    t2 ∘ t1
"""
@inline compose(outer::Transformation, inner::Transformation) = Composed(outer, inner)

@inline Base.:(∘)(outer::Transformation, inner::Transformation) = compose(outer, inner)

@inline Base.:(==)(a1::Composed, a2::Composed) = a1.outer == a2.outer && a1.inner == a2.inner

@inline outer(composed::Composed) = composed.outer
@inline inner(composed::Composed) = composed.inner

@inline transform(t::Composed, x) = transform(outer(t), transform(inner(t), x))

@inline inverse_transform(t::Composed, x) = inverse_transform(inner(t), inverse_transform(outer(t), x))

@inline Base.inv(t::Composed) = inv(inner(t)) ∘ inv(outer(t))

# enables `outer, inner = compose(outer, inner)` syntax
Base.iterate(t::Composed, state=1) = state == 1 ? (t.outer, 2) : (state == 2 ? (t.inner, nothing) : nothing)


"""
    Inverse{T<:Transformation} <: Transformation

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

@inline function compose(t2::Inverse{T}, t1::T) where T<:Transformation
    t2.parent === t1 && return Identity()
    Composed(t2, t1)
end

@inline function compose(t2::T, t1::Inverse{T}) where T<:Transformation
    t2 === t1.parent && return Identity()
    Composed(t2, t1)
end
