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
