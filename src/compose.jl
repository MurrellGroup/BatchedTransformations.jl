"""
    Composed{Outer<:Transformations,Inner<:Transformations}

A `Composed` contains two transformations `t2` and `t1` that are composed.
It can be constructed with `compose(t2, t1)` and `t2 ∘ t1`, where `t1` is the
transformation to be applied first, and `t2` second.
"""
struct Composed{Outer<:Transformations,Inner<:Transformations} <: Transformations
    outer::Outer
    inner::Inner
end

"""
    compose(t2, t1)
    t2 ∘ t1
"""
@inline compose(outer::Transformations, inner::Transformations) = Composed(outer, inner)

@inline Base.:(∘)(outer::Transformations, inner::Transformations) = compose(outer, inner)

outer(composed::Composed) = composed.outer
inner(composed::Composed) = composed.inner

transform(t::Composed, x) = transform(outer(t), transform(inner(t), x))

inverse_transform(t::Composed, x) = inverse_transform(inner(t), inverse_transform(outer(t), x))

Base.inv(t::Composed) = compose(inv(inner(t)), inv(outer(t)))

# t2, t1 = compose(t2, t1)
Base.iterate(t::Composed, state=1) = state == 1 ? (t.outer, 2) : (state == 2 ? (t.inner, nothing) : nothing)
