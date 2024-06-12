# TODO: evaluate the composition of two affine maps and construct a new affine map

"""
    ComposedTransformations{Outer<:Transformations,Inner<:Transformations}

A `ComposedTransformations` contains two transformations `t2` and `t1` that are composed.
It can be constructed with `compose(t2, t1)`, `t2 ∘ t1`, and `t2(t1)`, where `t1` is the
transformation to be applied first, and `t2` second.
"""
struct ComposedTransformations{Outer<:Transformations,Inner<:Transformations} <: Transformations
    outer::Outer
    inner::Inner
end

"""
    compose(t2, t1)
    t2 ∘ t1
"""
@inline compose(outer::Transformations, inner::Transformations) = ComposedTransformations(outer, inner)

@inline Base.:(∘)(outer::Transformations, inner::Transformations) = compose(outer, inner)

@inline outer(composed::ComposedTransformations) = composed.outer
@inline inner(composed::ComposedTransformations) = composed.inner

transform(t::ComposedTransformations, x::AbstractArray) = transform(outer(t), transform(inner(t), x))
inverse_transform(t::ComposedTransformations, x::AbstractArray) = inverse_transform(inner(t), inverse_transform(outer(t), x))

Base.inv(t::ComposedTransformations) = compose(inv(inner(t)), inv(outer(t)))

# t2, t1 = compose(t2, t1)
Base.iterate(t::ComposedTransformations, state=1) = state == 1 ? (t.outer, 2) : (state == 2 ? (t.inner, nothing) : nothing)