# TODO: evaluate the composition of two affine maps and construct a new affine map

"""
    ComposedTransformations{Outer<:Transformations,Inner<:Transformations}

A `ComposedTransformations` contains two transformations `outer` and `inner` that are composed.
It can be constructed with `compose(outer, inner)`, `outer ∘ inner`, and `outer(inner)`, where inner is the
transformation to be applied first, and `outer` second.
"""
struct ComposedTransformations{Outer<:Transformations,Inner<:Transformations} <: Transformations
    outer::Outer
    Inner::Inner
end

@inline compose(outer::Transformations, inner::Transformations) = ComposedTransformations(outer, inner)

@inline Base.:(∘)(outer::Transformations, inner::Transformations) = compose(outer, inner)
@inline (outer::Transformations)(inner::Transformations) = compose(outer, inner)

@inline outer(composed::ComposedTransformations) = composed.outer
@inline inner(composed::ComposedTransformations) = composed.inner

transform(t::ComposedTransformations, x::AbstractArray) = transform(outer(t), transform(inner(t), x))
inverse_transform(t::ComposedTransformations, x::AbstractArray) = inverse_transform(inner(t), inverse_transform(outer(t), x))

Base.inv(t::ComposedTransformations) = compose(inv(inner(t)), inv(outer(t)))
