# TODO: evaluate the composition of two affine maps and construct a new affine map

"""
    ComposedTransformations{T2<:Transformations,T1<:Transformations}

A `ComposedTransformations` contains two transformations `t2` and `t1` that are composed.
It can be constructed with `compose(t2, t1)`, `t2 ∘ t1`, and `t2(t1)`, where t1 is the
transformation to be applied first, and `t2` second.
"""
struct ComposedTransformations{T2<:Transformations,T1<:Transformations} <: Transformations
    t2::T2
    t1::T1
end

compose(t2::Transformations, t1::Transformations) = ComposedTransformations(t2, t1)

Base.:(∘)(t2::Transformations, t1::Transformations) = compose(t2, t1)
(t2::Transformations)(t1::Transformations) = compose(t2, t1)

transform(t::ComposedTransformations, x::AbstractArray) = transform(t.t2, transform(t.t1, x))
inverse_transform(t::ComposedTransformations, x::AbstractArray) = inverse_transform(t.t1, inverse_transform(t.t2, x))

Base.inv(t::ComposedTransformations) = compose(inv(t.t1), inv(t.t2))
