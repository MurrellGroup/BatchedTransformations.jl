"""
    Transformations

A `Transformations` contains a batch of transformations that can be applied to an array.
A `Transformations` `t` can be applied to `x` with `transform(t, x)`, `t ∘ x`, and t(x).
"""
abstract type Transformations end

"""
    transform(t, x)
    t ∘ x
    t(x)
"""
transform(t::Transformations, ::AbstractArray) = error("transform not defined for $(typeof(t))")

Base.inv(t::Transformations) = error("inverse not defined for $(typeof(t))")
inverse_transform(t::Transformations, x::AbstractArray) = transform(inv(t), x)

@inline Base.:(∘)(t::Transformations, x::AbstractArray) = transform(t, x)
@inline (t::Transformations)(x::AbstractArray) = transform(t, x)

function Base.show(io::IO, ::MIME"text/plain", t::Transformations)
    print(io, summary(t))
end
