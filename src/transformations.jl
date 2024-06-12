"""
    Transformations

An abstract type whose concrete subtypes contain batches of transformations
that can be applied to an array. A `Transformations` `t` can be applied to
`x` with `transform(t, x)`, `t * x`, andt(x).
"""
abstract type Transformations end

"""
    transform(t, x)
    t * x
    t(x)
"""
transform(t::Transformations, ::Any) = error("transform not defined for $(typeof(t)) and $(typeof(x))")

Base.inv(t::Transformations) = error("inverse not defined for $(typeof(t))")
inverse_transform(t::Transformations, x) = transform(inv(t), x)

@inline Base.:(*)(t::Transformations, x) = transform(t, x)
@inline (t::Transformations)(x) = transform(t, x)

function Base.show(io::IO, ::MIME"text/plain", t::Transformations)
    print(io, summary(t))
end

"""
    transform(t2, t1)

A non-lazy composition that instantiates a new transformation.
"""
transform(t2::Transformations, t1::Transformations) = error("transform not defined for $(typeof(t2)) and $(typeof(t1))")