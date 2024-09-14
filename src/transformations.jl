"""
    Transformations

An abstract type whose concrete subtypes contain batches of transformations
that can be applied to an array. A `Transformations` `t` can be applied to
`x` with `transform(t, x)`, `t * x`, and t(x).
"""
abstract type Transformations end

"""
    transform(t, x)
    t * x
    t(x)
"""
transform(t::Transformations, x) = error("transform not defined for $(typeof(t)) and $(typeof(x))")

Base.inv(t::Transformations) = error("inverse not defined for $(typeof(t)) ")
@inline inverse_transform(t::Transformations, x) = transform(inv(t), x)

@inline Base.:(*)(t::Transformations, x) = transform(t, x)
@inline (t::Transformations)(x) = transform(t, x)

Base.show(io::IO, ::MIME"text/plain", t::Transformations) = print(io, summary(t))

