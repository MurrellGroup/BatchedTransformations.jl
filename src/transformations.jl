"""
    Transformation

An abstract type whose concrete subtypes contain batches of transformations
that can be applied to an array. A `Transformation` `t` can be applied to
`x` with `transform(t, x)`, `t * x`, and t(x).
"""
abstract type Transformation end

batchsize(t::Transformation) = error("batchsize not defined for $(typeof(t))")

"""
    transform(t, x)
    t * x
    t(x)
"""
transform(t::Transformation, x) = error("transform not defined for $(typeof(t)) and $(typeof(x))")

Base.inv(t::Transformation) = error("inverse not defined for $(typeof(t)) ")
@inline inverse_transform(t::Transformation, x) = transform(inv(t), x)

@inline Base.:(*)(t::Transformation, x) = transform(t, x)
@inline (t::Transformation)(x) = transform(t, x)

Base.show(io::IO, ::MIME"text/plain", t::Transformation) = print(io, summary(t))

