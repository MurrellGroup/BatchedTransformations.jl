struct Identity <: Transformation end

transform(::Identity, x) = x
inverse_transform(::Identity, x) = x

Base.inv(::Identity) = Identity()

@inline compose(::Identity, ::Identity) = Identity()
@inline compose(::Identity, t::Transformation) = t
@inline compose(t::Transformation, ::Identity) = t