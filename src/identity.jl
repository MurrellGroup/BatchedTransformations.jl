struct Identity{T<:Transformations} <: Transformations end

transform(::Identity, t::Identity) = t
transform(::Identity, t::Transformations) = t
transform(t::Transformations, ::Identity) = t
