"""
    Translations{A<:AbstractArray} <: Transformations

Contains a batch of n-dimensional translation vectors,
represented by an array of size `(n, 1, b1, b2, ...)`.
"""
struct Translations{A<:AbstractArray} <: Transformations
    values::A
end

Base.values(t::Translations) = t.values

linear(::Translations) = Identity{AbstractLinearMaps}()
translation(t::Translations) = t

transform(t::Translations, x::AbstractArray) = x .+ values(t)
inverse_transform(t::Translations, x::AbstractArray) = x .- values(t)

Base.inv(t::Translations) = Translations(-values(t))

transform(t2::Translations, t1::Translations) = Translations(transform(t2, values(t1)))

transform(l::AbstractLinearMaps, t::Translations) = Translations(transform(l, values(t)))
transform(::AbstractLinearMaps, t::Identity{Translations}) = t
