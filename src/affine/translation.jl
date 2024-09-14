"""
    Translations{A<:AbstractArray} <: Transformations

Contains a batch of n-dimensional translation vectors,
represented by an array of size `(n, 1, b1, b2, ...)`.
"""
struct Translations{A<:AbstractArray} <: Transformations
    values::A
end

Base.values(t::Translations) = t.values

translation(t::Translations) = t

transform(t::Translations, x::AbstractArray) = x .+ values(t)
inverse_transform(t::Translations, x::AbstractArray) = x .- values(t)

Base.inv(t::Translations) = Translations(-values(t))

@inline compose(t2::Translations, t1::Translations) = Translations(t2 * values(t1))