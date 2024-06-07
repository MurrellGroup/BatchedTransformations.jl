# Abstract Linear
# ---------------

export AbstractLinearMaps

abstract type AbstractLinearMaps <: Transformations end

@inline linear(t::AbstractLinearMaps) = t

transform(t::AbstractLinearMaps, x::AbstractArray) = values(t) ⊠ x

# Linear
# ------

struct LinearMaps{A<:AbstractArray} <: AbstractLinearMaps
    values::A
end

@inline Base.values(t::LinearMaps) = t.values

Base.inv(t::LinearMaps) = LinearMaps(mapslices(inv, values(t), dims=(1,2)))

# Translation
# -----------

struct Translations{A<:AbstractArray} <: Transformations
    values::A
end

@inline translation(t::Translations) = t
@inline Base.values(t::Translations) = t.values

transform(t::Translations, x::AbstractArray) = x .+ values(t)
inverse_transform(t::Translations, x::AbstractArray) = x .- values(t)

Base.inv(t::Translations) = Translations(-values(t))

# Affine
# ------

const AbstractAffineMaps = ComposedTransformations{T,L} where {T<:Translations,L<:AbstractLinearMaps}

@inline translation(t::AbstractAffineMaps) = outer(t)
@inline linear(t::AbstractAffineMaps) = inner(t)

const AffineMaps = AbstractAffineMaps{<:Translations,<:LinearMaps}

AffineMaps(translations::Translations, linear::LinearMaps) = ComposedTransformations(translations, linear)