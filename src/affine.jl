# Abstract Linear
# ---------------

export AbstractLinearMaps

"""
    AbstractLinearMaps <: Transformations
"""
abstract type AbstractLinearMaps <: Transformations end

@inline linear(t::AbstractLinearMaps) = t

transform(t::AbstractLinearMaps, x::AbstractArray) = values(t) ⊠ x

# Linear
# ------

"""
    LinearMaps{A<:AbstractArray} <: AbstractLinearMaps
"""
struct LinearMaps{A<:AbstractArray} <: AbstractLinearMaps
    values::A
end

@inline Base.values(t::LinearMaps) = t.values

Base.inv(t::LinearMaps) = LinearMaps(mapslices(inv, values(t), dims=(1,2)))

# Rotation
# --------

"""
    Rotations{A<:AbstractArray} <: AbstractLinearMaps
"""
struct Rotations{A<:AbstractArray} <: AbstractLinearMaps
    values::A
end

@inline Base.values(t::Rotations) = t.values

Base.inv(t::Rotations{<:AbstractArray{<:Any,3}}) = Rotations(batched_transpose(values(t)))

# would ideally be a lazy transpose, but NNlib.batched_transpose only allows for 1 batch dimension
Base.inv(t::Rotations{<:AbstractArray{<:Any,N}}) where N = Rotations(permutedims(values(t), (2, 1, 3:N...)))

# Translation
# -----------

"""
    Translations{A<:AbstractArray} <: Transformations
"""
struct Translations{A<:AbstractArray} <: Transformations
    values::A
end

@inline translation(t::Translations) = t
@inline Base.values(t::Translations) = t.values

transform(t::Translations, x::AbstractArray) = x .+ values(t)
inverse_transform(t::Translations, x::AbstractArray) = x .- values(t)

Base.inv(t::Translations) = Translations(-values(t))

# Abstract Affine
# ---------------

const AbstractAffineMaps = ComposedTransformations{T,L} where {T<:Translations,L<:AbstractLinearMaps}

@inline translation(t::AbstractAffineMaps) = outer(t)
@inline linear(t::AbstractAffineMaps) = inner(t)

# Affine
# ------

const AffineMaps = AbstractAffineMaps{<:Translations,<:AbstractLinearMaps}

"""
    AffineMaps(translations::Translations, linear::AbstractLinearMaps)
"""
AffineMaps(translations::Translations, linear::AbstractLinearMaps) = ComposedTransformations(translations, linear)

# Rigid
# -----

const RigidTransformations = AbstractAffineMaps{<:Translations,<:Rotations}

"""
    RigidTransformations(translations::Translations, rotations::Rotations)
"""
RigidTransformations(translations::Translations, rotations::Rotations) = ComposedTransformations(translations, rotations)