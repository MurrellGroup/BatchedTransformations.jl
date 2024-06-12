# Abstract Linear
# ---------------

export AbstractLinearMaps

"""
    AbstractLinearMaps <: Transformations
"""
abstract type AbstractLinearMaps <: Transformations end

@inline linear(t::AbstractLinearMaps) = t

transform(t::AbstractLinearMaps, x::AbstractArray) = values(t) ⊠ x

transform(l2::AbstractLinearMaps, l1::AbstractLinearMaps) = LinearMaps(l2(values(l1)))

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

transform(r2::Rotations, r1::Rotations) = Rotations(r2(values(r1)))

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

transform(t2::Translations, t1::Translations) = Translations(transform(t2, values(t1)))

# Abstract Affine
# ---------------

const AbstractAffineMaps = ComposedTransformations{T,L} where {T<:Translations,L<:AbstractLinearMaps}

@inline translation(t::AbstractAffineMaps) = outer(t)
@inline linear(t::AbstractAffineMaps) = inner(t)

# the following could be simplified with an IdentityTransformations type,
# that gets returned when we call e.g. linear(::Translations) or translation(::AbstractLinearMaps)
# but then this argument splatting might not work

transform(t2::Translations, l1::AbstractLinearMaps) = t2 ∘ l1
transform(l2::AbstractLinearMaps, t1::Translations) = Translations(l2(values(t1))) ∘ l2

transform(t2::Translations, (t1, l1)::AbstractAffineMaps) = t2(t1) ∘ l1
transform((t2, l2)::AbstractAffineMaps, t1::Translations) = Translations(t2(l2(values(t1)))) ∘ l2

transform(l2::AbstractLinearMaps, (t1, l1)::AbstractAffineMaps) = l2(t1) ∘ l2(l1)
transform((t2, l2)::AbstractAffineMaps, l1::AbstractLinearMaps) = t2 ∘ l2(l1)

transform((t2, l2)::AbstractAffineMaps, (t1, l1)::AbstractAffineMaps) = Translations(t2(l2(values(t1)))) ∘ l2(l1)

Base.inv(t::AbstractAffineMaps) = transform(inv(inner(t)), inv(outer(t)))

# Affine
# ------

const AffineMaps = AbstractAffineMaps{<:Translations,<:AbstractLinearMaps}

"""
    AffineMaps(translations::Translations, linear::AbstractLinearMaps)
"""
(::Type{AffineMaps})(translations::Translations, linear::AbstractLinearMaps) = ComposedTransformations(translations, linear)

# Rigid
# -----

const RigidTransformations = AbstractAffineMaps{<:Translations,<:Rotations}

"""
    RigidTransformations(translations::Translations, rotations::Rotations)
"""
(::Type{RigidTransformations})(translations::Translations, rotations::Rotations) = ComposedTransformations(translations, rotations)
