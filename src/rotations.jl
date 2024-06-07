# Rotations
# ---------

struct Rotations{A<:AbstractArray} <: AbstractLinearMaps
    values::A
end

@inline Base.values(t::Rotations) = t.values

Base.inv(t::Rotations{<:AbstractArray{<:Any,3}}) = Rotations(batched_transpose(values(t)))

# would ideally be a lazy transpose, but NNlib.batched_transpose only allows for 1 batch dimension
Base.inv(t::Rotations{<:AbstractArray{<:Any,N}}) where N = Rotations(permutedims(values(t), (2, 1, 3:N...)))

# Rigid
# -----

const RigidTransformations = AbstractAffineMaps{<:Translations,<:Rotations}

RigidTransformations(translations::Translations, rotations::Rotations) = ComposedTransformations(translations, rotations)