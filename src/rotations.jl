# Rotations
# ---------

struct Rotations{L<:LinearMaps} <: AbstractLinearMaps
    linear::L
end

Rotations(rotations::AbstractArray) = Rotations(LinearMaps(rotations))

@inline Base.values(t::Rotations) = values(t.linear)

Base.inv(t::Rotations) = Rotations(batched_transpose(values(t)))

# Rigid
# -----

const RigidTransformations = AbstractAffineMaps{<:Translations,<:Rotations}

RigidTransformations(translations::Translations, rotations::Rotations) = ComposedTransformations(translations, rotations)