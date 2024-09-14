include("batched_utils.jl")
include("linear.jl")
include("translation.jl")

const AbstractAffineMaps = Composed{<:Translations,<:AbstractLinearMaps}
const AffineMaps = Composed{<:Translations,<:LinearMaps}
const RigidTransformations = Composed{<:Translations,<:Rotations}

translation(a::AbstractAffineMaps) = outer(a)
linear(a::AbstractAffineMaps) = inner(a)

function Base.inv(a::AbstractAffineMaps)
    inv_t, inv_l = inv(translation(a)), inv(linear(a))
    return inv_l(inv_t) ∘ inv_l
end

function transform(
    a2::Union{AbstractAffineMaps,AbstractLinearMaps,Translations},
    a1::Union{AbstractAffineMaps,AbstractLinearMaps,Translations},
)
    t2, l2 = translation(a2), linear(a2)
    t1, l1 = translation(a1), linear(a1)
    return t2(l2(t1)) ∘ l2(l1)
end
