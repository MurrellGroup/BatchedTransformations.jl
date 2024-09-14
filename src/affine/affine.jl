include("batched_utils.jl")
include("linear.jl")
include("translation.jl")

const AffineMaps = Composed{<:Translations,<:LinearMaps}
const RigidTransformations = Composed{<:Translations,<:Rotations}

const AbstractAffineMaps = Union{AffineMaps,RigidTransformations}

translation(a::AbstractAffineMaps) = outer(a)
linear(a::AbstractAffineMaps) = inner(a)

function Base.inv(a::AbstractAffineMaps)
    inv_t, inv_l = inv(translation(a)), inv(linear(a))
    return inv_l ∘ inv_t
end

@inline compose(l2::AbstractLinearMaps, t1::Translations) = Translations(l2 * values(t1)) ∘ l2

@inline compose((t2,l2)::AbstractAffineMaps, (t1,l1)::AbstractAffineMaps) = (t2 ∘ (l2 ∘ t1)) ∘ l1
@inline compose((t2,l2)::AbstractAffineMaps, l1::AbstractLinearMaps) = t2 ∘ (l2 ∘ l1)
@inline compose((t2,l2)::AbstractAffineMaps, t1::Translations) = t2 ∘ (l2 ∘ t1)
@inline compose(l2::AbstractLinearMaps, (t1,l1)::AbstractAffineMaps) = (l2 ∘ t1) ∘ l1
@inline compose(t2::Translations, (t1,l1)::AbstractAffineMaps) = (t2 ∘ t1) ∘ l1
