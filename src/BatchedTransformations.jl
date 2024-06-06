module BatchedTransformations

using NNlib: ⊠
export ⊠

include("batched_utils.jl")

include("transformations.jl")
export Transformations, transform, inverse_transform

include("inverse.jl")
export InverseTransformations, inverse

include("compose.jl")
export ComposedTransformations, compose

include("affine.jl")
export AbstractAffineMaps, LinearMaps, Translations, AffineMaps
export linear, translation

include("rotations.jl")
export Rotations, RigidTransformations

include("rand.jl")

end
