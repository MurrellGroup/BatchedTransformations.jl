module BatchedTransformations

include("batched_utils.jl")

include("transformations.jl")
export Transformations, transform, inverse_transform

include("inverse.jl")
export InverseTransformations, inverse

include("compose.jl")
export ComposedTransformations, compose
export outer, inner

include("affine.jl")
export AbstractLinearMaps, LinearMaps, Rotations
export Translations
export AbstractAffineMaps, AffineMaps, RigidTransformations
export translation, linear

include("rand.jl")

end
