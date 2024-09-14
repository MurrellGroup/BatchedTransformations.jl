module BatchedTransformations

include("transformations.jl")
export Transformations, transform, inverse_transform

include("identity.jl")
export Identity

include("inverse.jl")
export Inverse, inverse

include("compose.jl")
export Composed, compose
export outer, inner

include("affine/affine.jl")
export AbstractLinearMaps, LinearMaps, Rotations
export Translations
export AbstractAffineMaps, AffineMaps, RigidTransformations
export translation, linear

include("rand.jl")

end