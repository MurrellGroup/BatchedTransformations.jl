module BatchedTransformations

include("transformations.jl")
export Transformation, transform, inverse_transform
export batchsize

include("identity.jl")
export Identity

include("inverse.jl")
export Inverse, inverse

include("compose.jl")
export Composed, compose
export outer, inner

include("geometric/geometric.jl")
export GeometricTransformation, AbstractAffine, AbstractLinear
export Translation, Linear, Affine
export Rotation, Rigid
export linear, translation

end