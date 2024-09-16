module BatchedTransformations

include("core.jl")
export Transformation, transform, inverse_transform
export batchsize
export Identity
export Composed, compose
export outer, inner
export Inverse, inverse

include("geometric/geometric.jl")
export GeometricTransformation, AbstractAffine, AbstractLinear
export Translation, Linear, Affine
export Rotation, Rigid
export linear, translation

end