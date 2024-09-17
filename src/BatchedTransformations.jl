module BatchedTransformations

include("core.jl")
export Transformation, transform, inverse_transform
export Identity
export Composed, compose
export outer, inner
export Inverse, inverse

include("batched/batched.jl")
export BatchedTransformation
export batchsize, batchreshape, batchunsqueeze
export AbstractAffine, translation, linear
export Translation
export Homomorphic, Endomorphic, Automorphic
export Linear, Orthonormal, Rotation, Reflection
export Affine, Rigid

end