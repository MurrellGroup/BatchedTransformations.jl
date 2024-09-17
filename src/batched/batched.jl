using NNlib: ⊠, batched_mul, batched_transpose
using MLUtils: unsqueeze

include("batched_utils.jl")

function batchsize end
function batchreshape end
function batchunsqueeze end

batchsize(t::Transformation, d::Integer) = batchsize(t)[d]
batchsize(t::Inverse{<:Transformation}) = batchsize(t.parent)

abstract type GeometricTransformation <: Transformation end

include("affine.jl")
include("rand.jl")

