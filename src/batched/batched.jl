using NNlib: ⊠, batched_mul, batched_transpose
using MLUtils: unsqueeze

include("batched_utils.jl")

function batchsize end
function batchreshape end
function batchunsqueeze end

batchsize(t::Transformation, d::Integer) = batchsize(t)[d]
batchsize(t::Inverse{<:Transformation}) = batchsize(t.parent)

abstract type GeometricTransformation{T} <: Transformation end

include("affine.jl")
include("rand.jl")

using Adapt

Adapt.adapt_structure(to, linear::Linear{M}) where M = Linear{M}(Adapt.adapt(to, linear.values))
