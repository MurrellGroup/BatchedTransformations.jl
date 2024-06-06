"""
Both `batched_transpose` and `batched_adjoint` are defined in NNlib,
but they do not allow for multiple batch dimensions.
"""

"""
    batched_transpose(data::AbstractArray)

A lazy batched transpose of an array `data`, swapping the first two dimensions.
"""
function batched_transpose(data::A) where {T,N,A<:AbstractArray{T,N}}
    perm = (2,1,3:N...)
    PermutedDimsArray{T,N,perm,perm,A}(data)
end

#=
# FIXME: this rrule breaks when PermutedDimsArray is used, but not when permutedims is used
# also see https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/LinearAlgebra/structured.jl#L187
# TODO: see if ProjectTo(x)(Δy) is necessary
_batched_transpose_pullback(Δy::NoTangent) = (NoTangent(), Δy)
_batched_transpose_pullback(Δy::AbstractArray) = (NoTangent(), batched_transpose(Δy))
_batched_transpose_pullback(Δy::AbstractThunk) = _batched_transpose_pullback(unthunk(Δy))
function ChainRulesCore.rrule(::typeof(batched_transpose), x::AbstractArray)
    batched_transpose_pullback(Δy) = _batched_transpose_pullback(Δy)
    batched_transpose(x), batched_transpose_pullback
end
=#

batched_adjoint(data::AbstractArray{<:Real}) = batched_transpose(data)

# might need custom chain rule
# could do map(det, eachslice(data, dims=size(data)[3:end], drop=false))
batched_det(data::AbstractArray{<:Real}) = mapslices(det, data, dims=(1,2))