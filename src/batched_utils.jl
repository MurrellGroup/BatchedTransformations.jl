using NNlib: ⊠, batched_mul, batched_transpose

function batched_mul_T1(x::AbstractArray{T1,N}, y::AbstractArray{T2,N}) where {T1,T2,N}
    batch_size = size(x)[3:end]
    @assert batch_size == size(y)[3:end] "batch size has to be the same for the two arrays."
    x2 = reshape(x, size(x, 1), size(x, 2), :) |> batched_transpose
    y2 = reshape(y, size(y, 1), size(y, 2), :)
    z = batched_mul(x2, y2)
    return reshape(z, size(z, 1), size(z, 2), batch_size...)
end

function batched_mul_T2(x::AbstractArray{T1,N}, y::AbstractArray{T2,N}) where {T1,T2,N}
    batch_size = size(x)[3:end]
    @assert batch_size == size(y)[3:end] "batch size has to be the same for the two arrays."
    x2 = reshape(x, size(x, 1), size(x, 2), :)
    y2 = reshape(y, size(y, 1), size(y, 2), :) |> batched_transpose
    z = batched_mul(x2, y2)
    return reshape(z, size(z, 1), size(z, 2), batch_size...)
end

# might need custom chain rule
# could do map(det, eachslice(data, dims=size(data)[3:end], drop=false))
batched_det(data::AbstractArray{<:Real}) = mapslices(det, data, dims=(1,2))

#=
# doesn't work with batched_mul on GPU
function _batched_transpose(data::A) where {T,N,A<:AbstractArray{T,N}}
    perm = (2,1,3:N...)
    PermutedDimsArray{T,N,perm,perm,A}(data)
end

_batched_adjoint(data::AbstractArray{<:Real}) = _batched_transpose(data)

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