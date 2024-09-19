function batched_mul_T1(x::AbstractArray, y::AbstractArray)
    x2 = reshape(x, size(x, 1), size(x, 2), :) |> batched_transpose
    y2 = reshape(y, size(y, 1), size(y, 2), :)
    z = batched_mul(x2, y2)
    return reshape(z, size(z, 1), size(z, 2), size(x)[3:end]...)
end

function batched_mul_T2(x::AbstractArray, y::AbstractArray)
    x2 = reshape(x, size(x, 1), size(x, 2), :)
    y2 = reshape(y, size(y, 1), size(y, 2), :) |> batched_transpose
    z = batched_mul(x2, y2)
    return reshape(z, size(z, 1), size(z, 2), size(x)[3:end]...)
end

function batched_mul_large_small(A::AbstractArray, x::AbstractVecOrMat)
    A′ = reshape(A, size(A, 1), size(A, 2), :)
    y′ = batched_mul(A′, reshape(x, size(x, 1), size(x, 2)))
    y = reshape(y′, size(A, 1), size(x, 2))
    return y
end

# might need custom chain rule
# could also try map(det, eachslice(data, dims=size(data)[3:end], drop=false))
batched_det(data::AbstractArray{<:Real}) = mapslices(det, data, dims=(1,2))
