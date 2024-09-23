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

function batched_mul_large_small(x::AbstractArray, y::AbstractVecOrMat)
    x′ = reshape(x, size(x, 1), size(x, 2), :)
    z′ = batched_mul(x′, reshape(y, size(y, 1), size(y, 2)))
    z = reshape(z′, size(x, 1), size(y, 2), size(x)[3:end]...)
    return z
end

# might need custom chain rule
# could also try map(det, eachslice(data, dims=size(data)[3:end], drop=false))
batched_det(data::AbstractArray{<:Real}) = mapslices(det, data, dims=(1,2))
