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
