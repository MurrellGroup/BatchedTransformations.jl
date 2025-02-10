using LinearAlgebra: qr, Diagonal, diag, det
using Random: AbstractRNG, default_rng

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Linear}, (n, m)::Pair{<:Integer,<:Integer}, batchdims::Dims=())
    values = randn(rng, T, m, n, batchdims...)
    return Linear(values)
end

Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Linear}, n::Integer, batchdims::Dims=()) =
    rand(rng, T, Linear, n => n, batchdims)

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Translation}, n::Integer, batchdims::Dims=())
    values = randn(rng, T, n, 1, batchdims...)
    return Translation(values)
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Affine}, n::Integer, batchdims::Dims=())
    translation = rand(rng, T, Translation, n, batchdims)
    linear = Linear{Automorphic}(values(rand(rng, T, Linear, n, batchdims))) # doesn't actually guarantee invertibility :/
    return translation ∘ linear
end

function rand_rotation(rng::AbstractRNG, T::Type{<:Real}, n::Integer)
    A = randn(rng, T, n, n)
    Q, R = qr(A)
    Q = Q * Diagonal(sign.(diag(R)))
    det(Q) < 0 && (Q[:, end] *= -1)
    return Q
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Rotation}, n::Integer, batchdims::Dims=())
    values = reshape(stack([rand_rotation(rng, T, n) for _ in 1:prod(batchdims)]), n, n, batchdims...)
    return Rotation(values)
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{QuaternionRotation}, n::Integer, batchdims::Dims=())
    @assert n == 3
    return convert(QuaternionRotation, rand(rng, T, Rotation, 3, batchdims))
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Rigid}, n::Integer, batchdims::Dims=())
    translations = rand(rng, T, Translation, n, batchdims)
    rotations = rand(rng, T, Rotation, n, batchdims)
    return translations ∘ rotations
end

Base.rand(T::Type{<:Real}, Tr::Type{<:Transformation}, dims, batchdims::Dims=()) = rand(default_rng(), T, Tr, dims, batchdims)
Base.rand(Tr::Type{<:Transformation}, dims, batchdims::Dims=()) = rand(Float64, Tr, dims, batchdims)
Base.rand(Tr::Type{<:Transformation}, dims::Integer) = rand(Tr, dims, ()) # needed to avoid ambiguity with other rand methods