using LinearAlgebra: qr, Diagonal, diag, det
using Random: AbstractRNG, default_rng

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Linear}, (n, m)::Pair{<:Integer,<:Integer}, batch_size::Dims=())
    values = randn(rng, T, m, n, batch_size...)
    return Linear(values)
end

Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Linear}, n::Integer, batch_size::Dims=()) =
    rand(rng, T, Linear, n => n, batch_size)

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Translation}, n::Integer, batch_size::Dims=())
    values = randn(rng, T, n, 1, batch_size...)
    return Translation(values)
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Affine}, n::Integer, batch_size::Dims=())
    translation = rand(rng, T, Translation, n, batch_size)
    linear = rand(rng, T, Linear, n, batch_size)
    return translation ∘ linear
end

function rand_rotation(rng::AbstractRNG, T::Type{<:Real}, n::Integer)
    A = randn(rng, T, n, n)
    Q, R = qr(A)
    Q = Q * Diagonal(sign.(diag(R)))
    det(Q) < 0 && (Q[:, end] *= -1)
    return Q
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Rotation}, n::Integer, batch_size::Dims=())
    values = reshape(stack([rand_rotation(rng, T, n) for _ in 1:prod(batch_size)]), n, n, batch_size...)
    return Rotation(values)
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Rigid}, n::Integer, batch_size::Dims=())
    translations = rand(rng, T, Translation, n, batch_size)
    rotations = rand(rng, T, Rotation, n, batch_size)
    return translations ∘ rotations
end

Base.rand(T::Type{<:Real}, Tr::Type{<:Transformation}, dims, batch_size::Dims=()) = rand(default_rng(), T, Tr, dims, batch_size)
Base.rand(Tr::Type{<:Transformation}, dims, batch_size::Dims) = rand(Float64, Tr, dims, batch_size)
Base.rand(Tr::Type{<:Transformation}, dims::Integer) = rand(Tr, dims, ())