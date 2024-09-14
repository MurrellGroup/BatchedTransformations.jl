using LinearAlgebra: qr, Diagonal, diag, det
using Random: AbstractRNG, default_rng

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{LinearMaps}, (n, m)::Pair{<:Integer,<:Integer}, batch_size::Dims)
    values = rand(rng, T, m, n, batch_size...)
    return LinearMaps(values)
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Translations}, n::Integer, batch_size::Dims)
    values = rand(rng, T, n, 1, batch_size...)
    return Translations(values)
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{AffineMaps}, (n, m)::Pair{<:Integer,<:Integer}, batch_size::Dims)
    translation = rand(rng, T, Translations, m, batch_size)
    linear = rand(rng, T, LinearMaps, n => m, batch_size)
    return translation ∘ linear
end

function rand_rotation(rng::AbstractRNG, T::Type{<:Real}, n::Integer)
    A = randn(rng, T, n, n)
    Q, R = qr(A)
    Q = Q * Diagonal(sign.(diag(R)))
    det(Q) < 0 && (Q[:, end] *= -1)
    return Q
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{Rotations}, n::Integer, batch_size::Dims)
    values = reshape(stack([rand_rotation(rng, T, n) for _ in 1:prod(batch_size)]), n, n, batch_size...)
    return Rotations(values)
end

function Base.rand(rng::AbstractRNG, T::Type{<:Real}, ::Type{RigidTransformations}, n::Integer, batch_size::Dims)
    translations = rand(rng, T, Translations, n, batch_size)
    rotations = rand(rng, T, Rotations, n, batch_size)
    return translations ∘ rotations
end

Base.rand(T::Type{<:Real}, Tr::Type{<:Transformations}, dims, batch_size::Dims) = rand(default_rng(), T, Tr, dims, batch_size)
Base.rand(Tr::Type{<:Transformations}, dims, batch_size::Dims) = rand(Float64, Tr, dims, batch_size)
