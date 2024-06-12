using LinearAlgebra: qr, Diagonal, diag, det

function _rand_rotation(n::Integer, T::Type{<:Real})
    A = randn(T, n, n)
    Q, R = qr(A)
    Q = Q * Diagonal(sign.(diag(R)))
    if det(Q) < 0
        Q[:, end] *= -1
    end
    return Q
end

function Base.rand(::Type{LinearMaps}, T::Type{<:Real}, m::Integer, n::Integer, batch_size::Dims)
    values = rand(T, m, n, batch_size...)
    return LinearMaps(values)
end

function Base.rand(::Type{Translations}, T::Type{<:Real}, n::Integer, batch_size::Dims)
    values = rand(T, n, 1, batch_size...)
    return Translations(values)
end

function Base.rand(::Type{AffineMaps}, T::Type{<:Real}, m::Integer, n::Integer, batch_size::Dims)
    translation = rand(Translations, T, m, batch_size)
    linear = rand(LinearMaps, T, m, n, batch_size)
    return AffineMaps(translation, linear)
end

function Base.rand(::Type{Rotations}, T::Type{<:Real}, n::Integer, batch_size::Dims)
    values = reshape(
        stack([_rand_rotation(n, T) for _ in 1:prod(batch_size)]),
        n, n, batch_size...)
    return Rotations(values)
end

function Base.rand(::Type{RigidTransformations}, T::Type{<:Real}, n::Integer, batch_size::Dims)
    translations = rand(Translations, T, n, batch_size)
    rotations = rand(Rotations, T, n, batch_size)
    return RigidTransformations(translations, rotations)
end

Base.rand(T::Type{<:Transformations}, args...) = rand(T, Float32, args...)