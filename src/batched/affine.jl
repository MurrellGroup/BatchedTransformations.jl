abstract type AbstractAffine{T} <: GeometricTransformation{T} end

function translation end
function linear end

Base.iterate(affine::AbstractAffine, state=0) = state == 0 ? (translation(affine), 1) : (state == 1 ? (linear(affine), nothing) : nothing)

abstract type Homomorphic end
abstract type Endomorphic <: Homomorphic end
abstract type Automorphic <: Endomorphic end

struct Linear{M<:Homomorphic,T,A<:AbstractArray{T}} <: AbstractAffine{T}
    values::A
end

abstract type Orthonormal{Det} <: Automorphic end

const Rotation = Linear{Orthonormal{1}}
const Reflection = Linear{Orthonormal{-1}}

@inline Linear{M}(values::A) where {M,T,A<:AbstractArray{T}} = Linear{M,T,A}(values)
@inline Linear{M}(linear::Linear) where M = Linear{M}(values(linear))

@inline function Linear{M}(values::A) where {M<:Endomorphic,T,A<:AbstractArray{T}}
    size(values, 1) == size(values, 2) || error("endomorphic linear map values must have size (n, n, batchdims...)")
    Linear{M,T,A}(values)
end

@inline function Linear(values::A) where {T,A<:AbstractArray{T}}
    M = size(values, 1) == size(values, 2) ? Endomorphic : Homomorphic
    Linear{M,T,A}(values)
end

@inline compose(l2::Linear{M1}, l1::Linear{M2}) where {M1<:Homomorphic,M2<:Homomorphic} = Linear{typejoin(M1,M2)}(l2 * values(l1))

@inline linear(linear::Linear) = linear
@inline translation(::Linear) = Identity()

@inline Base.values(linear::Linear) = linear.values
@inline Base.:(==)(l1::Linear, l2::Linear) = values(l1) == values(l2)

batchsize(linear::Linear) = size(values(linear))[3:end]

function batchrepeat(linear::Linear{M}, args...) where M
    A = values(linear)
    Linear{M}(repeat(A, 1, 1, args...))
end

function batchreshape(linear::Linear{M}, args...) where M
    A = values(linear)
    Linear{M}(reshape(A, size(A, 1), size(A, 2), args...))
end

function batchunsqueeze(linear::Linear{M}; dims::Int) where M
    @assert dims > 0
    Linear{M}(unsqueeze(values(linear), dims=dims+2))
end

transform(l::Linear, x::AbstractArray) = values(l) ⊠ x

transform(linear::Linear, x::AbstractVecOrMat) = batched_mul_large_small(values(linear), x)

inverse_transform(t::Linear{<:Orthonormal}, x::AbstractArray) = batched_mul_T1(values(t), x)

Base.inv(t::Linear{M}) where M<:Automorphic = Linear{M}(mapslices(inv, values(t), dims=(1,2)))

Base.inv(t::Linear{M,T,<:AbstractArray{T,2}}) where {M<:Orthonormal,T} = Linear{M}(transpose(values(t)))
Base.inv(t::Linear{M,T,<:AbstractArray{<:Any,3}}) where {M<:Orthonormal,T} = Linear{M}(batched_transpose(values(t)))
Base.inv(t::Linear{M}) where M<:Orthonormal = Linear{M}(permutedims(values(t), (2, 1, 3:ndims(values(t))...)))


struct Translation{T,A<:AbstractArray{T}} <: AbstractAffine{T}
    values::A

    function Translation{T,A}(values::A) where {T,A<:AbstractArray{T}}
        size(values, 2) == 1 || error("translation values must have size (n, 1, batchdims...)")
        new{T,A}(values)
    end
end

Translation(values::A) where {T,A<:AbstractArray{T}} = Translation{T,A}(values)

@inline linear(::Translation) = Identity()
@inline translation(translation::Translation) = translation

@inline Base.values(translation::Translation) = translation.values
@inline Base.:(==)(t1::Translation, t2::Translation) = values(t1) == values(t2)

batchsize(translation::Translation) = size(values(translation))[3:end]

function batchrepeat(translation::Translation, args...)
    b = values(translation)
    Translation(repeat(b, 1, 1, args...))
end

function batchreshape(translation::Translation, args...)
    b = values(translation)
    Translation(reshape(b, size(b, 1), 1, args...))
end

function batchunsqueeze(translation::Translation; dims::Int)
    @assert dims > 0
    Translation(unsqueeze(values(translation), dims=dims+2))
end

transform(t::Translation, x::AbstractArray) = x .+ values(t)
inverse_transform(t::Translation, x::AbstractArray) = x .- values(t)

Base.inv(t::Translation) = Translation(-values(t))

@inline compose(t2::Translation, t1::Translation) = Translation(t2 * values(t1))


struct Affine{T,O<:Translation{T},I<:Linear{<:Automorphic,T}} <: AbstractAffine{T}
    composed::Composed{O,I}
end

const Rigid = Affine{T,<:Translation,<:Rotation} where T

@inline linear(affine::Affine) = inner(affine.composed)
@inline translation(affine::Affine) = outer(affine.composed)

@inline Base.:(==)(affine1::Affine, affine2::Affine) = affine1.composed == affine2.composed

batchrepeat(affine::Affine, args...) =
    batchrepeat(translation(affine), args...) ∘ batchrepeat(linear(affine), args...)

batchreshape(affine::Affine, args...) =
    batchreshape(translation(affine), args...) ∘ batchreshape(linear(affine), args...)

batchunsqueeze((t,l)::Affine; dims::Int) =
    batchunsqueeze(t; dims) ∘ batchunsqueeze(l; dims)

transform(affine::Affine, x::AbstractArray) = transform(affine.composed, x)
inverse_transform(affine::Affine, x::AbstractArray) = inverse_transform(affine.composed, x)

Base.inv(affine::Affine) = inv(affine.composed)

Base.show(io::IO, affine::Affine) = print(io, "$(translation(affine)) ∘ $(linear(affine))")

@inline compose(translation::Translation, linear::Linear) = Affine(Composed(translation, linear))
@inline compose(linear::Linear, translation::Translation) = Translation(linear * values(translation)) ∘ linear

@inline compose((t2,l2)::AbstractAffine, (t1,l1)::AbstractAffine) = (t2 ∘ (l2 ∘ t1)) ∘ l1
@inline compose((t2,l2)::AbstractAffine, l1::Linear) = t2 ∘ (l2 ∘ l1)
@inline compose(t2::Translation, (t1,l1)::AbstractAffine) = (t2 ∘ t1) ∘ l1
