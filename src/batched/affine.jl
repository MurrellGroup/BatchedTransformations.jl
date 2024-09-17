abstract type AbstractAffine <: GeometricTransformation end

function translation end
function linear end

Base.iterate(affine::AbstractAffine, state=0) = state == 0 ? (translation(affine), 1) : (state == 1 ? (linear(affine), nothing) : nothing)

abstract type Homomorphic end
abstract type Endomorphic <: Homomorphic end
abstract type Automorphic <: Endomorphic end

struct Linear{M<:Homomorphic,A<:AbstractArray} <: AbstractAffine
    values::A
end

abstract type Orthonormal{Det} <: Automorphic end

const Rotation = Linear{Orthonormal{1}}
const Reflection = Linear{Orthonormal{-1}}

@inline Linear{M}(values::A) where {M,A} = Linear{M,A}(values)
@inline Linear{M}(linear::Linear) where M = Linear{M}(values(linear))

@inline function Linear{M}(values::A) where {M<:Endomorphic,A<:AbstractArray}
    size(values, 1) == size(values, 2) || error("rotation values must have size (n, n, batchdims...)")
    Linear{M,A}(values)
end

@inline function Linear(values::A) where A<:AbstractArray
    M = size(values, 1) == size(values, 2) ? Endomorphic : Homomorphic
    Linear{M,A}(values)
end

@inline compose(l2::Linear{M1}, l1::Linear{M2}) where {M1<:Homomorphic,M2<:Homomorphic} = Linear{typejoin(M1,M2)}(l2 * values(l1))

@inline linear(linear::Linear) = linear
@inline translation(::Linear) = Identity()

@inline Base.values(linear::Linear) = linear.values
@inline Base.:(==)(l1::Linear, l2::Linear) = values(l1) == values(l2)

batchsize(linear::Linear) = size(values(linear))[3:end]

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

Base.inv(t::Linear{M,<:AbstractArray{<:Any,2}}) where M<:Orthonormal = Linear{M}(transpose(values(t)))
Base.inv(t::Linear{M,<:AbstractArray{<:Any,3}}) where M<:Orthonormal = Linear{M}(batched_transpose(values(t)))
Base.inv(t::Linear{M}) where M<:Orthonormal = Linear{M}(permutedims(values(t), (2, 1, 3:ndims(values(t))...)))


struct Translation{A<:AbstractArray} <: AbstractAffine
    values::A

    function Translation{A}(values::A) where A<:AbstractArray
        size(values, 2) == 1 || error("translation values must have size (n, 1, batchdims...)")
        new{A}(values)
    end
end

Translation(values::A) where A = Translation{A}(values)

@inline linear(::Translation) = Identity()
@inline translation(translation::Translation) = translation

@inline Base.values(translation::Translation) = translation.values
@inline Base.:(==)(t1::Translation, t2::Translation) = values(t1) == values(t2)

batchsize(translation::Translation) = size(values(translation))[3:end]

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


struct Affine{T<:Translation,L<:Linear{<:Automorphic}} <: AbstractAffine
    composed::Composed{T,L}
end

const Rigid = Affine{<:Translation,<:Rotation}

@inline linear(affine::Affine) = inner(affine.composed)
@inline translation(affine::Affine) = outer(affine.composed)

@inline Base.:(==)(affine1::Affine, affine2::Affine) = affine1.composed == affine2.composed

function batchunsqueeze((translation,linear)::Affine; dims::Int)
    batchunsqueeze(translation; dims) ∘ batchunsqueeze(linear; dims)
end

transform(affine::Affine, x::AbstractArray) = transform(affine.composed, x)
inverse_transform(affine::Affine, x::AbstractArray) = inverse_transform(affine.composed, x)

Base.inv(affine::Affine) = inv(affine.composed)

Base.show(io::IO, affine::Affine) = print(io, "$(translation(affine)) ∘ $(linear(affine))")

@inline compose(translation::Translation, linear::Linear) = Affine(Composed(translation, linear))
@inline compose(linear::Linear, translation::Translation) = Translation(linear * values(translation)) ∘ linear

@inline compose((t2,l2)::AbstractAffine, (t1,l1)::AbstractAffine) = (t2 ∘ (l2 ∘ t1)) ∘ l1
@inline compose((t2,l2)::AbstractAffine, l1::Linear) = t2 ∘ (l2 ∘ l1)
@inline compose(t2::Translation, (t1,l1)::AbstractAffine) = (t2 ∘ t1) ∘ l1