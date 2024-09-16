using NNlib: ⊠, batched_mul, batched_transpose

include("batched_utils.jl")

abstract type GeometricTransformation <: Transformation end

abstract type AbstractAffine <: GeometricTransformation end

function translation end
function linear end

Base.iterate(affine::AbstractAffine, args...) = iterate(affine.composed, args...)

abstract type AbstractLinear <: AbstractAffine end

@inline linear(linear::AbstractLinear) = linear
@inline translation(::AbstractLinear) = Identity()

@inline Base.values(linear::AbstractLinear) = linear.values
@inline Base.:(==)(l1::AbstractLinear, l2::AbstractLinear) = values(l1) == values(l2)
@inline batchsize(linear::AbstractLinear) = size(values(linear))[3:end]

transform(l::AbstractLinear, x::AbstractArray) = values(l) ⊠ x

function transform(linear::AbstractLinear, x::AbstractVecOrMat)
    A = values(linear)
    batch_size = size(A)[3:end]
    A′ = reshape(A, size(A, 1), size(A, 2), :)
    y′ = A′ ⊠ x
    y = reshape(y′, size(A, 1), size(y′, 2), batch_size...)
    return y
end

@inline compose(l2::AbstractLinear, l1::AbstractLinear) = Linear(l2 * values(l1))


struct Linear{A<:AbstractArray} <: AbstractLinear
    values::A
end

Base.inv(t::Linear) = Linear(mapslices(inv, values(t), dims=(1,2)))


struct Translation{A<:AbstractArray} <: AbstractAffine
    values::A
end

@inline linear(::Translation) = Identity()
@inline translation(translation::Translation) = translation

@inline Base.values(translation::Translation) = translation.values
@inline Base.:(==)(t1::Translation, t2::Translation) = values(t1) == values(t2)
@inline batchsize(translation::Translation) = size(values(translation))[3:end]

transform(t::Translation, x::AbstractArray) = x .+ values(t)
inverse_transform(t::Translation, x::AbstractArray) = x .- values(t)

Base.inv(t::Translation) = Translation(-values(t))

@inline compose(t2::Translation, t1::Translation) = Translation(t2 * values(t1))


struct Affine{T<:Translation,L<:AbstractLinear} <: AbstractAffine
    composed::Composed{T,L}
end

@inline linear(affine::Affine) = inner(affine.composed)
@inline translation(affine::Affine) = outer(affine.composed)

@inline Base.:(==)(affine1::Affine, affine2::Affine) = affine1.composed == affine2.composed

transform(affine::Affine, x::AbstractArray) = transform(affine.composed, x)
inverse_transform(affine::Affine, x::AbstractArray) = inverse_transform(affine.composed, x)

Base.inv(affine::Affine) = inv(affine.composed)

Base.show(io::IO, affine::Affine) = print(io, "$(translation(affine)) ∘ $(linear(affine))")

@inline compose(translation::Translation, linear::AbstractLinear) = Affine(Composed(translation, linear))
@inline compose(linear::AbstractLinear, translation::Translation) = Translation(linear * values(translation)) ∘ linear

@inline compose((t2,l2)::AbstractAffine, (t1,l1)::AbstractAffine) = (t2 ∘ (l2 ∘ t1)) ∘ l1
@inline compose((t2,l2)::AbstractAffine, l1::AbstractLinear) = t2 ∘ (l2 ∘ l1)
@inline compose(t2::Translation, (t1,l1)::AbstractAffine) = (t2 ∘ t1) ∘ l1


struct Rotation{A<:AbstractArray} <: AbstractLinear
    values::A
end

inverse_transform(r::Rotation, x::AbstractArray) = batched_mul_T1(values(r), x)

Base.inv(t::Rotation{<:AbstractArray{<:Any,3}}) = Rotation(batched_transpose(values(t)))
Base.inv(t::Rotation{<:AbstractArray{<:Any,N}}) where N = Rotation(permutedims(values(t), (2, 1, 3:N...)))

@inline compose(r2::Rotation, r1::Rotation) = Rotation(r2 * values(r1))

const Rigid = Affine{<:Translation,<:Rotation}


include("rand.jl")
