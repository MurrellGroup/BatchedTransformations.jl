"""
    AbstractLinearMaps <: Transformations
"""
abstract type AbstractLinearMaps <: Transformations end

Base.values(t::AbstractLinearMaps) = t.values

linear(l::AbstractLinearMaps) = l
translation(::AbstractLinearMaps) = Identity{Translations}()

transform(l::AbstractLinearMaps, x::AbstractArray) = values(l) ⊠ x

transform(l2::AbstractLinearMaps, l1::AbstractLinearMaps) = LinearMaps(l2(values(l1)))


"""
    LinearMaps{A<:AbstractArray} <: AbstractLinearMaps

Contains a batch of linear maps mapping from n-dimensional to m-dimensional space,
represented by an array of size `(m, n, b1, b2, ...)`.
"""
struct LinearMaps{A<:AbstractArray} <: AbstractLinearMaps
    values::A
end

Base.inv(t::LinearMaps) = LinearMaps(mapslices(inv, values(t), dims=(1,2)))


"""
    Rotations{A<:AbstractArray} <: AbstractLinearMaps

Contains a batch of n-dimensional rotations matrices,
represented by an array of size `(n, n, b1, b2, ...)`.
"""
struct Rotations{A<:AbstractArray} <: AbstractLinearMaps
    values::A
end

Base.inv(t::Rotations{<:AbstractArray{<:Any,3}}) = Rotations(batched_transpose(values(t)))
Base.inv(t::Rotations{<:AbstractArray{<:Any,N}}) where N = Rotations(permutedims(values(t), (2, 1, 3:N...)))

inverse_transform(r::Rotations, x::AbstractArray) = batched_mul_T1(values(r), x)

transform(r2::Rotations, r1::Rotations) = Rotations(r2(values(r1)))
