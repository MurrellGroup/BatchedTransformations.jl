module ChainRulesCoreExt

# maybe move this module to src since ChainRulesCore is lite, and gets loaded by NNlib anyway.
# still nice to keep it distinct from the core package though.

using BatchedTransformations
using ChainRulesCore

using BatchedTransformations: batched_transpose

function ChainRulesCore.rrule(::typeof(transform), rigid::RigidTransformations, x::AbstractArray)
    translations, rotations = outer(rigid), linear(rigid)
    t, R = values(translations), values(rotations)

    y = R ⊠ x .+ t

    function transform_pullback(_Δy)
        Δy = unthunk(_Δy)

        ΔR = @thunk(Δy ⊠ batched_transpose(x))
        Δt = @thunk(sum(Δy, dims=2))
        Δx = @thunk(batched_transpose(R) ⊠ Δy)

        Δtranslations = Tangent{typeof(translations)}(; values=Δt)
        Δrotations = Tangent{typeof(rotations)}(; linear=Tangent{typeof(rotations.linear)}(; values=ΔR))
        Δrigid = Tangent{typeof(rigid)}(; outer=Δtranslations, inner=Δrotations)

        return NoTangent(), Δrigid, Δx
    end

    return y, transform_pullback
end

function ChainRulesCore.rrule(::typeof(inverse_transform), rigid::RigidTransformations, x::AbstractArray)
    translations, rotations = translation(rigid), linear(rigid)
    t, R = values(translations), values(rotations)

    z = (x .- t)
    y = batched_transpose(R) ⊠ z

    function inverse_transform_pullback(_Δy)
        Δy = unthunk(_Δy)

        ΔR = @thunk(z ⊠ batched_transpose(Δy))
        Δx = @thunk(R ⊠ Δy)
        Δt = @thunk(-sum(Δx, dims=2)) # t is in the same position as x, but negated and broadcasted

        Δtranslations = Tangent{typeof(translations)}(; values=Δt)
        Δrotations = Tangent{typeof(rotations)}(; linear=Tangent{typeof(rotations.linear)}(; values=ΔR))
        Δrigid = Tangent{typeof(rigid)}(; outer=Δtranslations, inner=Δrotations)

        return NoTangent(), Δrigid, Δx
    end

    return y, inverse_transform_pullback
end

end