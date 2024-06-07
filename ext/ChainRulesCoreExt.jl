module ChainRulesCoreExt

# maybe move this module to src since ChainRulesCore is lite, and gets loaded by NNlib anyway.
# still nice to keep it distinct from the core package though.

using BatchedTransformations
using ChainRulesCore

using BatchedTransformations: batched_mul, batched_mul_T1, batched_mul_T2

function ChainRulesCore.rrule(::typeof(transform), rigid::RigidTransformations, x::AbstractArray)
    translations, rotations = outer(rigid), linear(rigid)
    t, R = values(translations), values(rotations)

    y = batched_mul(R, x) .+ t

    function transform_pullback(_Δy)
        Δy = unthunk(_Δy)

        ΔR = @thunk(batched_mul_T2(Δy, x))
        Δt = @thunk(sum(Δy, dims=2))
        Δx = @thunk(batched_mul_T1(R, Δy))

        Δtranslations = Tangent{typeof(translations)}(; values=Δt)
        Δrotations = Tangent{typeof(rotations)}(; values=ΔR)
        Δrigid = Tangent{typeof(rigid)}(; outer=Δtranslations, inner=Δrotations)

        return NoTangent(), Δrigid, Δx
    end

    return y, transform_pullback
end

function ChainRulesCore.rrule(::typeof(inverse_transform), rigid::RigidTransformations, x::AbstractArray)
    translations, rotations = translation(rigid), linear(rigid)
    t, R = values(translations), values(rotations)

    z = (x .- t)
    y = batched_mul_T1(R, z)

    function inverse_transform_pullback(_Δy)
        Δy = unthunk(_Δy)

        ΔR = @thunk(batched_mul_T2(z, Δy))
        Δx = @thunk(batched_mul(R, Δy))
        Δt = @thunk(-sum(Δx, dims=2)) # t is in the same position as x, but negated and broadcasted

        Δtranslations = Tangent{typeof(translations)}(; values=Δt)
        Δrotations = Tangent{typeof(rotations)}(;  values=ΔR)
        Δrigid = Tangent{typeof(rigid)}(; outer=Δtranslations, inner=Δrotations)

        return NoTangent(), Δrigid, Δx
    end

    return y, inverse_transform_pullback
end

end