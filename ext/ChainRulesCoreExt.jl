module ChainRulesCoreExt

# maybe move this module to src since ChainRulesCore is lite, and gets loaded by NNlib anyway.
# still nice to keep it distinct from the core package though.

using BatchedTransformations
using ChainRulesCore

using BatchedTransformations: batched_transpose

function ChainRulesCore.rrule(::typeof(transform), rigid::RigidTransformations, x::AbstractArray)
    t2, t1 = rigid.t2, rigid.t1
    t, R = values(t2), values(t1)

    y = R ⊠ x .+ t

    function transform_pullback(_Δy)
        Δy = unthunk(_Δy)

        ΔR = @thunk(Δy ⊠ batched_transpose(x))
        Δt = @thunk(sum(Δy, dims=2))
        Δx = @thunk(batched_transpose(R) ⊠ Δy)

        Δt2 = Tangent{typeof(t2)}(; values=Δt)
        Δt1 = Tangent{typeof(t1)}(; linear=Tangent{typeof(t1.linear)}(; values=ΔR))
        Δrigid = Tangent{typeof(rigid)}(; t2=Δt2, t1=Δt1)

        return NoTangent(), Δrigid, Δx
    end

    return y, transform_pullback
end

function ChainRulesCore.rrule(::typeof(inverse_transform), rigid::RigidTransformations, x::AbstractArray)
    t2, t1 = rigid.t2, rigid.t1
    t, R = values(t2), values(t1)

    z = (x .- t)
    y = batched_transpose(R) ⊠ z

    function inverse_transform_pullback(_Δy)
        Δy = unthunk(_Δy)

        ΔR = @thunk(z ⊠ batched_transpose(Δy))
        Δx = @thunk(R ⊠ Δy)
        Δt = @thunk(-sum(Δx, dims=2)) # t is in the same position as x, but negated and broadcasted

        Δt2 = Tangent{typeof(t2)}(; values=Δt)
        Δt1 = Tangent{typeof(t1)}(; linear=Tangent{typeof(t1.linear)}(; values=ΔR))
        Δrigid = Tangent{typeof(rigid)}(; t2=Δt2, t1=Δt1)

        return NoTangent(), Δrigid, Δx
    end

    return y, inverse_transform_pullback
end

end