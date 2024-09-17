module ChainRulesCoreExt

# maybe move this module to src since ChainRulesCore is light, and gets loaded by NNlib anyway.
# still nice to keep it distinct from the core package though.

using BatchedTransformations
using ChainRulesCore

using BatchedTransformations: batched_mul, batched_mul_T1, batched_mul_T2

function ChainRulesCore.rrule(::typeof(transform), affine::Affine, x::AbstractArray)
    translation, linear = affine.composed.outer, affine.composed.inner
    t, R = values(translation), values(linear)

    y = batched_mul(R, x) .+ t

    function transform_pullback(_Δy)
        Δy = unthunk(_Δy)

        ΔR = @thunk(batched_mul_T2(Δy, x))
        Δt = @thunk(sum(Δy, dims=2))
        Δx = @thunk(batched_mul_T1(R, Δy))

        Δtranslation = Tangent{typeof(translation)}(; values=Δt)
        Δlinear = Tangent{typeof(linear)}(; values=ΔR)
        Δcomposed = Tangent{typeof(affine.composed)}(; outer=Δtranslation, inner=Δlinear)
        Δaffine = Tangent{typeof(affine)}(; composed=Δcomposed)

        return NoTangent(), Δaffine, Δx
    end

    return y, transform_pullback
end

# can probably relax Orthonormal{1} to Orthonormal
function ChainRulesCore.rrule(::typeof(inverse_transform), rigid::Rigid, x::AbstractArray)
    translation, rotation = rigid.composed.outer, rigid.composed.inner
    z = inverse_transform(translation, x) # x .- t
    y = inverse_transform(rotation, z) # R' * (x .- t)
    t, R = values(translation), values(rotation)

    function inverse_transform_pullback(_Δy)
        Δy = unthunk(_Δy)

        ΔR = @thunk(batched_mul_T2(z, Δy))
        Δx = @thunk(batched_mul(R, Δy))
        Δt = @thunk(-sum(Δx, dims=2)) # t is in the same position as x, but negated and broadcasted

        Δtranslation = Tangent{typeof(translation)}(; values=Δt)
        Δrotation = Tangent{typeof(rotation)}(;  values=ΔR)
        Δcomposed = Tangent{typeof(rigid.composed)}(; outer=Δtranslation, inner=Δrotation)
        Δrigid = Tangent{typeof(rigid)}(; composed=Δcomposed)

        return NoTangent(), Δrigid, Δx
    end

    return y, inverse_transform_pullback
end

end