module FunctorsExt

using BatchedTransformations
import Functors: @functor, functor

@functor Inverse
@functor Composed
@functor Translation
@functor Affine

# See:
# https://github.com/FluxML/Functors.jl/issues/40
# https://github.com/FluxML/Functors.jl/issues/6#issuecomment-744895426
function functor(::Type{<:Linear{M}}, x) where M
    reconstruct_linear(xs) = Linear{M}(xs.values)
    return (; values = x.values), reconstruct_linear
end

end