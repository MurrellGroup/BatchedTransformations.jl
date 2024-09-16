module FunctorsExt

using BatchedTransformations
using Functors: @functor

@functor Inverse
@functor Composed
@functor Linear
@functor Translation
@functor Affine
@functor Rotation

end