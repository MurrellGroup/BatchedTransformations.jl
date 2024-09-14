module FunctorsExt

using BatchedTransformations
using Functors: @functor

@functor Inverse
@functor Composed
@functor LinearMaps
@functor Rotations
@functor Translations

end