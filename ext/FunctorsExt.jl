module FunctorsExt

using BatchedTransformations
using Functors: @functor

@functor InverseTransformations
@functor ComposedTransformations
@functor LinearMaps
@functor Rotations
@functor Translations

end