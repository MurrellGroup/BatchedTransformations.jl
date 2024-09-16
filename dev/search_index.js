var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = BatchedTransformations","category":"page"},{"location":"#BatchedTransformations","page":"Home","title":"BatchedTransformations","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BatchedTransformations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [BatchedTransformations]","category":"page"},{"location":"#BatchedTransformations.Composed","page":"Home","title":"BatchedTransformations.Composed","text":"Composed{Outer<:Transformation,Inner<:Transformation}\n\nA Composed contains two transformations outer and inner that are composed, where inner gets applied first, and then outer.. It can be constructed with compose(outer, inner) or outer ∘ inner, unless the compose function is overloaded for the specific types.\n\n\n\n\n\n","category":"type"},{"location":"#BatchedTransformations.Identity","page":"Home","title":"BatchedTransformations.Identity","text":"Identity <: Transformation\n\n\n\n\n\n","category":"type"},{"location":"#BatchedTransformations.Inverse","page":"Home","title":"BatchedTransformations.Inverse","text":"Inverse{T<:Transformation} <: Transformation\n\nAn Inverse represents a lazy inverse of a Transformation t.\n\ninverse(t) is a lazy inverse that defaults to inv(t) when evaluated. transform(inverse(t), x) is equivalent to inverse_transform(t, x). This allows for specialized inverse transform implementations that don't require the inverse to be computed explicitly.\n\n\n\n\n\n","category":"type"},{"location":"#BatchedTransformations.Transformation","page":"Home","title":"BatchedTransformations.Transformation","text":"Transformation\n\nAn abstract type whose concrete subtypes contain batches of transformations that can be applied to an array. A Transformation t can be applied to x with transform(t, x), t * x, and t(x).\n\n\n\n\n\n","category":"type"},{"location":"#BatchedTransformations.compose-Tuple{Transformation, Transformation}","page":"Home","title":"BatchedTransformations.compose","text":"compose(t2, t1)\nt2 ∘ t1\n\n\n\n\n\n","category":"method"},{"location":"#BatchedTransformations.transform-Tuple{Transformation, Any}","page":"Home","title":"BatchedTransformations.transform","text":"transform(t, x)\nt * x\nt(x)\n\n\n\n\n\n","category":"method"}]
}
