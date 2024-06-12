using Functors: functor

@testset "FunctorsExt.jl" begin

    @test !isempty(functor(inverse(rand(Translations, Float32, 3, (1,))))[1])
    @test !isempty(functor(compose(rand(Translations, Float32, 3, (1,)), rand(LinearMaps, 3, 3, (1,))))[1])
    @test !isempty(functor(rand(Translations, Float32, 3, (1,)))[1])
    @test !isempty(functor(rand(LinearMaps, Float32, 3, 3, (1,)))[1])
    @test !isempty(functor(rand(Rotations, Float32, 3, (1,)))[1])

end