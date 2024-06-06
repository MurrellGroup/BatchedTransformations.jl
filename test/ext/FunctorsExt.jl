using Functors: functor

@testset "FunctorsExt.jl" begin

    @test !isempty(functor(inverse(rand(Translations, 3, (1,))))[1])
    @test !isempty(functor(compose(rand(Translations, 3, (1,)), rand(LinearMaps, 3, (1,))))[1])
    @test !isempty(functor(rand(Translations, 3, (1,)))[1])
    @test !isempty(functor(rand(  LinearMaps, 3, (1,)))[1])
    @test !isempty(functor(rand(   Rotations, 3, (1,)))[1])

end