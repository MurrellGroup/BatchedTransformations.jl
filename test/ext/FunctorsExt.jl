using Functors: functor

@testset "FunctorsExt.jl" begin

    @test !isempty(functor(inverse(rand(Float32, Translations, 3, (1,))))[1])
    @test !isempty(functor(compose(rand(Float32, Translations, 3, (1,)), rand(LinearMaps, 3 => 3, (1,))))[1])
    @test !isempty(functor(rand(Float32, Translations, 3, (1,)))[1])
    @test !isempty(functor(rand(Float32, LinearMaps, 3 => 3, (1,)))[1])
    @test !isempty(functor(rand(Float32, Rotations, 3, (1,)))[1])

end