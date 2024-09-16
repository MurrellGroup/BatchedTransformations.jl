using Functors: functor

@testset "FunctorsExt.jl" begin

    @test !isempty(functor(Inverse(rand(Float32, Translation, 3)))[1])
    @test !isempty(functor(Composed(rand(Float32, Translation, 3), rand(Float32, Translation, 3)))[1])
    @test !isempty(functor(rand(Float32, Translation, 3))[1])
    @test !isempty(functor(rand(Float32, Linear, 3 => 3))[1])
    @test !isempty(functor(rand(Float32, Rotation, 3))[1])
    @test !isempty(functor(rand(Float32, Affine, 3))[1])

end