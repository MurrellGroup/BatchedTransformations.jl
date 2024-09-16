using BatchedTransformations
using Test

using BatchedTransformations: ⊠
using ChainRulesTestUtils: test_rrule

# TODO: test other array types (e.g. CuArray)
# TODO: benchmarking

@testset "BatchedTransformations.jl" begin

    include("ext/ChainRulesCoreExt.jl")
    include("ext/FunctorsExt.jl")

    @testset "transformations.jl" begin
        struct FooTransformation{A<:AbstractArray} <: Transformation; values::A end
        t = FooTransformation(rand(Float64, ()))
        x = rand(3, 2, 4)
        @test_throws ErrorException transform(t, x)
        @test_throws ErrorException inv(t)
        @test_throws ErrorException inverse_transform(t, x)
        @test_throws ErrorException t * x
        @test_throws ErrorException t(x)

        io = IOBuffer()
        show(io, MIME("text/plain"), t)
        str = String(take!(io))
        @test str == "FooTransformation{Array{Float64, 0}}"

    end

    @testset "inverse.jl" begin
        m, n = 3, 3 # linear map needs to be square
        batch_size = (2, 4)
        x = rand(Float32, n, 2, batch_size...)

        l = rand(Float32, Linear, n => m, batch_size)
        @test inverse(inverse(l)) === l
        @test inverse(l) * x == inv(l) * x
    end

    @testset "compose.jl" begin
        m, n = 3, 3 # out, in
        batch_size = (2, 4)
        x = rand(Float32, n, 5, batch_size...)

        t = rand(Float32, Translation, m, batch_size)
        l = rand(Float32, Linear, n => m, batch_size)
        c = compose(t, l)
        @test t ∘ l == c
        #@test c(x) == t(l)(x)
        @test c * x == t(l(x))
        @test inv(c)(x) ≈ inv(l) * (inv(t) * x)
    end

    @testset "affine.jl" begin
        m, n = 3, 3 # out, in
        batch_size = (2, 4)
        x = rand(Float32, n, 5, batch_size...)

        @testset "Linear" begin
            l = rand(Float32, Linear, n => m, batch_size)
            @test linear(l) isa Linear
            @test values(l) isa AbstractArray
            @test l * x == values(l) ⊠ x
            @test (inv(l) ∘ l) * x ≈ x
            @test inv(l) * (l * x) ≈ x
        end

        @testset "Translation" begin
            t = rand(Float32, Translation, n, batch_size)
            @test translation(t) isa Translation
            @test values(t) isa AbstractArray
            @test t * x == x .+ values(t)
            @test (inv(t) ∘ t) * x ≈ x
            @test inv(t) * (t * x) ≈ x
        end

        @testset "Affine" begin
            affine = rand(Float32, Affine, n, batch_size)
            @test linear(affine) isa Linear
            @test translation(affine) isa Translation
            @test affine * x == values(linear(affine)) ⊠ x .+ values(translation(affine))
            @test (inv(affine) ∘ affine) * x ≈ x
            @test inv(affine) * (affine * x) ≈ x
        end

        n = 3
        batch_size = (2, 4)
        x = rand(Float32, n, 5, batch_size...)

        @testset "Rotation" begin
            rotation = rand(Float32, Rotation, n, batch_size)
            @test linear(rotation) isa Rotation
            @test values(rotation) isa AbstractArray
            @test rotation * x == values(linear(rotation)) ⊠ x
            @test (inv(rotation) ∘ rotation) * x ≈ x
            @test inv(rotation) * (rotation * x) ≈ x

            # NNlib.batched_transpose only supports one batch dimension
            @test !isa(values(inv(rand(Float32, Rotation, n, (2,)))), Array)
            @test isa(values(inv(rand(Float32, Rotation, n, (2,1)))), Array)
        end

        @testset "Rigid" begin
            rigid = rand(Float32, Rigid, n, batch_size)
            @test linear(rigid) isa Rotation
            @test translation(rigid) isa Translation
            @test rigid * x == values(linear(rigid)) ⊠ x .+ values(translation(rigid))
            @test (inv(rigid) ∘ rigid) * x ≈ x
            @test inv(rigid) * (rigid * x) ≈ x
            @test rigid ∘ rigid isa Rigid
        end

    end

    @testset "rand.jl" begin

        @testset "Rotation" begin
            rotations = rand(Float32, Rotation, 3, (2, 4))
            @test BatchedTransformations.batched_det(values(rotations)) ≈ ones(Float32, 1, 1, 2, 4)
        end

    end

end
