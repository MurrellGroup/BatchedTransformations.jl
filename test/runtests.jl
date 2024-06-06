using BatchedTransformations
using Test

using ChainRulesTestUtils: test_rrule

# TODO: test other array types (e.g. CuArray)
# TODO: benchmarking

@testset "BatchedTransformations.jl" begin

    include("ext/ChainRulesCoreExt.jl")
    include("ext/FunctorsExt.jl")

    @testset "batched_utils.jl" begin
        r = rand(   Float32, 2, 3, 4, 5)
        z = rand(ComplexF32, 2, 3, 4, 5)

        @testset "batched_transpose" begin
            @test BatchedTransformations.batched_transpose(r) == permutedims(r, (2, 1, 3, 4))
            @test BatchedTransformations.batched_transpose(z) == permutedims(z, (2, 1, 3, 4))
        end

        @testset "batched_adjoint" begin
            @test BatchedTransformations.batched_adjoint(r) == permutedims(r, (2, 1, 3, 4))
            @test_throws MethodError BatchedTransformations.batched_adjoint(z)
        end

    end

    @testset "transformations.jl" begin
        
    end

    @testset "inverse.jl" begin
        m, n = 3, 3 # linear map needs to be square
        batch_size = (2, 4)
        x = rand(Float32, n, 2, batch_size...)

        l = rand(LinearMaps, Float32, m, n, batch_size)
        @test inverse(inverse(l)) === l
        @test inverse(l) ∘ x == inv(l) ∘ x
    end

    @testset "compose.jl" begin
        m, n = 3, 3 # out, in
        batch_size = (2, 4)
        x = rand(Float32, n, 5, batch_size...)

        t = rand(Translations, Float32, m, batch_size)
        l = rand(LinearMaps, Float32, m, n, batch_size)
        c = compose(t, l)
        @test t ∘ l == c
        @test t(l) == c
        @test c ∘ x == t ∘ (l ∘ x)
        @test inv(c) ∘ x == inv(l) ∘ (inv(t) ∘ x)
    end

    @testset "affine.jl" begin
        m, n = 3, 3 # out, in
        batch_size = (2, 4)
        x = rand(Float32, n, 5, batch_size...)

        @testset "LinearMaps" begin
            l = rand(LinearMaps, Float32, m, n, batch_size)
            @test linear(l) isa LinearMaps
            @test values(l) isa AbstractArray
            @test l ∘ x == values(l) ⊠ x
            @test (inv(l) ∘ l) ∘ x ≈ x
            @test inv(l) ∘ (l ∘ x) ≈ x
        end

        @testset "Translations" begin
            t = rand(Translations, Float32, n, batch_size)
            @test translation(t) isa Translations
            @test values(t) isa AbstractArray
            @test t ∘ x == x .+ values(t)
            @test (inv(t) ∘ t) ∘ x ≈ x
            @test inv(t) ∘ (t ∘ x) ≈ x
        end

        @testset "AffineMaps" begin
            affine = rand(AffineMaps, Float32, m, n, batch_size)
            @test linear(affine) isa LinearMaps
            @test translation(affine) isa Translations
            @test affine ∘ x == values(linear(affine)) ⊠ x .+ values(translation(affine))
            @test (inv(affine) ∘ affine) ∘ x ≈ x
            @test inv(affine) ∘ (affine ∘ x) ≈ x
        end

    end

    @testset "rigid.jl" begin
        n = 3
        batch_size = (2, 4)
        x = rand(Float32, n, 5, batch_size...)

        @testset "Rotations" begin
            rotation = rand(Rotations, Float32, n, batch_size)
            @test linear(rotation) isa Rotations
            @test values(rotation) isa AbstractArray
            @test rotation ∘ x == values(linear(rotation)) ⊠ x
            @test (inv(rotation) ∘ rotation) ∘ x ≈ x
            @test inv(rotation) ∘ (rotation ∘ x) ≈ x
        end

        @testset "RigidTransformations" begin
            rigid = rand(RigidTransformations, Float32, n, batch_size)
            @test linear(rigid) isa Rotations
            @test translation(rigid) isa Translations
            @test rigid ∘ x == values(linear(rigid)) ⊠ x .+ values(translation(rigid))
            @test (inv(rigid) ∘ rigid) ∘ x ≈ x
            @test inv(rigid) ∘ (rigid ∘ x) ≈ x

        end

    end

    @testset "rand.jl" begin

        @testset "Rotations" begin
            rotations = rand(Rotations, Float32, 3, (2, 4))
            @test BatchedTransformations.batched_det(values(rotations)) ≈ ones(Float32, 1, 1, 2, 4)
        end

    end

end
