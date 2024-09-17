using BatchedTransformations
using Test

using BatchedTransformations: ⊠
using ChainRulesTestUtils: test_rrule

# TODO: test other array types (e.g. CuArray)
# TODO: benchmarking

@testset "BatchedTransformations.jl" begin

    include("ext/ChainRulesCoreExt.jl")
    include("ext/FunctorsExt.jl")

    @testset "core.jl" begin

        @testset "Transformation" begin
            struct Foo <: Transformation end
            t = Foo()
            x = "Bar"
            @test_throws ErrorException transform(t, x)
            @test_throws ErrorException inv(t)
            @test_throws ErrorException inverse_transform(t, x)
            @test_throws ErrorException t * x
            @test_throws ErrorException t(x)
    
            io = IOBuffer()
            show(io, MIME("text/plain"), t)
            str = String(take!(io))
            @test str == "Foo"
        end

        @testset "Identity" begin
            @test Identity() ∘ Identity() == Identity()
            @test inverse(Identity()) == Identity()
        end

        @testset "Inverse" begin
            n = 3
            batchdims = (2, 4)
            x = rand(Float32, n, 2, batchdims...)

            l = rand(Float32, Translation, n, batchdims)
            @test inverse(inverse(l)) === l
            @test inverse(l) * x == inv(l) * x
        end

        @testset "Composed" begin
            n = 3
            batchdims = (2, 4)
            x = rand(Float32, n, 5, batchdims...)

            t2 = rand(Float32, Translation, n, batchdims)
            t1 = rand(Float32, Translation, n, batchdims)
            c = Composed(t2, t1)
            @test c * x == t2(t1(x))
            @test inv(c)(x) ≈ inv(t2) * (inv(t1) * x)
        end

    end

    @testset "batched.jl" begin

        @testset "affine.jl" begin
            n = 3
            batchdims = (2, 4)
            x = rand(Float32, n, 5, batchdims...)

            @testset "Linear" begin
                l = rand(Float32, Linear, n, batchdims)
                @test linear(l) isa Linear
                @test values(l) isa AbstractArray
                @test l * x == values(l) ⊠ x

                @test_throws ErrorException inv(l)
                invertible_l = Linear{Automorphic}(l)
                @test (inv(invertible_l) ∘ invertible_l) * x ≈ x
                @test inv(invertible_l) * (invertible_l * x) ≈ x

                @test batchsize(l) == batchdims
                @test batchsize(batchreshape(l, 1, batchdims...)) == (1, batchdims...)
                @test batchsize(batchunsqueeze(l, dims=1)) == (1, batchdims...)
            end

            @testset "Translation" begin
                t = rand(Float32, Translation, n, batchdims)
                @test translation(t) isa Translation
                @test values(t) isa AbstractArray
                @test t * x == x .+ values(t)
                @test (inv(t) ∘ t) * x ≈ x
                @test inv(t) * (t * x) ≈ x

                @test batchsize(t) == batchdims
                @test batchsize(batchreshape(t, 1, batchdims...)) == (1, batchdims...)
                @test batchsize(batchunsqueeze(t, dims=1)) == (1, batchdims...)
            end

            @testset "Affine" begin
                affine = rand(Float32, Affine, n, batchdims)
                @test linear(affine) isa Linear
                @test translation(affine) isa Translation
                @test affine * x == values(linear(affine)) ⊠ x .+ values(translation(affine))
                @test (inv(affine) ∘ affine) * x ≈ x
                @test inv(affine) * (affine * x) ≈ x
            end

            n = 3
            batchdims = (2, 4)
            x = rand(Float32, n, 5, batchdims...)

            @testset "Rotation" begin
                rotation = rand(Float32, Rotation, n, batchdims)
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
                rigid = rand(Float32, Rigid, n, batchdims)
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

end
